import logging
import sys
import os
import numpy as np
import json
import shutil
import timeit

import pandas as pd
import tensorflow

from maxatac.utilities.constants import (
    TRAIN_MONITOR,
    INPUT_LENGTH,
    INPUT_CHANNELS,
    OUTPUT_LENGTH,
    BP_RESOLUTION,
    MODEL_CONFIG_UPDATE_LIST
)
from maxatac.utilities.system_tools import Mute

with Mute():
    from tensorflow.keras.models import load_model
    from maxatac.utilities.callbacks import get_callbacks
    from maxatac.utilities.training_tools import (
        DataGenerator,
        DataGenerator_v2,
        MaxATACModel,
        ROIPool,
        SeqDataGenerator,
        model_selection,
        save_metadata,
        CHIP_sample_weight_adjustment,
        ValidDataGen,
        DataGen,
        dataset_mapping,
        update_model_config_from_args,
        generate_tfds_files
    )
    from maxatac.utilities.plot import (
        export_binary_metrics,
        export_loss_mse_coeff,
        export_model_structure,
        plot_attention_weights,
    )
    from maxatac.utilities.genome_tools import (
        build_chrom_sizes_dict,
    )



def run_training(args):
    """
    Train a maxATAC model using ATAC-seq and ChIP-seq data

    The primary input to the training function is a meta file that contains all of the information for the locations of
    ATAC-seq signal, ChIP-seq signal, TF, and Cell type.

    Example header for meta file. The meta file must be a tsv file, but the order of the columns does not matter. As
    long as the column names are the same:

    TF | Cell_Type | ATAC_Signal_File | Binding_File | ATAC_Peaks | ChIP_peaks

    ## An example meta file is included in our repo

    _________________
    Workflow Overview

    1) Set up the directories and filenames
    2) Initialize the model based on the desired architectures
    3) Read in training and validation pools
    4) Initialize the training and validation generators
    5) Fit the models with the specific parameters

    :params args: arch, seed, output, prefix, output_activation, lrate, decay, weights,
    dense, batch_size, val_batch_size, train roi, validate roi, meta_file, sequence, average, threads, epochs, batches,
    tchroms, vchroms, shuffle_cell_type, rev_comp

    :returns: Trained models saved after each epoch
    """
    logging.info(args)

    gpus = tensorflow.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tensorflow.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tensorflow.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)

    if tensorflow.test.gpu_device_name():
        print("GPU device found")
    else:
        print("No GPU found")

    # Start Timer
    startTime = timeit.default_timer()

    logging.info("Set up model parameters")

    # Read model config
    with open(args.model_config, "r") as f:
        model_config = json.load(f)

    model_config=update_model_config_from_args(model_config,args,MODEL_CONFIG_UPDATE_LIST)

    # Initialize the model with the architecture of choice
    maxatac_model = MaxATACModel(
        arch=args.arch,
        seed=args.seed,
        model_config=model_config,
        output_directory=args.output,
        prefix=args.prefix,
        threads=args.threads,
        meta_path=args.meta_file,
        output_activation=args.output_activation,
        dense=args.dense,
        weights=args.weights,
        inter_fusion=model_config["INTER_FUSION"],
    )

    # export model structure
    export_model_structure(
        maxatac_model.nn_model, maxatac_model.results_location, ext=".pdf"
    )

    logging.info("Import training regions")

    # Import training regions
    train_examples = ROIPool(
        chroms=args.tchroms,
        roi_file_path=args.train_roi,
        meta_file=args.meta_file,
        prefix=args.prefix,
        output_directory=maxatac_model.output_directory,
        shuffle=True,
        tag="training",
    )

    # Import validation regions
    validate_examples = ROIPool(
        chroms=args.vchroms,
        roi_file_path=args.validate_roi,
        meta_file=args.meta_file,
        prefix=args.prefix,
        output_directory=maxatac_model.output_directory,
        shuffle=True,
        tag="validation",
    )

    if args.ATAC_Sampling_Multiplier != 0:
        steps_per_epoch_v2 = int(
            train_examples.ROI_pool_CHIP.shape[0]
            * maxatac_model.meta_dataframe[
                maxatac_model.meta_dataframe["Train_Test_Label"] == "Train"
            ].shape[0]
            // np.ceil((args.batch_size / (1.0 + float(args.ATAC_Sampling_Multiplier))))
        )
        validation_steps_v2 = int(
            validate_examples.ROI_pool.shape[0] // args.batch_size
        )

        # override max epoch when training sample upper bound is available
        if args.training_sample_upper_bound != 0:
            args.epochs = int(
                min(
                    args.epochs,
                    int(
                        args.training_sample_upper_bound
                        // (steps_per_epoch_v2 * args.batch_size)
                    ),
                )
            )

        # annotate CHIP ROI with additional sample weight adjustment
        train_examples.ROI_pool_CHIP = CHIP_sample_weight_adjustment(
            train_examples.ROI_pool_CHIP
        )
        validate_examples.ROI_pool_CHIP = CHIP_sample_weight_adjustment(
            validate_examples.ROI_pool_CHIP
        )

    logging.info("Initialize data generator")

    # If tfds files need to be generated
    if args.get_tfds:
        generate_tfds_files(
            args, maxatac_model, train_examples, validate_examples, model_config
        )
        logging.info("Generating tfds files completed!")
        sys.exit()


    # Specify max_que_size
    if args.max_queue_size:
        queue_size = int(args.max_queue_size)
        logging.info("User specified Max Queue Size: " + str(queue_size))
    else:
        queue_size = args.threads * 2
        logging.info("Max Queue Size found: " + str(queue_size))

    # get tfds train and valid object
    data_meta = pd.read_csv(args.tfds_meta, header=0, sep="\t")

    # train data
    # atac
    atac_tfds_file = data_meta[
        (data_meta["train or valid"] == "train") & (data_meta["roi_type"] == "ATAC")
    ]["path"].values[0]
    atac_tfds = tensorflow.data.Dataset.load(
        atac_tfds_file,
        compression="GZIP",
    )

    # chip
    chip_tfds = []
    for cell_type, file_path in data_meta[
        (data_meta["train or valid"] == "train") & (data_meta["roi_type"] == "CHIP")
    ][["cell_type", "path"]].values:
        if (
            maxatac_model.meta_dataframe[
                maxatac_model.meta_dataframe["Cell_Line"] == cell_type
            ]["Train_Test_Label"].values[0]
            == "Train"
        ):
            tfds = tensorflow.data.Dataset.load(
                file_path,
                compression="GZIP",
            )
            data = tfds
            chip_tfds.append(data)

    _chip_size = len(chip_tfds)
    _chip_prob = 1.0 / (1.0 + float(args.ATAC_Sampling_Multiplier))
    _atac_prob = 1.0 - _chip_prob

    # vstack
    train_data_chip = chip_tfds[0]
    if len(chip_tfds) > 1:
        for k in range(1, len(chip_tfds)):
            train_data_chip = train_data_chip.concatenate(chip_tfds[k])

    # re-assign steps_per_epoch_v2 here
    steps_per_epoch_v2 = int(train_data_chip.cardinality().numpy() // np.ceil(
        (args.batch_size / (1.0 + float(args.ATAC_Sampling_Multiplier))))
    )

    train_data = (
        tensorflow.data.Dataset.sample_from_datasets(
            [
                train_data_chip.cache()
                .map(
                    map_func=dataset_mapping[args.SHUFFLE_AUGMENTATION],
                    num_parallel_calls=tensorflow.data.AUTOTUNE,
                )
                .shuffle(train_data_chip.cardinality().numpy())
                .repeat(args.epochs),
                atac_tfds.cache()
                .map(
                    map_func=dataset_mapping[args.SHUFFLE_AUGMENTATION],
                    num_parallel_calls=tensorflow.data.AUTOTUNE,
                )
                .shuffle(atac_tfds.cardinality().numpy())
                .repeat(args.epochs),
            ],
            weights=[_chip_prob, _atac_prob],
            stop_on_empty_dataset=False,
            rerandomize_each_iteration=False,
        )
        .batch(
            batch_size=args.batch_size,
            num_parallel_calls=tensorflow.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=False,
        )
        .prefetch(tensorflow.data.AUTOTUNE)
    )

    # valid data
    valid_tfds_file = data_meta[data_meta["train or valid"] == "valid"]["path"].values[
        0
    ]
    valid_tfds = tensorflow.data.Dataset.load(
        valid_tfds_file,
        compression="GZIP",
    )
    valid_data = (
        valid_tfds.take(
            (validate_examples.ROI_pool.shape[0] // args.batch_size) * args.batch_size
        )
        .cache()
        .map(
            map_func=dataset_mapping["peak_centric"]
            if args.SHUFFLE_AUGMENTATION != "no_map"
            else dataset_mapping[args.SHUFFLE_AUGMENTATION],
            num_parallel_calls=tensorflow.data.AUTOTUNE,
        )  # whether to use non-shuffle validation
        .repeat(args.epochs)
        .batch(
            batch_size=args.batch_size,
            num_parallel_calls=tensorflow.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=False,
        )
        .prefetch(tensorflow.data.AUTOTUNE)
    )

    # Save metadata
    save_metadata(
        args.output,
        args,
        model_config,
        extra={
            "training CHIP ROI total regions": train_examples.ROI_pool_CHIP.shape[
                0
            ],
            "training ATAC ROI total regions": train_examples.ROI_pool_ATAC.shape[
                0
            ],
            "validate CHIP ROI total regions": validate_examples.ROI_pool_CHIP.shape[
                0
            ],
            "validate ATAC ROI total regions": validate_examples.ROI_pool_ATAC.shape[
                0
            ],
            "training CHIP ROI unique regions": train_examples.ROI_pool_unique_region_size_CHIP,
            "training ATAC ROI unique regions": train_examples.ROI_pool_unique_region_size_ATAC,
            "validate CHIP ROI unique regions": validate_examples.ROI_pool_unique_region_size_CHIP,
            "validate ATAC ROI unique regions": validate_examples.ROI_pool_unique_region_size_ATAC,
            "batch size": args.batch_size,
            "training batches per epoch": steps_per_epoch_v2,
            "validation batches per epoch": validation_steps_v2,
            "total epochs": args.epochs,
            "ATAC_Sampling_Multiplier": args.ATAC_Sampling_Multiplier,
            "CHIP_Sample_Weight_Baseline": args.CHIP_Sample_Weight_Baseline,
        },
    )

    # Fit the model
    training_history = maxatac_model.nn_model.fit(
        train_data,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch_v2,
        validation_data=valid_data,
        validation_steps=validation_steps_v2,
        callbacks=get_callbacks(
            model_location=maxatac_model.results_location,
            log_location=maxatac_model.log_location,
            tensor_board_log_dir=maxatac_model.tensor_board_log_dir,
            monitor=TRAIN_MONITOR,
            reduce_lr_on_plateau=args.reduce_lr_on_plateau,
        ),
        max_queue_size=queue_size,
        use_multiprocessing=False,
        workers=1,
        verbose=1,
    )

    logging.info("Plot and save results")

    # Select best model
    best_epoch = model_selection(
        training_history=training_history, output_dir=maxatac_model.output_directory
    )

    if args.plot:
        tf = maxatac_model.train_tf
        TCL = "_".join(maxatac_model.cell_types)
        ARC = args.arch
        RR = args.rand_ratio

        export_binary_metrics(
            training_history, tf, RR, ARC, maxatac_model.results_location, best_epoch
        )

    logging.info("Results are saved to: " + maxatac_model.results_location)

    # Measure End Time of Training
    stopTime = timeit.default_timer()
    totalTime = stopTime - startTime

    # Output running time in a nice format.
    mins, secs = divmod(totalTime, 60)
    hours, mins = divmod(mins, 60)

    logging.info("Total training time: %d:%d:%d.\n" % (hours, mins, secs))

    sys.exit()