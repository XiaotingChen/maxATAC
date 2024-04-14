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
    MODEL_CONFIG_UPDATE_LIST,
    DNA_INPUT_CHANNELS
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
        generate_tfds_files,
        get_tfds_data,
        model_selection_v2,
    )
    from maxatac.utilities.plot import (
        export_binary_metrics,
        export_loss_mse_coeff,
        export_model_structure,
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

    model_config = update_model_config_from_args(
        model_config, args, MODEL_CONFIG_UPDATE_LIST
    )

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
        deterministic=args.DETERMINISTIC,
    )

    # export model structure
    #export_model_structure(
    #    maxatac_model.nn_model, maxatac_model.results_location, ext=".pdf"
    #)

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

    logging.info("Import validation regions")
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

    if args.ATAC_SAMPLING_MULTIPLIER != 0:
        train_steps_per_epoch_v2 = int(
            train_examples.ROI_pool_CHIP.shape[0]
            * maxatac_model.meta_dataframe[
                maxatac_model.meta_dataframe["Train_Test_Label"] == "Train"
            ].shape[0]
            // np.ceil((args.batch_size / (1.0 + float(args.ATAC_SAMPLING_MULTIPLIER))))
        )
        valid_steps_per_epoch_v2 = int(
            validate_examples.ROI_pool.shape[0] // args.batch_size
        )

        # override max epoch when training sample upper bound is available
        if args.TRAINING_SAMPLE_UPPER_BOUND != 0:
            args.epochs = int(
                min(
                    args.epochs,
                    int(
                        args.TRAINING_SAMPLE_UPPER_BOUND
                        // (train_steps_per_epoch_v2 * args.batch_size)
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

    # If tfds files need to be generated
    if args.GET_TFDS:
        logging.info("Initialize data generation")
        # Determine how many input channels there are
        meta_file = pd.read_csv(args.meta_file, sep="\t")
        num_signal_cols = len(
            [i for i in meta_file.columns.tolist() if "Signal_File" in i]
        )
        generate_tfds_files(
            args, maxatac_model, train_examples, validate_examples, model_config, num_channels=DNA_INPUT_CHANNELS+num_signal_cols
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

    logging.info("Loading data cache for training")
    # get tfds train and valid object
    data_meta = pd.read_csv(args.TFDS_META, header=0, sep="\t")

    train_data_atac = get_tfds_data(data_meta, maxatac_model, "train", "ATAC")
    train_data_chip = get_tfds_data(data_meta, maxatac_model, "train", "CHIP")
    valid_data_atac = get_tfds_data(data_meta, maxatac_model, "valid", "ATAC")
    valid_data_chip = get_tfds_data(data_meta, maxatac_model, "valid", "CHIP")
    valid_data_combined = valid_data_chip.concatenate(valid_data_atac)
    # chip first to avoid positive sample truncation

    # re-assign train_steps_per_epoch_v2 here
    train_steps_per_epoch_v2 = int(
        train_data_chip.cardinality().numpy()
        // np.ceil((args.batch_size / (1.0 + float(args.ATAC_SAMPLING_MULTIPLIER))))
    )
    _chip_prob = 1.0 / (1.0 + float(args.ATAC_SAMPLING_MULTIPLIER))
    _atac_prob = 1.0 - _chip_prob

    repeat_scale = 1
    if (
        train_data_chip.cardinality().numpy() * float(args.ATAC_SAMPLING_MULTIPLIER)
        > train_data_atac.cardinality().numpy()
    ):
        repeat_scale = int(
            np.ceil(
                train_data_chip.cardinality().numpy()
                * float(args.ATAC_SAMPLING_MULTIPLIER)
                / train_data_atac.cardinality().numpy()
            )
        )

    train_data = (
        tensorflow.data.Dataset.sample_from_datasets(
            [
                train_data_chip.cache()
                .map(
                    map_func=dataset_mapping[args.SHUFFLE_AUGMENTATION],
                    num_parallel_calls=args.threads
                    if args.DETERMINISTIC
                    else tensorflow.data.AUTOTUNE,
                    deterministic=args.DETERMINISTIC,
                )
                .shuffle(
                    train_data_chip.cardinality().numpy(),
                    seed=args.seed + 1 if args.DETERMINISTIC else None,
                )
                .repeat(args.epochs),
                train_data_atac.cache()
                .map(
                    map_func=dataset_mapping[args.SHUFFLE_AUGMENTATION],
                    num_parallel_calls=args.threads
                    if args.DETERMINISTIC
                    else tensorflow.data.AUTOTUNE,
                    deterministic=args.DETERMINISTIC,
                )
                .shuffle(
                    train_data_atac.cardinality().numpy(),
                    seed=args.seed + 2 if args.DETERMINISTIC else None,
                )
                .repeat(args.epochs * repeat_scale),
            ],
            weights=[_chip_prob, _atac_prob],
            stop_on_empty_dataset=False,
            rerandomize_each_iteration=False,
            seed=args.seed + 3 if args.DETERMINISTIC else None,
        )
        .batch(
            batch_size=args.batch_size,
            num_parallel_calls=args.threads
            if args.DETERMINISTIC
            else tensorflow.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=args.DETERMINISTIC,
        )
        .prefetch(tensorflow.data.AUTOTUNE)
    )

    valid_data = (
        valid_data_combined.take(
            (valid_data_combined.cardinality().numpy() // args.batch_size)
            * args.batch_size
        )
        .cache()
        .map(
            map_func=dataset_mapping["peak_centric"]
            if args.SHUFFLE_AUGMENTATION != "no_map"
            else dataset_mapping[args.SHUFFLE_AUGMENTATION],
            num_parallel_calls=args.threads
            if args.DETERMINISTIC
            else tensorflow.data.AUTOTUNE,
            deterministic=args.DETERMINISTIC,
        )  # whether to use non-shuffle validation
        .repeat(args.epochs)
        .batch(
            batch_size=args.batch_size,
            num_parallel_calls=args.threads
            if args.DETERMINISTIC
            else tensorflow.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=args.DETERMINISTIC,
        )
        .prefetch(tensorflow.data.AUTOTUNE)
    )

    logging.info("Logging meta info")
    # Save metadata
    save_metadata(
        args.output,
        args,
        model_config,
        extra={
            "all listed cell types": maxatac_model.cell_types,
            "training cell types": train_examples.used_cell_types,
            "training CHIP ROI total regions": "{} | {}".format(
                train_examples.ROI_pool_CHIP.shape[0],
                train_data_chip.cardinality().numpy(),
            ),
            "training ATAC ROI total regions": "{} | {}".format(
                train_examples.ROI_pool_ATAC.shape[0],
                train_data_atac.cardinality().numpy(),
            ),
            "validate CHIP ROI total regions": "{} | {}".format(
                validate_examples.ROI_pool_CHIP.shape[0],
                valid_data_chip.cardinality().numpy(),
            ),
            "validate ATAC ROI total regions": "{} | {}".format(
                validate_examples.ROI_pool_ATAC.shape[0],
                valid_data_atac.cardinality().numpy(),
            ),
            "training CHIP ROI unique regions": train_examples.ROI_pool_unique_region_size_CHIP,
            "training ATAC ROI unique regions": train_examples.ROI_pool_unique_region_size_ATAC,
            "validate CHIP ROI unique regions": validate_examples.ROI_pool_unique_region_size_CHIP,
            "validate ATAC ROI unique regions": validate_examples.ROI_pool_unique_region_size_ATAC,
            "batch size": args.batch_size,
            "train batches per epoch": train_steps_per_epoch_v2,
            "valid batches per epoch": valid_steps_per_epoch_v2,
            "total epochs": args.epochs,
            "ATAC_SAMPLING_MULTIPLIER": args.ATAC_SAMPLING_MULTIPLIER,
            "CHIP_SAMPLE_WEIGHT_BASELINE": args.CHIP_SAMPLE_WEIGHT_BASELINE,
        },
    )

    logging.info("Start model fitting")
    # Fit the model
    training_history = maxatac_model.nn_model.fit(
        train_data,
        epochs=args.epochs,
        steps_per_epoch=train_steps_per_epoch_v2,
        validation_data=valid_data,
        validation_steps=valid_steps_per_epoch_v2,
        callbacks=get_callbacks(
            model_location=maxatac_model.results_location,
            log_location=maxatac_model.log_location,
            tensor_board_log_dir=maxatac_model.tensor_board_log_dir,
            monitor=TRAIN_MONITOR,
            reduce_lr_on_plateau=args.REDUCE_LR_ON_PLATEAU,
        ),
        max_queue_size=queue_size,
        use_multiprocessing=False,
        workers=1,
        verbose=1,
    )

    logging.info("Plot and save results")

    # Select best model
    best_epoch = model_selection_v2(
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
