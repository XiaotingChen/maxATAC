import logging
import sys
import timeit

from keras.utils.data_utils import OrderedEnqueuer

from maxatac.utilities.constants import TRAIN_MONITOR, INPUT_LENGTH
from maxatac.utilities.system_tools import Mute

with Mute():
    from maxatac.utilities.callbacks import get_callbacks
    from maxatac.utilities.training_tools import DataGenerator, MaxATACModel, ROIPool, SeqDataGenerator, model_selection
    from maxatac.utilities.plot import export_binary_metrics, export_loss_mse_coeff, export_model_structure

from maxatac.utilities.constants import KERNEL_INITIALIZER, INPUT_LENGTH, INPUT_CHANNELS, INPUT_FILTERS, \
    INPUT_KERNEL_SIZE, INPUT_ACTIVATION, OUTPUT_FILTERS, OUTPUT_KERNEL_SIZE, FILTERS_SCALING_FACTOR, DILATION_RATE, \
    OUTPUT_LENGTH, CONV_BLOCKS, PADDING, POOL_SIZE, ADAM_BETA_1, ADAM_BETA_2, DEFAULT_ADAM_LEARNING_RATE, \
    DEFAULT_ADAM_DECAY, BATCH_SIZE, DEFAULT_TRAIN_EPOCHS, DEFAULT_LEARNING_RATE,DEFAULT_EXPONENTIALDECAY_DECAY_RATE, \
    DEFAULT_EXPONENTIALDECAY_DECAY_STEPS, DEFAULT_COSINEDECAYRESTART_FIRST_DECAY_STEPS,DEFAULT_COSINEDECAYRESTART_ALPHA, \
    DEFAULT_COSINEDECAYRESTART_T_MUL, DEFAULT_COSINEDECAYRESTART_M_MUL

import wandb
from wandb.keras import WandbMetricsLogger

def run_hparam_optim(args):
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
    tchroms, vchroms, shuffle_cell_type, rev_comp, multiprocessing, max_que_size

    :returns: Trained models saved after each epoch
    """
    # # Start Timer
    # startTime = timeit.default_timer()
    #
    # logging.info(f"Training Parameters:\n" +
    #               f"Architecture: {args.arch} \n" +
    #               f"Filename prefix: {args.prefix} \n" +
    #               f"Output directory: {args.output} \n" +
    #               f"Meta file: {args.meta_file} \n" +
    #               f"Output activation: {args.output_activation} \n" +
    #               f"Number of threads: {args.threads} \n" +
    #               f"Use dense layer?: {args.dense} \n" +
    #               f"Training ROI file (if provided): {args.train_roi} \n" +
    #               f"Validation ROI file (if provided): {args.validate_roi} \n" +
    #               f"2bit sequence file: {args.sequence} \n" +
    #               "Restricting to chromosomes: \n   - " + "\n   - ".join(args.chroms) + "\n" +
    #               "Restricting training to chromosomes: \n   - " + "\n   - ".join(args.tchroms) + "\n" +
    #               "Restricting validation to chromosomes: \n   - " + "\n   - ".join(args.vchroms) + "\n" +
    #               f"Number of batches: {args.batches} \n" +
    #               f"Number of examples per batch: {args.batch_size} \n" +
    #               f"Proportion of examples drawn randomly: {args.rand_ratio} \n" +
    #               f"Shuffle training regions amongst cell types: {args.shuffle_cell_type} \n" +
    #               f"Train with the reverse complement sequence: {args.rev_comp} \n" +
    #               f"Number of epochs: {args.epochs} \n" +
    #               f"Use multiprocessing?: {args.multiprocessing} \n" +
    #               f"Max number of workers to queue: {args.max_queue_size} \n"
    #               )

    # define wandb sweep
    sweep_configuration = {
        'method': 'bayes',
        'name': args.wandb_name,
        'metric': {
            'goal': 'minimize',
            'name': 'val_loss'
        },
        'parameters': {
            'input_filters': {
                "max": 25,
                "min": 10,
            },
            "input_kernel_size": {
                "max": 15,
                "min": 4,
            },
            "epoch": {
                "max": args.wandb_epoch_max,
                "min": args.wandb_epoch_min,
            },
            "default_learning_rate": {
                "values": [5e-3, 1e-3, 1e-4],
            },
            "optimizer": {
                "values": ['adam', 'sgd', 'adamw'],
            },
            "lr_schedule": {
                "values": ['None', 'exponential_decay', 'cosine_decay_restarts'],
            },
        },
        'run_cap': args.wandb_count,
    }

    def agent_func():
        # initialize wandb
        wandb.init(
            # project="maxATAC_hparam_BO",
            # track hyperparameters and run metadata with wandb.config
            group=args.wandb_group_name,
            config={
                'adam_learning_rate': DEFAULT_ADAM_LEARNING_RATE,
                'adam_decay': DEFAULT_ADAM_DECAY,
                'input_length': INPUT_LENGTH,
                'input_channels': INPUT_CHANNELS,
                'input_filters': INPUT_FILTERS,
                'input_kernel_size': INPUT_KERNEL_SIZE,
                'input_activation': INPUT_ACTIVATION,
                'output_filters': OUTPUT_FILTERS,
                'output_kernel_size': OUTPUT_KERNEL_SIZE,
                'filters_scaling_factor': FILTERS_SCALING_FACTOR,
                'dilation_rate': DILATION_RATE,
                'output_length': OUTPUT_LENGTH,
                'conv_blocks': CONV_BLOCKS,
                'padding': PADDING,
                'pool_size': POOL_SIZE,
                'adam_beta_1': ADAM_BETA_1,
                'adam_beta_2': ADAM_BETA_2,
                'batch_size': BATCH_SIZE,
                'epoch': DEFAULT_TRAIN_EPOCHS,
                'optimizer': 'adam',
                'lr_schedule': 'None',
                'default_learning_rate': DEFAULT_LEARNING_RATE,
                'exponential_decay_decay_rate': DEFAULT_EXPONENTIALDECAY_DECAY_RATE,
                'exponential_decay_decay_steps': DEFAULT_EXPONENTIALDECAY_DECAY_STEPS,
                'cosine_decay_restart_first_decay_steps': DEFAULT_COSINEDECAYRESTART_FIRST_DECAY_STEPS,
                'cosine_decay_restart_alpha': DEFAULT_COSINEDECAYRESTART_ALPHA,
                'cosine_decay_restart_t_mul': DEFAULT_COSINEDECAYRESTART_T_MUL,
                'cosine_decay_restart_m_mul': DEFAULT_COSINEDECAYRESTART_M_MUL,
            }
        )

        # Initialize the model with the architecture of choice
        maxatac_model = MaxATACModel(arch=args.arch,
                                     seed=args.seed,
                                     output_directory=args.output,
                                     prefix=args.prefix,
                                     threads=args.threads,
                                     meta_path=args.meta_file,
                                     output_activation=args.output_activation,
                                     dense=args.dense,
                                     weights=args.weights,
                                     hparam_optim=True,
                                     wandb_config=wandb.config
                                     )

        logging.info("Import training regions")

        # Import training regions
        train_examples = ROIPool(chroms=args.tchroms,
                                 roi_file_path=args.train_roi,
                                 meta_file=args.meta_file,
                                 prefix=args.prefix,
                                 output_directory=maxatac_model.output_directory,
                                 blacklist=args.blacklist,
                                 region_length=INPUT_LENGTH,
                                 chrom_sizes_file=args.chrom_sizes
                                 )

        # Import validation regions
        validate_examples = ROIPool(chroms=args.vchroms,
                                    roi_file_path=args.validate_roi,
                                    meta_file=args.meta_file,
                                    prefix=args.prefix,
                                    output_directory=maxatac_model.output_directory,
                                    blacklist=args.blacklist,
                                    region_length=INPUT_LENGTH,
                                    chrom_sizes_file=args.chrom_sizes
                                    )

        logging.info("Initialize training data generator")

        # Initialize the training generator
        train_gen = DataGenerator(sequence=args.sequence,
                                  meta_table=maxatac_model.meta_dataframe,
                                  roi_pool=train_examples.ROI_pool,
                                  cell_type_list=maxatac_model.cell_types,
                                  rand_ratio=args.rand_ratio,
                                  chroms=args.tchroms,
                                  batch_size=args.batch_size,
                                  shuffle_cell_type=args.shuffle_cell_type,
                                  rev_comp_train=args.rev_comp
                                  )

        # Create keras.utils.sequence object from training generator
        seq_train_gen = SeqDataGenerator(batches=args.batches, generator=train_gen)

        # Specify max_que_size
        if args.max_queue_size:
            queue_size = int(args.max_queue_size)
            logging.info("User specified Max Queue Size: " + str(queue_size))
        else:
            queue_size = args.threads * 2
            logging.info("Max Queue Size found: " + str(queue_size))

        # Builds a Enqueuer from a Sequence.
        # Specify multiprocessing
        if args.multiprocessing:
            logging.info("Training with multiprocessing")
            train_gen_enq = OrderedEnqueuer(seq_train_gen, use_multiprocessing=True)
            train_gen_enq.start(workers=args.threads, max_queue_size=queue_size)

        else:
            logging.info("Training without multiprocessing")
            train_gen_enq = OrderedEnqueuer(seq_train_gen, use_multiprocessing=False)
            train_gen_enq.start(workers=1, max_queue_size=queue_size)

        enq_train_gen = train_gen_enq.get()

        logging.info("Initialize validation data generator")

        # Initialize the validation generator
        val_gen = DataGenerator(sequence=args.sequence,
                                meta_table=maxatac_model.meta_dataframe,
                                roi_pool=validate_examples.ROI_pool,
                                cell_type_list=maxatac_model.cell_types,
                                rand_ratio=args.rand_ratio,
                                chroms=args.vchroms,
                                batch_size=args.batch_size,
                                shuffle_cell_type=args.shuffle_cell_type,
                                rev_comp_train=args.rev_comp
                                )

        # Create keras.utils.sequence object from validation generator
        seq_validate_gen = SeqDataGenerator(batches=args.batches, generator=val_gen)

        # Builds a Enqueuer from a Sequence.
        # Specify multiprocessing
        if args.multiprocessing:
            logging.info("Training with multiprocessing")
            val_gen_enq = OrderedEnqueuer(seq_validate_gen, use_multiprocessing=True)
            val_gen_enq.start(workers=args.threads, max_queue_size=queue_size)
        else:
            logging.info("Training without multiprocessing")
            val_gen_enq = OrderedEnqueuer(seq_validate_gen, use_multiprocessing=False)
            val_gen_enq.start(workers=1, max_queue_size=queue_size)

        enq_val_gen = val_gen_enq.get()


        logging.info("Fit model")

        # Fit the model
        training_history = maxatac_model.nn_model.fit(enq_train_gen,
                                                    validation_data=enq_val_gen,
                                                    steps_per_epoch=args.batches,
                                                    validation_steps=args.batches,
                                                    epochs=wandb.config.epoch,
                                                    callbacks=[
                                                        WandbMetricsLogger(log_freq=5),
                                                    ],
                                                    # callbacks=get_callbacks(
                                                    #     model_location=maxatac_model.results_location,
                                                    #     log_location=maxatac_model.log_location,
                                                    #     tensor_board_log_dir=maxatac_model.tensor_board_log_dir,
                                                    #     monitor=TRAIN_MONITOR
                                                    #     ),
                                                    max_queue_size=10,
                                                    use_multiprocessing=False,
                                                    workers=1,
                                                    verbose=1
                                                    )
        wandb.log(
            vars(args)
        )

        # logging.info("Plot and save results")
        #
        # # Select best model
        # best_epoch = model_selection(training_history=training_history,
        #                              output_dir=maxatac_model.output_directory)
        #
        # # If plot then plot the model structure and training metrics
        # if args.plot:
        #     tf = maxatac_model.train_tf
        #     TCL = '_'.join(maxatac_model.cell_types)
        #     ARC = args.arch
        #     RR = args.rand_ratio
        #
        #     export_model_structure(maxatac_model.nn_model, maxatac_model.results_location)
        #
        #     export_binary_metrics(training_history, tf, RR, ARC, maxatac_model.results_location, best_epoch)
        #
        # # If save_roi save the ROI files
        # if args.save_roi:
        #     # Write the ROI pools
        #     train_examples.write_data(prefix=args.prefix, output_dir=maxatac_model.output_directory, set_tag="training")
        #     validate_examples.write_data(prefix=args.prefix, output_dir=maxatac_model.output_directory,
        #                                  set_tag="validation")
        #
        # logging.info("Results are saved to: " + maxatac_model.results_location)
        #
        # # Measure End Time of Training
        # stopTime = timeit.default_timer()
        # totalTime = stopTime - startTime
        #
        # # Output running time in a nice format.
        # mins, secs = divmod(totalTime, 60)
        # hours, mins = divmod(mins, 60)
        #
        # logging.info("Total training time: %d:%d:%d.\n" % (hours, mins, secs))


        #sys.exit()

    # start sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.wandb_proj_name,)
    wandb.agent(sweep_id=sweep_id, function=agent_func, count=args.wandb_count)
