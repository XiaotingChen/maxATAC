import argparse
import random
from os import getcwd

from yaml import dump

from maxatac.utilities.system_tools import (get_version,
                                            get_absolute_path,
                                            get_cpu_count,
                                            Mute
                                            )

with Mute():
    from maxatac.analyses.average import run_averaging
    from maxatac.analyses.predict import run_prediction
    from maxatac.analyses.train import run_training
    from maxatac.analyses.normalize import run_normalization
    from maxatac.analyses.benchmark import run_benchmarking
    from maxatac.analyses.prediction_signal import run_prediction_signal
    from maxatac.utilities.genome_tools import load_bigwig, load_2bit
    from maxatac.analyses.peaks import call_peaks

from maxatac.utilities.constants import (DEFAULT_TRAIN_VALIDATE_CHRS,
                                         LOG_LEVELS,
                                         DEFAULT_LOG_LEVEL,
                                         DEFAULT_TRAIN_EPOCHS,
                                         DEFAULT_TRAIN_BATCHES_PER_EPOCH,
                                         DEFAULT_MIN_PREDICTION,
                                         BATCH_SIZE,
                                         VAL_BATCH_SIZE,
                                         INPUT_LENGTH,
                                         DEFAULT_TRAIN_CHRS,
                                         DEFAULT_VALIDATE_CHRS,
                                         DEFAULT_CHROM_SIZES,
                                         BLACKLISTED_REGIONS,
                                         DEFAULT_VALIDATE_RAND_RATIO,
                                         DEFAULT_ROUND,
                                         DEFAULT_TEST_CHRS,
                                         BLACKLISTED_REGIONS_BIGWIG,
                                         DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
                                         DEFAULT_BENCHMARKING_BIN_SIZE,
                                         ALL_CHRS, AUTOSOMAL_CHRS
                                         )


def normalize_args(args, skip_list=[], cwd_abs_path=None):
    """
    Converts all relative path arguments to absolute
    ones relatively to the cwd_abs_path or current working directory.
    Skipped arguments and None will be returned unchanged.
    """
    cwd_abs_path = getcwd() if cwd_abs_path is None else cwd_abs_path
    normalized_args = {}
    for key, value in args.__dict__.items():
        if key not in skip_list and value is not None:
            if isinstance(value, list):
                for v in value:
                    normalized_args.setdefault(key, []).append(
                        get_absolute_path(v, cwd_abs_path)
                    )
            else:
                normalized_args[key] = get_absolute_path(value, cwd_abs_path)
        else:
            normalized_args[key] = value
    return argparse.Namespace(**normalized_args)


def get_synced_chroms(chroms, files, ignore_regions=None):
    """
    If ignore_regions is True, set regions to the whole chromosome length
    Returns something like this
        {
            "chr2": {"length": 243199373, "region": [0, 243199373]},
            "chr3": {"length": 198022430, "region": [0, 198022430]}
        }
    """

    chroms_and_regions = {}
    for chrom in chroms:
        chrom_name, *region = chrom.replace(",", "").split(":")  # region is either [] or ["start-end", ...]
        chroms_and_regions[chrom_name] = None
        if not ignore_regions:
            try:
                chroms_and_regions[chrom_name] = [int(i) for i in region[0].split("-")]
            except (IndexError, ValueError):
                pass

    loaded_chroms = set()

    for file in [f for f in files if f is not None]:
        try:
            with load_2bit(file) as data_stream:
                avail_chroms = set([(k, v) for k, v in data_stream.chroms().items()])
        except RuntimeError:
            with load_bigwig(file) as data_stream:
                avail_chroms = set([(k, v) for k, v in data_stream.chroms().items()])
        loaded_chroms = loaded_chroms.intersection(
            avail_chroms) if loaded_chroms else avail_chroms  # checks both chrom_name and chrom_length are the same

    synced_chroms = {}
    for chrom_name, chrom_length in loaded_chroms:
        if chrom_name not in chroms_and_regions: continue
        region = chroms_and_regions[chrom_name]
        if not region or \
                region[0] < 0 or \
                region[1] <= 0 or \
                region[0] >= region[1] or \
                region[1] > chrom_length:
            region = [0, chrom_length]
        synced_chroms[chrom_name] = {
            "length": chrom_length,
            "region": region
        }

    return synced_chroms


def assert_and_fix_args_for_training(args):
    setattr(args, "preferences", None)
    setattr(args, "signal", None)
    setattr(args, "tsites", None)
    synced_tchroms = get_synced_chroms(
        args.tchroms,
        [
            args.sequence,
        ],
        True
    )
    synced_vchroms = get_synced_chroms(
        args.vchroms,
        [
            args.sequence,
        ],
        True
    )

    assert set(synced_tchroms).isdisjoint(set(synced_vchroms)), \
        "--tchroms and --vchroms shouldn't intersect. Exiting"

    synced_chroms = get_synced_chroms(  # call it just to take --chroms without possible regions
        args.chroms,
        [
            args.sequence,
        ],
        True
    )

    assert set(synced_tchroms).union(set(synced_vchroms)).issubset(set(synced_chroms)), \
        "--tchroms and --vchroms should be subset of --chroms. Exiting"

    synced_chroms = get_synced_chroms(
        set(synced_chroms) - set(synced_tchroms) - set(synced_vchroms),
        [
            args.sequence,
        ],
        True
    )

    synced_chroms.update(synced_tchroms)
    synced_chroms.update(synced_vchroms)

    assert len(synced_chroms) > 0, \
        "--chroms, --tchroms or --vchroms failed to sync with the provided files. Exiting"

    setattr(args, "tchroms", synced_tchroms)
    setattr(args, "vchroms", synced_vchroms)
    setattr(args, "chroms", synced_chroms)

    if args.threads is None:
        args.threads = 1  # TODO: maybe choose a smarter way to set default threads number


def assert_and_fix_args(args):
    if args.func == run_training:
        assert_and_fix_args_for_training(args)
    else:
        pass


def get_parser():
    # Parent (general) parser
    parent_parser = argparse.ArgumentParser(add_help=False)
    general_parser = argparse.ArgumentParser(description="Neural networks for predicting TF binding using ATAC-seq")
    subparsers = general_parser.add_subparsers()
    subparsers.required = True

    general_parser.add_argument("--version",
                                action="version",
                                version=get_version(),
                                help="Print version information and exit"
                                )

    # Average parser
    average_parser = subparsers.add_parser("average",
                                           parents=[parent_parser],
                                           help="Run maxATAC average"
                                           )

    # Set the default function to run averaging
    average_parser.set_defaults(func=run_averaging)

    average_parser.add_argument("--bigwigs",
                                dest="bigwig_files",
                                type=str,
                                nargs="+",
                                required=True,
                                help="Input bigwig files to average."
                                )

    average_parser.add_argument("--prefix",
                                dest="prefix",
                                type=str,
                                required=True,
                                help="Output filename prefix."
                                )

    average_parser.add_argument("--chrom_sizes",
                                dest="chrom_sizes",
                                type=str,
                                default=DEFAULT_CHROM_SIZES,
                                help="Input chromosome sizes file. Default is hg38."
                                )

    average_parser.add_argument("--chromosomes",
                                dest="chromosomes",
                                type=str,
                                nargs="+",
                                default=AUTOSOMAL_CHRS,
                                help="Chromosomes for averaging. \
                                      Default: 1-22"
                                )

    average_parser.add_argument("--output",
                                dest="output_dir",
                                type=str,
                                default="./average",
                                help="Output directory."
                                )

    average_parser.add_argument("--loglevel",
                                dest="loglevel",
                                type=str,
                                default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                                choices=LOG_LEVELS.keys(),
                                help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                )

    # Predict parser
    predict_parser = subparsers.add_parser("predict",
                                           parents=[parent_parser],
                                           help="Run maxATAC prediction",
                                           )

    predict_parser.set_defaults(func=run_prediction)

    predict_parser.add_argument("--models", dest="models", type=str, nargs="+",
                                required=True,
                                help="Trained model file(s)"
                                )

    predict_parser.add_argument("--quant",
                                dest="quant",
                                action='store_true',
                                default=False,
                                help="This argument should be set to true to build models based on quantitative data"
                                )

    predict_parser.add_argument("--sequence",
                                dest="sequence",
                                type=str,
                                required=True,
                                help="Genome sequence 2bit file"
                                )

    predict_parser.add_argument("--signal",
                                dest="signal",
                                type=str,
                                required=True,
                                help="Input signal file"
                                )

    predict_parser.add_argument("--output",
                                dest="output",
                                type=str,
                                default="./prediction_results",
                                help="Folder for prediction results. Default: ./prediction_results"
                                )

    predict_parser.add_argument("--blacklist",
                                dest="blacklist",
                                type=str,
                                default=BLACKLISTED_REGIONS,
                                help="The blacklisted regions to exclude"
                                )

    predict_parser.add_argument("--roi",
                                dest="roi",
                                default=False,
                                required=False,
                                help="Bed file with ranges for input sequences to be predicted. \
                                      Default: None, predictions are done on the whole chromosome length"
                                )

    predict_parser.add_argument("--stranded",
                                dest="stranded",
                                default=False,
                                action='store_true',
                                required=False,
                                help="Whether to make predictions based on both strands")

    predict_parser.add_argument("--threads",
                                dest="threads",
                                default=get_cpu_count(),
                                type=int,
                                help="# of processes to run prediction in parallel. \
                                        Default: # of --models multiplied by # of --chromosomes"
                                )

    predict_parser.add_argument("--loglevel",
                                dest="loglevel",
                                type=str,
                                default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                                choices=LOG_LEVELS.keys(),
                                help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                )

    predict_parser.add_argument("--batch_size",
                                dest="batch_size",
                                type=int,
                                default=10000,
                                help="Number of regions to predict on at a time"
                                )

    predict_parser.add_argument("--step_size",
                                dest="step_size",
                                type=int,
                                default=INPUT_LENGTH,
                                help="Step size to use to build sliding window regions"
                                )

    predict_parser.add_argument("--prefix",
                                dest="prefix",
                                type=str,
                                default="maxatac_predict",
                                help="Prefix for filename"
                                )

    predict_parser.add_argument("--chromosome_sizes",
                                dest="chromosome_sizes",
                                type=str,
                                default=DEFAULT_CHROM_SIZES,
                                help="The chromosome sizes file to reference"
                                )

    predict_parser.add_argument("--chromosomes",
                                dest="chromosomes",
                                type=str,
                                nargs="+",
                                default=DEFAULT_TEST_CHRS,
                                help="Chromosomes from --chromosomes fixed for prediction. \
                                      Default: 1, 8"
                                )

    # Train parser
    train_parser = subparsers.add_parser("train",
                                         parents=[parent_parser],
                                         help="Run maxATAC training"
                                         )

    train_parser.set_defaults(func=run_training)

    train_parser.add_argument("--sequence",
                              dest="sequence",
                              type=str,
                              required=True,
                              help="Genome sequence 2bit file"
                              )

    train_parser.add_argument("--quant",
                              dest="quant",
                              action='store_true',
                              default=False,
                              help="This argument should be set to true to build models based on quantitative data"
                              )

    train_parser.add_argument("--meta_file",
                              dest="meta_file",
                              type=str,
                              required=True,
                              help="Meta file containing ATAC Signal and peak path for all cell lines (.tsv format)"
                              )

    train_parser.add_argument("--train_roi",
                              dest="train_roi",
                              type=str,
                              required=False,
                              help="Optional BED format file that will be used as the training regions of interest "
                                   "instead of using the peak files to build training regions"
                              )

    train_parser.add_argument("--validate_roi",
                              dest="validate_roi",
                              type=str,
                              required=False,
                              help="Optional BED format file that will be used as the validation regions of interest "
                                   "instead of using the peak files to build validation regions"
                              )

    train_parser.add_argument("--target_scale_factor",
                              dest="target_scale_factor",
                              type=float,
                              required=False,
                              default=1,
                              help="Scaling factor for scaling model targets signal. Used only for quantitative models"
                              )

    # I set default to sigmoid.
    train_parser.add_argument("--output_activation",
                              dest="output_activation",
                              type=str,
                              required=False,
                              default="sigmoid",
                              help="Activation function used for model output layer. Default: sigmoid"
                              )

    train_parser.add_argument("--chroms",
                              dest="chroms",
                              type=str,
                              nargs="+",
                              required=False,
                              default=DEFAULT_TRAIN_VALIDATE_CHRS,
                              help="Chromosome list to use for training and validation."
                              )

    train_parser.add_argument("--tchroms",
                              dest="tchroms",
                              type=str,
                              nargs="+",
                              required=False,
                              default=DEFAULT_TRAIN_CHRS,
                              help="Chromosome list to use for training."
                              )

    train_parser.add_argument("--vchroms",
                              dest="vchroms",
                              type=str,
                              nargs="+",
                              required=False,
                              default=DEFAULT_VALIDATE_CHRS,
                              help="Chromosome list to use for validation"
                              )

    train_parser.add_argument("--arch",
                              dest="arch",
                              type=str,
                              required=False,
                              default="DCNN_V2",
                              help="Specify the model architecture. Currently support DCNN_V2, RES_DCNN_V2, "
                                   "MM_DCNN_V2 and MM_Res_DCNN_V2 "
                              )

    train_parser.add_argument("--rand_ratio",
                              dest="rand_ratio",
                              type=float,
                              required=False,
                              default=.3,
                              help="Ratio for controlling fraction of random sequences in each training batch. "
                                   "Default: .3 "
                              )

    train_parser.add_argument("--seed",
                              dest="seed",
                              type=int,
                              default=random.randint(1, 99999),
                              help="Seed for pseudo-random generanor. Default: random int [1, 99999]"
                              )

    train_parser.add_argument("--weights",
                              dest="weights",
                              type=str,
                              help="Weights to initialize model before training. Default: do not load"
                              )

    train_parser.add_argument("--epochs",
                              dest="epochs",
                              type=int,
                              default=DEFAULT_TRAIN_EPOCHS,
                              help="Number of training epochs. Default: " + str(DEFAULT_TRAIN_EPOCHS)
                              )

    train_parser.add_argument("--batches",
                              dest="batches",
                              type=int,
                              default=DEFAULT_TRAIN_BATCHES_PER_EPOCH,
                              help="Number of training batches per epoch. Default: " + str(
                                  DEFAULT_TRAIN_BATCHES_PER_EPOCH)
                              )

    train_parser.add_argument("--batch_size",
                              dest="batch_size",
                              type=int,
                              default=BATCH_SIZE,
                              help="Number of examples per batch. Default: " + str(BATCH_SIZE)
                              )

    train_parser.add_argument("--val_batch_size",
                              dest="val_batch_size",
                              type=int,
                              default=VAL_BATCH_SIZE,
                              help="Number of examples per batch. Default: " + str(VAL_BATCH_SIZE)
                              )

    train_parser.add_argument("--prefix",
                              dest="prefix",
                              type=str,
                              default="maxatac_model",
                              help="Output prefix. Default: weights"
                              )

    train_parser.add_argument("--output",
                              dest="output",
                              type=str,
                              default="./training_results",
                              help="Folder for training results. Default: ./training_results"
                              )

    train_parser.add_argument("--plot",
                              dest="plot",
                              action="store_true",
                              default=True,
                              help="Plot model structure and training history. Default: True"
                              )

    train_parser.add_argument("--dense",
                              dest="dense",
                              action="store_true",
                              default=False,
                              help="If True, then make a dense layer before model output. Default: False"
                              )

    train_parser.add_argument("--threads",
                              dest="threads",
                              type=int,
                              default=get_cpu_count(),
                              help="Number of processes to run training in parallel. Default: 1"
                              )

    train_parser.add_argument("--loglevel",
                              dest="loglevel",
                              type=str,
                              default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                              choices=LOG_LEVELS.keys(),
                              help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                              )

    train_parser.add_argument("--shuffle_cell_type",
                              dest="shuffle_cell_type",
                              action="store_true",
                              default=False,
                              help="If shuffle_cell_type, then shuffle training ROI cell type label"
                              )

    train_parser.add_argument("--rev_comp",
                              dest="rev_comp",
                              action="store_true",
                              default=False,
                              help="If rev_comp, then use the reverse complement in training"
                              )

    # Normalize parser
    normalize_parser = subparsers.add_parser("normalize",
                                             parents=[parent_parser],
                                             help="Run minmax normalization")

    normalize_parser.set_defaults(func=run_normalization)

    normalize_parser.add_argument("--signal",
                                  dest="signal",
                                  type=str,
                                  required=True,
                                  help="Input signal bigWig file(s) to be normalized by reference")

    normalize_parser.add_argument("--chrom_sizes",
                                  dest="chrom_sizes",
                                  type=str,
                                  default=DEFAULT_CHROM_SIZES,
                                  help="Chrom sizes file")

    normalize_parser.add_argument("--chroms",
                                  dest="chroms",
                                  type=str,
                                  nargs="+",
                                  default=AUTOSOMAL_CHRS,
                                  help="Chromosome list for analysis. \
                                    Regions in a form of chrN:start-end are ignored. \
                                    Use --filters instead \
                                    Default: main human chromosomes, whole length"
                                  )

    normalize_parser.add_argument("--output",
                                  dest="output",
                                  type=str,
                                  default="./normalize",
                                  help="Folder for normalization results. Default: ./normalization_results")

    normalize_parser.add_argument("--prefix",
                                  dest="prefix",
                                  type=str,
                                  default="normalized",
                                  help="Name to use for filename")

    normalize_parser.add_argument("--min",
                                  dest="min",
                                  required=False,
                                  type=int,
                                  default=0,
                                  help="The minimum value to use for normalization")

    normalize_parser.add_argument("--max",
                                  dest="max",
                                  type=int,
                                  required=False,
                                  default=False,
                                  help="The maximum value to use for normalization")

    normalize_parser.add_argument("--clip",
                                  dest="clip",
                                  type=bool,
                                  required=False,
                                  default=False,
                                  help="Whether to clip minmax values to the range 0,1")

    normalize_parser.add_argument("--method",
                                  dest="method",
                                  type=str,
                                  default="min-max",
                                  help="The method to use for normalization")

    normalize_parser.add_argument("--max_percentile",
                                  dest="max_percentile",
                                  required=False,
                                  type=int,
                                  default=100,
                                  help="The maximum percentile to use for normalization")

    normalize_parser.add_argument("--loglevel",
                                  dest="loglevel",
                                  type=str,
                                  default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                                  choices=LOG_LEVELS.keys(),
                                  help="Logging level. Default: " + DEFAULT_LOG_LEVEL)

    normalize_parser.add_argument("--blacklist",
                                  dest="blacklist",
                                  type=str,
                                  default=BLACKLISTED_REGIONS_BIGWIG,
                                  help="The blacklisted regions to exclude"
                                  )

    # Benchmark parser
    benchmark_parser = subparsers.add_parser("benchmark",
                                             parents=[parent_parser],
                                             help="Run maxATAC benchmarking"
                                             )

    benchmark_parser.set_defaults(func=run_benchmarking)

    benchmark_parser.add_argument("--prediction",
                                  dest="prediction",
                                  type=str,
                                  required=True,
                                  help="Prediction bigWig file"
                                  )
    benchmark_parser.add_argument("--quant",
                                  dest="quant",
                                  action='store_true',
                                  default=False,
                                  help="This argument should be set to true for models based on quantitative data"
                                  )

    benchmark_parser.add_argument("--gold_standard",
                                  dest="gold_standard",
                                  type=str,
                                  required=True,
                                  help="Gold Standard bigWig file"
                                  )

    benchmark_parser.add_argument("--chromosomes",
                                  dest="chromosomes",
                                  type=str,
                                  nargs="+",
                                  default=DEFAULT_TEST_CHRS,
                                  help="Chromosomes list for analysis. \
                                        Optionally with regions in a form of chrN:start-end. \
                                        Default: main human chromosomes, whole length"
                                  )

    benchmark_parser.add_argument("--bin_size",
                                  dest="bin_size",
                                  type=int,
                                  default=DEFAULT_BENCHMARKING_BIN_SIZE,
                                  help="Bin size to split prediction and control data before running prediction. \
                                        Default: " + str(DEFAULT_BENCHMARKING_BIN_SIZE)
                                  )

    benchmark_parser.add_argument("--agg",
                                  dest="agg_function",
                                  type=str,
                                  default=DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
                                  help="Aggregation function to use for combining results into bins: \
                                        max, coverage, mean, std, min"
                                  )

    benchmark_parser.add_argument("--round_predictions",
                                  dest="round_predictions",
                                  type=int,
                                  default=DEFAULT_ROUND,
                                  help="Round binned values to this number of decimal places"
                                  )

    benchmark_parser.add_argument("--prefix",
                                  dest="prefix",
                                  type=str,
                                  required=True,
                                  help="Prefix for the file name"
                                  )

    benchmark_parser.add_argument("--output_directory",
                                  dest="output_directory",
                                  type=str,
                                  default="./benchmarking_results",
                                  help="Folder for benchmarking results. Default: ./benchmarking_results"
                                  )

    benchmark_parser.add_argument("--loglevel",
                                  dest="loglevel",
                                  type=str,
                                  default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                                  choices=LOG_LEVELS.keys(),
                                  help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                  )

    benchmark_parser.add_argument("--blacklist",
                                  dest="blacklist",
                                  type=str,
                                  default=BLACKLISTED_REGIONS_BIGWIG,
                                  help="The blacklisted regions to exclude"
                                  )

    # Prediction_signal parser
    prediction_signal_parser = subparsers.add_parser("prediction_signal",
                                             parents=[parent_parser],
                                             help="Run maxATAC prediction_signal"
                                             )

    prediction_signal_parser.set_defaults(func=run_prediction_signal)

    prediction_signal_parser.add_argument("--prediction",
                                  dest="prediction",
                                  type=str,
                                  required=True,
                                  help="Prediction bigWig file"
                                  )
    prediction_signal_parser.add_argument("--quant",
                                  dest="quant",
                                  action='store_true',
                                  default=False,
                                  help="This argument should be set to true for models based on quantitative data"
                                  )

    prediction_signal_parser.add_argument("--sequence",
                                  dest="sequence",
                                  type=str,
                                  required=True,
                                  help="hg38 sequence file"
                                  )

    prediction_signal_parser.add_argument("--chromosomes",
                                  dest="chromosomes",
                                  type=str,
                                  nargs="+",
                                  default=DEFAULT_TEST_CHRS,
                                  help="Chromosomes list for analysis. \
                                            Optionally with regions in a form of chrN:start-end. \
                                            Default: main human chromosomes, whole length"
                                  )

    prediction_signal_parser.add_argument("--bin_size",
                                  dest="bin_size",
                                  type=int,
                                  default=DEFAULT_BENCHMARKING_BIN_SIZE,
                                  help="Bin size to split prediction and control data before running prediction. \
                                            Default: " + str(DEFAULT_BENCHMARKING_BIN_SIZE)
                                  )

    prediction_signal_parser.add_argument("--agg",
                                  dest="agg_function",
                                  type=str,
                                  default=DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION,
                                  help="Aggregation function to use for combining results into bins: \
                                            max, coverage, mean, std, min"
                                  )

    prediction_signal_parser.add_argument("--round_predictions",
                                  dest="round_predictions",
                                  type=int,
                                  default=DEFAULT_ROUND,
                                  help="Round binned values to this number of decimal places"
                                  )

    prediction_signal_parser.add_argument("--prefix",
                                  dest="prefix",
                                  type=str,
                                  required=True,
                                  help="Prefix for the file name"
                                  )

    prediction_signal_parser.add_argument("--output_directory",
                                  dest="output_directory",
                                  type=str,
                                  default="./benchmarking_results",
                                  help="Folder for benchmarking results. Default: ./benchmarking_results"
                                  )

    prediction_signal_parser.add_argument("--loglevel",
                                  dest="loglevel",
                                  type=str,
                                  default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                                  choices=LOG_LEVELS.keys(),
                                  help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                                  )

    prediction_signal_parser.add_argument("--blacklist",
                                  dest="blacklist",
                                  type=str,
                                  default=BLACKLISTED_REGIONS_BIGWIG,
                                  help="The blacklisted regions to exclude"
                                  )

    # threshold_parser
    peaks_parser = subparsers.add_parser("peaks",
                                         parents=[parent_parser],
                                         help="Run maxATAC peaks"
                                         )

    # Set the default function to run averaging
    peaks_parser.set_defaults(func=call_peaks)

    peaks_parser.add_argument("--prefix",
                              dest="prefix",
                              type=str,
                              required=True,
                              help="Output prefix."
                              )

    peaks_parser.add_argument("--bin_size",
                              dest="bin_size",
                              type=int,
                              default=DEFAULT_BENCHMARKING_BIN_SIZE,
                              help="Chromosomes for averaging")

    peaks_parser.add_argument("--output",
                              dest="output_dir",
                              type=str,
                              default="./peaks",
                              help="Output directory."
                              )

    peaks_parser.add_argument("--input_bigwig",
                              dest="input_bigwig",
                              type=str,
                              help="Input bigwig")

    peaks_parser.add_argument("--threshold",
                              dest="threshold",
                              type=float,
                              help="Input bigwig")

    peaks_parser.add_argument("--loglevel",
                              dest="loglevel",
                              type=str,
                              default=LOG_LEVELS[DEFAULT_LOG_LEVEL],
                              choices=LOG_LEVELS.keys(),
                              help="Logging level. Default: " + DEFAULT_LOG_LEVEL
                              )

    return general_parser


def print_args(args, logger, header="Arguments:\n", excl=["func"]):
    filtered = {
        k: v for k, v in args.__dict__.items()
        if k not in excl
    }
    logger(header + dump(filtered))


# we need to cwd_abs_path parameter only for running unit tests
def parse_arguments(argsl, cwd_abs_path=None):
    cwd_abs_path = getcwd() if cwd_abs_path is None else cwd_abs_path
    if len(argsl) == 0:
        argsl.append("")  # otherwise fails with error if empty
    args, _ = get_parser().parse_known_args(argsl)

    if args.func == run_training:
        args = normalize_args(
            args,
            [
                "func", "loglevel", "threads", "seed",
                "proportion", "vchroms", "tchroms",
                "chroms", "keep", "epochs", "batches",
                "prefix", "plot", "lrate", "decay", "bin",
                "minimum", "test_cell_lines", "rand_ratio",
                "train_tf", "arch", "quant", "batch_size",
                "val_batch_size", "target_scale_factor",
                "output_activation", "dense", "shuffle_cell_type", "rev_comp"
            ],
            cwd_abs_path
        )

    assert_and_fix_args(args)

    return args
