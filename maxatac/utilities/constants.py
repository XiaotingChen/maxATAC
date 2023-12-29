from logging import FATAL, ERROR, WARNING, INFO, DEBUG
import os

# Internal use

LOG_LEVELS = {
    "fatal": FATAL,
    "error": ERROR,
    "warning": WARNING,
    "info": INFO,
    "debug": DEBUG,
}
LOG_FORMAT = "[%(asctime)s]\n%(message)s"
CPP_LOG_LEVEL = {FATAL: 3, ERROR: 3, WARNING: 2, INFO: 1, DEBUG: 0}

# Paths to Genomic resource and scripts for preparing data. This is where most of the hardcoded paths are generated.
# If the user runs maxatac data the data will be installed here. The default arguments for some commands rely on these paths.
# The user can put the data anywhere, but they will need to adjust the paths for each file
maxatac_data_path = os.path.join(os.path.expanduser("~"), "opt", "maxatac", "data")

# build path names
blacklist_path = os.path.join(
    maxatac_data_path, "hg38/hg38_maxatac_blacklist.bed"
)  # maxATAC extended blacklist as bed
blacklist_bigwig_path = os.path.join(
    maxatac_data_path, "hg38/hg38_maxatac_blacklist.bw"
)  # maxATAC extended blacklist as bigwig
chrom_sizes_path = os.path.join(
    maxatac_data_path, "hg38/hg38.chrom.sizes"
)  # chrom sizes file
sequence_path = os.path.join(maxatac_data_path, "hg38/hg38.2bit")  # sequence 2bit
prepare_atac_script_dir = os.path.join(
    maxatac_data_path, "scripts", "ATAC", "ATAC_bowtie2_pipeline.sh"
)  # bulk processing script
prepare_scatac_script_dir = os.path.join(
    maxatac_data_path, "scripts", "ATAC", "scatac_generate_bigwig.sh"
)  # scatac processing script

# normalize paths
DATA_PATH = os.path.normpath(maxatac_data_path)
BLACKLISTED_REGIONS = os.path.normpath(blacklist_path)
BLACKLISTED_REGIONS_BIGWIG = os.path.normpath(blacklist_bigwig_path)
DEFAULT_CHROM_SIZES = os.path.normpath(chrom_sizes_path)
REFERENCE_SEQUENCE_TWOBIT = os.path.normpath(sequence_path)
PREPARE_BULK_SCRIPT = os.path.normpath(prepare_atac_script_dir)
PREPARE_scATAC_SCRIPT = os.path.normpath(prepare_scatac_script_dir)

# Default chromosome sets
ALL_CHRS = [
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr7",
    "chr8",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
    "chrX",
    "chrY",
]

AUTOSOMAL_CHRS = [
    "chr1",
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr7",
    "chr8",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
]

# Defualt chrs excludes 1,8
DEFAULT_TRAIN_VALIDATE_CHRS = [
    "chr2",
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr7",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr19",
    "chr20",
    "chr21",
    "chr22",
    "chrX",
]

# Default train chrs exclude 1,2,8,19,X,Y,M
DEFAULT_TRAIN_CHRS = [
    "chr3",
    "chr4",
    "chr5",
    "chr6",
    "chr7",
    "chr9",
    "chr10",
    "chr11",
    "chr12",
    "chr13",
    "chr14",
    "chr15",
    "chr16",
    "chr17",
    "chr18",
    "chr20",
    "chr21",
    "chr22",
]

DEFAULT_VALIDATE_CHRS = ["chr2", "chr19"]

DEFAULT_TEST_CHRS = ["chr1", "chr8"]

DEFAULT_LOG_LEVEL = "info"

DEFAULT_TRAIN_EPOCHS = 20

DEFAULT_TRAIN_BATCHES_PER_EPOCH = 100


DEFAULT_ADAM_LEARNING_RATE = 0.001
DEFAULT_ADAM_DECAY = 1e-5
DEFAULT_VALIDATE_RAND_RATIO = 0.7

PRETRAINING_USE_CHIP_ROI = False

# True if using intermediate fusion of ATAC-seq
############################ CONFIGS FOR INTER FUSION #############################
INTER_FUSION = True
CONV_TOWER_CONFIGS_FUSION = {
    "genome": [
        {
            "num_layer": 1,
            "kernel": 10,
            "stride": 1,
            "padding": "same",
            "num_filters": 64,
            "activation": "relu",
        },
        {
            "num_layer": 1,
            "kernel": 10,
            "stride": 1,
            "padding": "same",
            "num_filters": 64,
            "activation": "relu",
        },
    ],
    "atac": [
        {
            "num_layer": 1,
            "kernel": 10,
            "stride": 1,
            "padding": "same",
            "num_filters": 64,
            "activation": "relu",
        },
        {
            "num_layer": 1,
            "kernel": 10,
            "stride": 1,
            "padding": "same",
            "num_filters": 64,
            "activation": "relu",
        },
    ],
    "merge": {
        "num_layer": 1,
        "kernel": 10,
        "stride": 1,
        "padding": "same",
        "num_filters": 64,
        "activation": "relu",
    },
}
###################################################################################
################### CONFIGS FOR INTER FUSION CROSS ATTENTION ######################
CONV_TOWER_CROSSATT_CONFIGS_FUSION = {
    "genome": [
        {
            "num_layer": 2,
            "kernel": 10,
            "stride": 1,
            "padding": "same",
            "num_filters": 64,
            "activation": "relu",
        },
    ],
    "signal": [
        {
            "num_layer": 2,
            "kernel": 10,
            "stride": 1,
            "padding": "same",
            "num_filters": 64,
            "activation": "relu",
        },
    ],
}
DOWNSAMPLE_METHOD_CONV_TOWER_CROSSATT = "None"
USE_TOKEN = True
NUM_HEADS_SELFATT = 4
EMBEDDING_SIZE_SELFATT = 64
NUM_MHA_SELFATT = 4
WHOLE_ATTENTION_KWARGS_SELFATT_GENOME = {
    "attention_dropout_rate": 0.05,
    "num_heads": NUM_HEADS_SELFATT,
    "value_size": EMBEDDING_SIZE_SELFATT // NUM_HEADS_SELFATT,  #
    "key_size": EMBEDDING_SIZE_SELFATT // NUM_HEADS_SELFATT,  #
    "num_relative_position_features": None,  # channels // num_heads,
    "positional_dropout_rate": 0.01,
    "relative_position_symmetric": True,
    "relative_position_functions": [  # leave these here although they are not used
        "positional_features_exponential",
        "positional_features_central_mask",
        "positional_features_gamma",
        #'positional_features_cosine',
        #'positional_features_linear_masks',
        #'positional_features_sin_cos',
    ],
    "relative_positions": True,
    "scaling": True,
    "initializer": "GlorotNormal",  # better to define the initializer here
    "zero_initialize": True,
}
NUM_HEADS_CROSSATT = 4
EMBEDDING_SIZE_CROSSATT = 64
NUM_MHA_CROSSATT = 4
WHOLE_ATTENTION_KWARGS_CROSSATT_SIGNAL = {
    "attention_dropout_rate": 0.05,
    "num_heads": NUM_HEADS_CROSSATT,
    "value_size": EMBEDDING_SIZE_CROSSATT // NUM_HEADS_CROSSATT,  #
    "key_size": EMBEDDING_SIZE_CROSSATT // NUM_HEADS_CROSSATT,  #
    "num_relative_position_features": None,  # channels // num_heads,
    "positional_dropout_rate": 0.01,
    "relative_position_symmetric": True,
    "relative_position_functions": [
        "positional_features_exponential",
        "positional_features_central_mask",
        "positional_features_gamma",
        #'positional_features_cosine',
        #'positional_features_linear_masks',
        #'positional_features_sin_cos',
    ],
    "relative_positions": True,
    "scaling": True,
    "initializer": "GlorotNormal",  # better to define the initializer here
    "zero_initialize": True,
}

###################################################################################

# Constants for the conv tower before MHA (key is number of conv layers, value is number of filters)
CONV_TOWER_CONFIGS = [
    {
        "num_layer": 2,
        "kernel": 10,
        "stride": 1,
        "padding": "same",
        "num_filters": 64,
        "activation": "relu",
    }
]
DOWNSAMPLE_METHOD_CONV_TOWER = "conv"

# Constants for the Inception module (nearly all branches have a 1x1 conv block first)
INCEPTION_BRANCHES = [
    [
        {
            "name": "conv",
            "num_layer": 1,
            "kernel": 1,
            "stride": 1,
            "padding": "same",
            "num_filters": 16,
            "activation": "relu",
        },
        {
            "name": "conv",
            "num_layer": 2,
            "kernel": 10,
            "stride": 1,
            "padding": "same",
            "num_filters": 16,
            "activation": "relu",
        },
    ],
    [
        {
            "name": "conv",
            "num_layer": 1,
            "kernel": 1,
            "stride": 1,
            "padding": "same",
            "num_filters": 16,
            "activation": "relu",
        },
        {
            "name": "conv",
            "num_layer": 1,
            "kernel": 13,
            "stride": 1,
            "padding": "same",
            "num_filters": 16,
            "activation": "relu",
        },
    ],
    [
        {"name": "pool", "pool_size": 13, "stride": 1, "padding": "same"},
        {
            "name": "conv",
            "num_layer": 1,
            "kernel": 1,
            "stride": 1,
            "padding": "same",
            "num_filters": 16,
            "activation": "relu",
        },
    ],
    [
        {
            "name": "conv",
            "num_layer": 1,
            "kernel": 1,
            "stride": 1,
            "padding": "same",
            "num_filters": 16,
            "activation": "relu",
        }
    ],
]

# Constants for self-attention (embedding size must be equal num_heads * key_dims, and must also be equal
#                               to the num filters of the last layer in the conv filters
# )
# The number of mha layers (used for both my and DeepMind's transformer)
NUM_MHA = 4

# Constants for my transformer template
EMBEDDING_SIZE = 64
NUM_HEADS = 4
KEY_DIMS = 16
D_FF = 256

# Constants for DeepMind's RPE (some variables for DeepMind's code will have the DM- prefix)
USE_RPE = True
DM_DROPOUT_RATE = 0.1
WHOLE_ATTENTION_KWARGS = {
    "attention_dropout_rate": 0.05,
    "num_heads": NUM_HEADS,
    "value_size": EMBEDDING_SIZE // NUM_HEADS,  #
    "key_size": 12,  #
    "num_relative_position_features": None,  # channels // num_heads,
    "positional_dropout_rate": 0.01,
    "relative_position_symmetric": True,
    "relative_position_functions": [  # leave these here although they are not used
        "positional_features_exponential",
        "positional_features_central_mask",
        "positional_features_gamma",
        #'positional_features_cosine',
        #'positional_features_linear_masks',
        #'positional_features_sin_cos',
    ],
    "relative_positions": True,
    "scaling": True,
    "initializer": "GlorotNormal",  # better to define the initializer here
    "zero_initialize": True,
}

BATCH_SIZE = 1000
VAL_BATCH_SIZE = 1000
BP_DICT = {"A": 0, "C": 1, "G": 2, "T": 3}
CHR_POOL_SIZE = 1000
BP_ORDER = ["A", "C", "G", "T"]
INPUT_FILTERS = 16

INPUT_KERNEL_SIZE = 7
BASENJI_INPUT_FILTERS = 288
BASENJI_INPUT_KERNEL_SIZE = 15
ENFORMER_INPUT_FILTERS = 768
ENFORMER_INPUT_KERNEL_SIZE = 15
INPUT_LENGTH = 1024
OUTPUT_LENGTH = 32  # INPUT_LENGTH/BP_RESOLUTION
INPUT_ACTIVATION = "relu"
KERNEL_INITIALIZER = "glorot_uniform"  # use he_normal initializer if activation is RELU
PADDING = "same"
FILTERS_SCALING_FACTOR = 1.5
PURE_CONV_LAYERS = 4
CONV_BLOCKS = 6
DNA_INPUT_CHANNELS = 4
DILATION_RATE = [1, 1, 2, 4, 8, 16]
BP_RESOLUTION = 32
OUTPUT_FILTERS = 1
OUTPUT_KERNEL_SIZE = 1
POOL_SIZE = 2
POOL_SIZE_BEFORE_FLATTEN = 8
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
TRAIN_SCALE_SIGNAL = (0.9, 1.15)  # min max scaling ranges

# Prediction Constants
DEFAULT_MIN_PREDICTION = 0.001  # min prediction value to be reported in the output
DEFAULT_ROUND = 9
DEFAULT_PREDICTION_BATCH_SIZE = 10000
OUTPUT_ACTIVATION = "sigmoid"

# Benchmarking Constants
DEFAULT_BENCHMARKING_AGGREGATION_FUNCTION = "max"
DEFAULT_BENCHMARKING_BIN_SIZE = 200

INPUT_CHANNELS = 5
TRAIN_MONITOR = "val_loss"

PREDICTION_HEAD_DROPOUT_RATE = 0.05
RESIDUAL_CONNECTION_DROPOUT_RATE = 0.05

DEFAULT_COSINEDECAYRESTARTS_INITIAL_LR_MULTIPLIER = 5
DEFAULT_COSINEDECAYRESTARTS_FIRST_DECAY_STEPS = 5*100
DEFAULT_COSINEDECAYRESTARTS_ALPHA = 0.05
DEFAULT_COSINEDECAYRESTARTS_T_MUL = 1.5
DEFAULT_COSINEDECAYRESTARTS_M_MUL = 0.6

DEFAULT_COSINEDECAY_DECAY_STEPS=10000
DEFAULT_COSINEDECAY_ALPHA=0.05
DEFAULT_COSINEDECAY_WARMUP_TARGET_MULTIPLIER=1.0
DEFAULT_COSINEDECAY_WARMUP_STEPS=1000

MODEL_CONFIG_UPDATE_LIST=[
"OPTIMIZER",
"USING_BASENJI_KERNEL",
"USING_ENFORMER_KERNEL",
"BASENJI_KERNEL_TRAINABLE",
"ENFORMER_KERNEL_TRAINABLE",
"KERNEL_REPLACING",
"SUPPRESS_DROPOUT",
"RESIDUAL_CONNECTION_DROPOUT_RATE",
"PREDICTION_HEAD_DROPOUT_RATE",
"COSINEDECAYRESTARTS",
"COSINEDECAYRESTARTS_FIRST_DECAY_STEPS",
"COSINEDECAY",
"COSINEDECAYALPHA",
"COSINEDECAYDECAYSTEPS",
"FOCAL_LOSS_ALPHA",
"FOCAL_LOSS_GAMMA",
"FOCAL_LOSS",
"FOCAL_LOSS_APPLY_ALPHA",
"INITIAL_LEARNING_RATE",
"REGULARIZATION",
"ELASTIC_L1",
"ELASTIC_L2",
"LOSS_FLANKING_TRUNCATION_SIZE",
"SHUFFLE_AUGMENTATION",
"FULL_TRANSFORMER_OUTPUT",
"OVERRIDE_ACTIVATION",
"SUPPRESS_CELL_TYPE_SPECIFIC_TN_WEIGHTS",
'dice_unknown_coef'
]