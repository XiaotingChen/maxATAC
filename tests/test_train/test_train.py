
import os
import pytest
from maxatac.analyses.train import *
from maxatac.utilities.system_tools import Namespace
import subprocess
import json

OUTPUT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "temp"))
TFDS_META=("/data/weirauchlab/team/ches2d/MyTools/maxATAC_DATA_no_TN_tuning_LOO/meta/data_meta_TCF7.csv")
sequence="/users/ches2d/opt/maxatac/data/hg38/hg38.2bit"
meta_file=("/data/weirauchlab/team/ches2d/MyTools/maxATAC_extension/__TEST_RUNS__/__batch_run__"
           "/meta_full_model/meta_file_TCF7.tsv")


model_config="maxATAC_extension_Transformer.json"

expected_outputs = {
    "Multiinput_1.h5": "02fba2efee33ad33aac8cf5f876d06d6",
}

with open("cmd_args.json",'r') as f:
    all_namespace=json.load(f)

def test_train(all_namespace):
    """
    Test that maxatac average can produce a .bw that has been filtered for chr22 and chr3

    """
    args_test_train = Namespace(
        genome="hg38",
        arch="Transformer",
        sequence=sequence,
        meta_file=meta_file,
        output=OUTPUT_FOLDER,
        prefix="Multiinput",
        epochs=1,
        model_config=model_config,
        threads=16,
        multiprocessing=True,
        max_queue_size=32,
        batch_size=1024,
        OPTIMIZER="Adam",
        RESIDUAL_CONNECTION_DROPOUT_RATE=0.05,
        PREDICTION_HEAD_DROPOUT_RATE=0.05,
        COSINEDECAYRESTARTS=True,
        TFDS_META=TFDS_META,
        ATAC_SAMPLING_MULTIPLIER=5,
        SHUFFLE_AUGMENTATION="peak_centric",
        FULL_TRANSFORMER_OUTPUT=True,
        DICE_UNKNOWN_COEF=1,
        OVERRIDE_ACTIVATION="mish",
    )

    for key in all_namespace.keys():
        if key not in args_test_train.__dict__.keys():
            setattr(args_test_train,key,all_namespace[key])

    run_training(args_test_train)

    output_file = os.path.join(OUTPUT_FOLDER, "Multiinput_1.h5")

    results = subprocess.run(f"md5sum {output_file}", shell=True, capture_output=True)

    md5sum = str(results.stdout.decode()).split(" ")[0]

    assert md5sum == expected_outputs["Multiinput_1.h5"]

