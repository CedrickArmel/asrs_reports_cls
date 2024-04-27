import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from kfp import compiler, Client
from kfp.dsl import pipeline
from omegaconf import OmegaConf
from wonderwords import RandomWord

current_dir = str(Path(__file__).parent).split('/')
for i in range(len(current_dir)+1):
    sys.path.append('/'.join(current_dir[:i]))
from feature_engineering.\
    components import (feature_engineering,
                       store_features)

load_dotenv()
fe_cfg = OmegaConf.load("conf/base/feature_engineering.yaml")
EXPERIMENT = fe_cfg.pipeline.experiment
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_PROJECT_REGION = os.getenv("GCP_PROJECT_REGION")
KUBEFLOW_HOST = os.getenv("KUBEFLOW_HOST")
KUBEFLOW_PIPELINES_ROOT = os.getenv("KUBEFLOW_PIPELINES_ROOT")
PIPELINE_NAME = fe_cfg.pipeline.name
RAW_SOURCE = os.getenv("RAW_SOURCE")

TRAIN_DATASET = os.path.join(str(RAW_SOURCE), fe_cfg.components.train_dataset)
TEST_DATASET = os.path.join(str(RAW_SOURCE), fe_cfg.components.test_dataset)


@pipeline(pipeline_root=KUBEFLOW_PIPELINES_ROOT)
def feature_engineering_pipeline():
    """Feature engineering pipeline.
    """
    feature_engineering_op = feature_engineering(train_set=TRAIN_DATASET,
                                                 test_set=TEST_DATASET)
    store_features(
        train_features=feature_engineering_op.outputs["train_data_out"],
        test_features=feature_engineering_op.outputs["test_data_out"])


if __name__ == "__main__":
    r = RandomWord()
    words = r.random_words(amount=2, word_max_length=10)
    JOB_NAME = "-".join(words)

    parser = argparse.ArgumentParser()
    parser.add_argument('--compile-only',
                        action='store_true')
    args = parser.parse_args()

    compiler.Compiler().compile(pipeline_func=feature_engineering_pipeline,
                                package_path=fe_cfg.pipeline.pkg_path)

    if not args.compile_only:
        client = Client(host=KUBEFLOW_HOST)
        experiment = client.create_experiment(name=EXPERIMENT)
        client.run_pipeline(
            experiment_id=experiment.experiment_id,
            job_name=JOB_NAME,
            pipeline_root=KUBEFLOW_PIPELINES_ROOT,
            pipeline_package_path=fe_cfg.pipeline.pkg_path)
