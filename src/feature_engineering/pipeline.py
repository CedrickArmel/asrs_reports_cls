import argparse
import os

from dotenv import load_dotenv
from kfp import compiler, Client
from kfp.dsl import (container_component, ContainerSpec, Dataset,
                     importer, Input, Output, pipeline)
from omegaconf import OmegaConf
from wonderwords import RandomWord

load_dotenv()
fe_cfg = OmegaConf.load("conf/base/feature_engineering.yaml")
EXPERIMENT = fe_cfg.pipeline.experiment
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_PROJECT_REGION = os.getenv("GCP_PROJECT_REGION")
IMAGE = os.getenv("IMAGE")
KUBEFLOW_HOST = os.getenv("KUBEFLOW_HOST")
KUBEFLOW_PIPELINES_ROOT = os.getenv("KUBEFLOW_PIPELINES_ROOT")
PIPELINE_NAME = fe_cfg.pipeline.name
RAW_SOURCE = os.getenv("RAW_SOURCE")

TRAIN_DATASET = os.path.join(str(RAW_SOURCE), fe_cfg.components.train_dataset)
TEST_DATASET = os.path.join(str(RAW_SOURCE), fe_cfg.components.test_dataset)


@container_component
def featureengineering(train_set: Input[Dataset],
                       test_set: Input[Dataset],
                       train_data_out: Output[Dataset],
                       test_data_out: Output[Dataset]):
    return ContainerSpec(
        image=IMAGE,
        command=["python"],
        args=["src/feature_engineering/components.py",
              "feature_engineering",
              "--train-set", train_set.path,
              "--test-set", test_set.path,
              "--train-data-out", train_data_out.path,
              "--test-data-out", test_data_out.path])


@container_component
def storefeatures(train_features: Input[Dataset],
                  test_features: Input[Dataset]):
    return ContainerSpec(
        image=IMAGE,
        command=["python"],
        args=["src/feature_engineering/components.py",
              "store_features",
              "--train-features", train_features.path,
              "--test-features", test_features.path])


@pipeline(pipeline_root=KUBEFLOW_PIPELINES_ROOT)
def feature_engineering_pipeline():
    """Feature engineering pipeline.
    """
    train_set_op = importer(artifact_uri=TRAIN_DATASET,
                            artifact_class=Dataset,
                            reimport=False)
    test_set_op = importer(artifact_uri=TEST_DATASET,
                           artifact_class=Dataset,
                           reimport=False)
    feature_engineering_op = featureengineering(
        train_set=train_set_op.output,
        test_set=test_set_op.output)
    storefeatures(
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
