"""ASRS Report Classification project feature engineering pipeline
built using KFP v2 SDK."""
import argparse
import os

from dotenv import load_dotenv, dotenv_values
from google.cloud import aiplatform
from kfp import Client, compiler
from kfp.dsl import (container_component, ContainerSpec, Dataset,
                     importer, Input, Output, pipeline)
from omegaconf import OmegaConf
from wonderwords import RandomWord

load_dotenv()
fe_cfg = OmegaConf.load("conf/base/feature_engineering.yaml")

ENDPOINT = os.getenv("ENDPOINT")
EXPERIMENT = fe_cfg.pipeline.experiment
IMAGE = os.getenv("IMAGE")
PIPELINE_NAME = fe_cfg.pipeline.name
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
RAW_DATA = os.getenv("RAW_DATA")

TRAIN_DATASET = os.path.join(str(RAW_DATA), fe_cfg.components.train_dataset)
TEST_DATASET = os.path.join(str(RAW_DATA), fe_cfg.components.test_dataset)


@container_component
def featureengineering(train_set: Input[Dataset],
                       test_set: Input[Dataset],
                       train_data_out: Output[Dataset],
                       test_data_out: Output[Dataset]):
    return ContainerSpec(
        image=IMAGE,
        command=["python"],
        args=["src/feature_engineering/components.py",
              "featureengineering",
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
              "storefeatures",
              "--train-features", train_features.path,
              "--test-features", test_features.path])


@pipeline(pipeline_root=PIPELINE_ROOT)
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

    parser = argparse.ArgumentParser(
        prog="Feature Engineering Pipeline",
        description="""Programme to run the Feature Engineering Pipeline""")
    subparsers = parser.add_subparsers(title="Platforms", dest="platform")

    vertex_parser = subparsers.add_parser(
        "vertexai",
        help="Run the pipeline on GCP Vertex AI.")

    vertex_parser.add_argument(
        "--project",
        default=PROJECT_ID,
        help="Your project ID on GCP. \
            If empty use the PROJECT_ID environment variable.")

    vertex_parser.add_argument(
        "--region",
        default=REGION,
        help="The region of your Vertex AI resource on GCP. \
            If empty use the REGION environment variable")

    kubeflow_parser = subparsers.add_parser(
        "kubeflow",
        help="Run the pipeline on any Kubeflow platform.")

    kubeflow_parser.add_argument(
        "--endpoint",
        default=ENDPOINT,
        help="The endpoint of your kubeflow instance. \
            If empty use the ENDPOINT environment variable")

    parser.add_argument('--compile-only',
                        action='store_true')
    args = parser.parse_args()

    compiler.Compiler().compile(pipeline_func=feature_engineering_pipeline,
                                package_path=fe_cfg.pipeline.pkg_path)

    if not args.compile_only:
        if args.platform == "vertexai":
            aiplatform.init(project=args.project,
                            location=args.region)
            run = aiplatform.PipelineJob(
                project=args.project,
                location=args.region,
                display_name=PIPELINE_NAME,
                template_path=fe_cfg.pipeline.pkg_path,
                job_id=JOB_NAME,
                pipeline_root=PIPELINE_ROOT,
                enable_caching=True)
            run.submit()

        if args.platform == "kubeflow":
            client = Client(host=args.endpoint)
            experiment = client.create_experiment(name=EXPERIMENT)
            client.run_pipeline(
                experiment_id=experiment.experiment_id,
                job_name=JOB_NAME,
                pipeline_root=PIPELINE_ROOT,
                pipeline_package_path=fe_cfg.pipeline.pkg_path)
