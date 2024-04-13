import argparse
from datetime import datetime
import os

from dotenv import load_dotenv
from google.cloud import aiplatform
from kfp import compiler
from kfp.dsl import pipeline, OutputPath
from omegaconf import OmegaConf

from feature_engineering.components import (feature_engineering,
                                            make_data_available,
                                            store_features)

load_dotenv()

fe_cfg = OmegaConf.load("../conf/base/feature_engineering.yaml")

DEST_BUCKET_NAME = os.getenv("DEST_BUCKET_NAME")
DEST_DIR = os.getenv("DEST_DIR")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
GCP_PROJECT_REGION = os.getenv("GCP_PROJECT_REGION")
PIPELINE_NAME = fe_cfg.pipeline_name
PIPELINES_ROOT = os.getenv("PIPELINES_ROOT")
SOURCE_BUCKET_NAME = os.getenv("SOURCE_BUCKET_NAME")
SOURCE_DIR = os.getenv("SOURCE_DIR")


@pipeline(name=PIPELINE_NAME,
          pipeline_root=PIPELINES_ROOT)
def feature_engineering_pipeline(source_bucket_name: str,
                                 source_directory: str,
                                 destination_bucket_name: str,
                                 destination_directory: OutputPath(str)):
    """Feature engineering pipeline.

    Args:
        source_bucket_name (str): raw-data-location-bucket-name.
        source_directory (str): raw-data-location-directory.
        destination_bucket_name (str): destination-bucket-name.
        destination_directory (OutputPath): destination-directory.
    """
    make_data_available(
        source_bucket_name=source_bucket_name,
        source_directory=source_directory,
        destination_bucket_name=destination_bucket_name,
        destination_directory=destination_directory)

    feature_engineering_op = feature_engineering(
        train_set=fe_cfg.train_dataset,
        test_set=fe_cfg.test_dataset,
        train_data_out=fe_cfg.train_feat,
        test_data_out=fe_cfg.est_feat)

    store_features(
        train_features=feature_engineering_op.outputs["train_data_out"],
        test_features=feature_engineering_op.outputs["test_data_out"])


if __name__ == "__main__":
    aiplatform.init(project=GCP_PROJECT_ID, location=GCP_PROJECT_REGION)

    TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M")

    parser = argparse.ArgumentParser()
    parser.add_argument('--compile-only',
                        action='store_true')
    args = parser.parse_args()

    compiler.Compiler().compile(pipeline_func=feature_engineering_pipeline,
                                package_path=fe_cfg.package_path)

    if not args.compile_only:
        run = aiplatform.PipelineJob(
            project=GCP_PROJECT_ID,
            location=GCP_PROJECT_REGION,
            display_name=PIPELINE_NAME,
            template_path=fe_cfg.package_path,
            job_id=f"{PIPELINE_NAME}-{TIMESTAMP}",
            pipeline_root=PIPELINES_ROOT,
            enable_caching=fe_cfg.enable_caching,
            parameter_values=dict(
                source_bucket_name=SOURCE_BUCKET_NAME,
                source_directory=SOURCE_DIR,
                destination_bucket_name=DEST_BUCKET_NAME,
                destination_directory=DEST_DIR))
        run.submit()
