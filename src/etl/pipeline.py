"""ASRS Report Classification project ETL pipeline built using KFP v2 SDK."""

import os
from typing import Optional

from dotenv import load_dotenv
from fire import Fire
from kfp import compiler
from kfp.dsl import (
    Artifact,
    ContainerSpec,
    Dataset,
    Input,
    Output,
    container_component,
    importer,
    pipeline,
)
from omegaconf import OmegaConf
from wonderwords import RandomWord

from src.utilitis.core import get_logger
from src.utilitis.platforms import Kubeflow, VertexAI

logger = get_logger(__name__)

load_dotenv()
etlconf = OmegaConf.load("conf/base/etl.yaml")

ENDPOINT = os.getenv("ENDPOINT")
EXPERIMENT = etlconf.pipeline.experiment
IMAGE = os.getenv("IMAGE")
PIPELINE_NAME = etlconf.pipeline.name
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
RAW_DATA = os.getenv("RAW_DATA")
TEMPLATE = etlconf.pipeline.template

TRAIN_DATASET = os.path.join(
    str(RAW_DATA),
    etlconf.components.train_dataset,
)
TEST_DATASET = os.path.join(
    str(RAW_DATA),
    etlconf.components.test_dataset,
)

# pylint: disable=C0116


@container_component
def extract(
    dataset1: Input[Dataset],
    dataset2: Input[Dataset],
    output: Output[Dataset],
):
    return ContainerSpec(
        image=IMAGE,
        command=["python"],
        args=[
            "src/etl/components.py",
            "extract",
            "--dataset1",
            dataset1.path,
            "--dataset2",
            dataset2.path,
            "--output",
            output.path,
        ],
    )


@container_component
def transform_and_load(
    data: Input[Dataset],
    gx_suite_output: Output[Artifact],
    fg_metadata_output: Output[Artifact],
):
    return ContainerSpec(
        image=IMAGE,
        command=["python"],
        args=[
            "src/etl/components.py",
            "transform-and-load",
            "--path",
            data.path,
            "--gxsuite-output",
            gx_suite_output.path,
            "--fg-metadata-output",
            fg_metadata_output.path,
        ],
    )


@pipeline(pipeline_root=PIPELINE_ROOT)
def etl_pipeline():
    """ETL pipeline."""
    # pylint: disable=E1120,E1101
    train_set_op = importer(
        artifact_uri=TRAIN_DATASET,
        artifact_class=Dataset,
        reimport=False,
    )
    test_set_op = importer(
        artifact_uri=TEST_DATASET,
        artifact_class=Dataset,
        reimport=False,
    )
    extract_op = extract(
        dataset1=train_set_op.output,
        dataset2=test_set_op.output,
    )
    transform_and_load(data=extract_op.output)


class Pipeline:
    """This Pipeline Extract the data, Transform, them, and load them
    in Hopsworks Feature Store.
    """

    def __init__(
        self,
        template: str = TEMPLATE,
        endpoint: Optional[str] = ENDPOINT,
        experiment: str = EXPERIMENT,
        pipeline_name: str = PIPELINE_NAME,
        project: Optional[str] = PROJECT_ID,
        region: Optional[str] = REGION,
        root: Optional[str] = PIPELINE_ROOT,
    ):
        """The default values are those present in the pipeline YAML \
            config file or defined as environnement variables (.env file).

        Args:
            template (str, optional): Path to pipeline YAML definition.\
                Defaults from config file.
            endpoint (Optional[str], optional): URL to the Kubeflow \
                endpoint. Defaults to ENDPOINT env. var..
            experiment (str, optional): Experiment name to use. Behaves as \
                namespace. Defaults from config file.
            pipeline_name (str, optional): Pipeline name to display. \
                Defaults from config file.
            project (Optional[str], optional): GCP PROJECT_ID. Necessary to run \
                on Vertex AI. Defaults to PROJECT_ID env. var..
            region (Optional[str], optional): GCP REGION. Necessary to run \
                on Vertex AI. Defaults to REGION env. var..
            root (Optional[str], optional): Location where to store pipeline's \
                artifacts. Typically a cloud bucket/S3 on the same ML \
                    platform. Defaults to PIPELINE_ROOT env. var..
        """
        self.template = template
        self.endpoint = endpoint
        self.experiment = experiment
        self.pipeline_name = pipeline_name
        self.project = project
        self.region = region
        self.root = root
        self.vertexai = VertexAI(
            func=etl_pipeline,
            jobname=self._get_jobname(),
            pipeline_name=self.pipeline_name,
            template=self.template,
            root=self.root,
            project_id=self.project,
            region=self.region,
        )
        self.kubeflow = Kubeflow(
            func=etl_pipeline,
            jobname=self._get_jobname(),
            template=self.template,
            root=self.root,
            experiment=self.experiment,
            endpoint=self.endpoint,
        )

    def compile(
        self,
    ):
        logger.info("🛠️ Compiling pipeline...🔄")
        compiler.Compiler().compile(
            pipeline_func=etl_pipeline,
            package_path=self.template,
        )
        logger.info("🛠️ Compiled pipeline successfully!✅")

    def _get_jobname(
        self,
    ):
        r = RandomWord()
        words = r.random_words(
            amount=2,
            word_max_length=10,
        )
        return "-".join(words)


if __name__ == "__main__":
    Fire(Pipeline)
