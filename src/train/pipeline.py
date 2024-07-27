"""ASRS Report Classification project Train pipeline built using KFP v2 SDK."""

import os

from dotenv import load_dotenv
from fire import Fire
from kfp import compiler
from kfp.dsl import (
    Artifact,
    ContainerSpec,
    Input,
    Model,
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

core = OmegaConf.load("conf/base/core.yaml")
etlconf = OmegaConf.load("conf/base/etl.yaml")
trainconf = OmegaConf.load("conf/base/train.yaml")

ACCELERATOR_LIMIT = trainconf.training.accelerator_limit
ACCELERATOR_TYPE = trainconf.training.accelerator_type
CHECKPOINT = trainconf.model.checkpoint
ENDPOINT = os.getenv("ENDPOINT")
EXPERIMENT = trainconf.pipeline.experiment
GPU_IMAGE = os.getenv("IMAGE")
IMAGE = os.getenv("GPU_IMAGE")
PIPELINE_NAME = trainconf.pipeline.name
PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")
TEMPLATE = trainconf.pipeline.template


# pylint: disable=C0116
@container_component
def create_training_data(
    td_metadata: Output[Artifact],
):
    return ContainerSpec(
        image=IMAGE,
        command=["python"],
        args=[
            "src/train/components.py",
            "create_training_data",
            "--td-metadata-output",
            td_metadata.path,
        ],
    )


@container_component
def train(
    td_metadata: Input[Artifact],
    state: Input[Artifact],
    checkpoint: Output[Artifact],
    scores: Output[Artifact],
):
    return ContainerSpec(
        image=GPU_IMAGE,
        command=["python"],
        args=[
            "src/train/components.py",
            "train",
            "--td_metadata_in",
            td_metadata.path,
            "--state",
            state.path,
            "--checkpoint",
            checkpoint.path,
            "--scores_out",
            scores.path,
        ],
    )


@pipeline(pipeline_root=PIPELINE_ROOT)
def train_pipeline():
    """Train pipeline."""
    # pylint: disable=E1101,E1120
    import_checkpoint_op = importer(
        artifact_uri=CHECKPOINT,
        artifact_class=Model,
        reimport=False,
    )
    create_training_data_op = create_training_data()
    train(
        td_metadata=create_training_data_op.output,
        state=import_checkpoint_op.output,
    ).add_node_selector_constraint(ACCELERATOR_TYPE).set_accelerator_type(
        ACCELERATOR_TYPE
    ).set_accelerator_limit(
        ACCELERATOR_LIMIT
    )


class Pipeline(object):
    """This Pipeline Extract the data, Transform, them, and load them
    in Hopsworks Feature Store.
    """

    def __init__(
        self,
        template: str = TEMPLATE,
        endpoint: str | None = ENDPOINT,
        experiment: str = EXPERIMENT,
        pipeline_name: str = PIPELINE_NAME,
        project: str | None = PROJECT_ID,
        region: str | None = REGION,
        root: str | None = PIPELINE_ROOT,
    ):
        """The default values are those present in the pipeline YAML \
            config file or defined as environnement variables (.env file).

        Args:
            template (str, optional): Path to pipeline YAML definition.\
                Defaults from config file.
            endpoint (str | None, optional): URL to the Kubeflow \
                endpoint. Defaults to ENDPOINT env. var..
            experiment (str, optional): Experiment name to use. Behaves as \
                namespace. Defaults from config file.
            pipeline_name (str, optional): Pipeline name to display. \
                Defaults from config file.
            project (str | None, optional): GCP PROJECT_ID. Necessary to run \
                on Vertex AI. Defaults to PROJECT_ID env. var..
            region (str | None, optional): GCP REGION. Necessary to run \
                on Vertex AI. Defaults to REGION env. var..
            root (str | None, optional): Location where to store pipeline's \
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
            func=train_pipeline,
            jobname=self._get_jobname(),
            pipeline_name=self.pipeline_name,
            template=self.template,
            root=self.root,
            project_id=self.project,
            region=self.region,
        )
        self.kubeflow = Kubeflow(
            func=train_pipeline,
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
            pipeline_func=train_pipeline,
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
