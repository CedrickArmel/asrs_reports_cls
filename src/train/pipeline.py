"""ASRS Report Classification project Train pipeline
built using KFP v2 SDK."""

import os

from dotenv import load_dotenv, dotenv_values
from fire import Fire
from google.cloud import aiplatform
from kfp import Client, compiler
from kfp.dsl import (Artifact, container_component,
                     ContainerSpec, Dataset, importer,
                     Input, Model, Output, pipeline)
from omegaconf import OmegaConf
from wonderwords import RandomWord

load_dotenv()

core = OmegaConf.load("conf/base/core.yaml")
etlconf = OmegaConf.load("conf/base/etl.yaml")
trainconf = OmegaConf.load("conf/base/train.yaml")

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


@container_component
def create_training_data(td_metadata: Output[Artifact]):
    return ContainerSpec(
        image=IMAGE,
        command=["python"],
        args=["src/train/components.py",
              "create_training_data",
              "--td-metadata-output", td_metadata.path])


@container_component
def train(td_metadata: Input[Artifact],
          state: Input[Artifact],
          checkpoint: Output[Artifact],
          scores: Output[Artifact]):
    return ContainerSpec(
        image=GPU_IMAGE,
        command=["python"],
        args=["src/train/components.py",
              "train",
              "--td_metadata_in", td_metadata.path,
              "--state", state.path,
              "--checkpoint", checkpoint.path,
              "--scores_out", scores.path])


@pipeline(pipeline_root=PIPELINE_ROOT)
def train_pipeline():
    """Train pipeline.
    """
    import_checkpoint_op = importer(
        artifact_uri=CHECKPOINT,
        artifact_class=Model,
        reimport=False)
    create_training_data_op = create_training_data()
    train(td_metadata=create_training_data_op.output,
          state=import_checkpoint_op.output)


class VertexAI(object):
    """Runs the pipeline on Google Cloud Vertex AI ML platform.
    """
    def __init__(self,
                 jobname: str,
                 pipeline_name: str,
                 template: str,
                 root: str | None = None,
                 project_id: str | None = None,
                 region: str | None = None):
        """The default values are those present in the pipeline YAML \
            config file or defined as environnement variables (.env file).

        Args:
            jobname (str): Name of the job.
            pipeline_name (str): User-defined name to display.
            template (str): Local path of the pipeline package (the \
                filename should end with one of the following \
                    .tar.gz, .tgz, .zip, .json).
            root (str | None, optional): Location where to store pipeline's \
                artifacts. Typically a cloud bucket/S3 on the same ML \
                    platform. Defaults to PIPELINE_ROOT env. var.. Defaults to None.
            project_id (str | None, optional): The project that you want \
                to run this PipelineJob in. Defaults to None.
            region (str | None, optional): Location to create PipelineJob. \
                Defaults to None.
        """
        self.jobname = jobname
        self.pipeline_name = pipeline_name
        self.template = template
        self.root = root
        self.project_id = project_id
        self.region = region

    def run(self,
            project: str | None = None,
            region: str | None = None):
        """Runs the pipeline on Google Cloud Vertex AI.

        Args:
            project (str | None, optional): The project that you want \
                to run this PipelineJob in. Defaults to None.
            region (str | None, optional):Location to create PipelineJob. \
                Defaults to None.Ò
        """
        compiler.Compiler().compile(pipeline_func=train_pipeline,
                                    package_path=self.template)
        aiplatform.init(
            project=self.project_id if project is None else project,
            location=self.region if region is None else region)
        run = aiplatform.PipelineJob(
            display_name=self.pipeline_name,
            template_path=self.template,
            job_id=self.jobname,
            pipeline_root=self.root,
            enable_caching=True)
        run.submit()


class Kubeflow(object):
    """Runs the pipeline on a Kubeflow ML platform."""
    def __init__(self,
                 experiment: str,
                 jobname: str,
                 template: str,
                 root: str | None = None,
                 endpoint: str | None = None):
        """The default values are those present in the pipeline YAML \
            config file or defined as environnement variables (.env file).

        Args:
            experiment (str): Experiment name to use. Behaves as \
                namespace.
            jobname (str): Name of the job.
            template (str, optional): Local path of the pipeline package (the \
                filename should end with one of the following \
                    .tar.gz, .tgz, .zip, .json).
            endpoint (str | None, optional): URL to the Kubeflow host endpoint.
            root (str | None, optional): Location where to store pipeline's \
                artifacts(outputs). Typically a cloud bucket/S3 on the same \
                    ML platform.
        """
        self.endpoint = endpoint
        self.experiment = experiment
        self.jobname = jobname
        self.template = template
        self.root = root

    def run(self,
            endpoint: str | None = None,
            experiment: str | None = None):
        """Runs the pipeline on any Kubeflow ML platform.

        Args:
            endpoint (str | None, optional): URL to the Kubeflow host. \
                Defaults to None.
            experiment (str | None, optional): Experiment name to use. \
                Behaves as namespace. Defaults to None.
        """
        compiler.Compiler().compile(pipeline_func=train_pipeline,
                                    package_path=self.template)
        client = Client(host=self.endpoint if endpoint is None else endpoint)
        exp = client.create_experiment(
            name=self.experiment if experiment is None else experiment)
        client.run_pipeline(
            experiment_id=exp.experiment_id,
            job_name=self.jobname,
            pipeline_root=self.root,
            pipeline_package_path=self.template)


class Pipeline(object):
    """This Pipeline Extract the data, Transform, them, and load them
    in Hopsworks Feature Store.
    """
    def __init__(self,
                 template: str = TEMPLATE,
                 endpoint: str | None = ENDPOINT,
                 experiment: str = EXPERIMENT,
                 pipeline_name: str = PIPELINE_NAME,
                 project: str | None = PROJECT_ID,
                 region: str | None = REGION,
                 root: str | None = PIPELINE_ROOT):
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
        self.vertexai = VertexAI(jobname=self._get_jobname(),
                                 pipeline_name=self.pipeline_name,
                                 template=self.template,
                                 root=self.root,
                                 project_id=self.project,
                                 region=self.region)
        self.kubeflow = Kubeflow(jobname=self._get_jobname(),
                                 template=self.template,
                                 root=self.root,
                                 experiment=self.experiment,
                                 endpoint=self.endpoint)

    def compile(self):
        compiler.Compiler().compile(pipeline_func=train_pipeline,
                                    package_path=self.template)

    def _get_jobname(self):
        r = RandomWord()
        words = r.random_words(amount=2, word_max_length=10)
        return "-".join(words)


if __name__ == "__main__":
    Fire(Pipeline)
