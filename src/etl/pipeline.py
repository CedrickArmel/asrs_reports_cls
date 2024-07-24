"""ASRS Report Classification project ETL pipeline built using KFP v2 SDK."""
import os

from dotenv import load_dotenv, dotenv_values
from fire import Fire
from google.cloud import aiplatform
from kfp import Client, compiler
from kfp.dsl import (Artifact, container_component,
                     ContainerSpec, Dataset, importer,
                     Input, Output, pipeline)
from omegaconf import OmegaConf
from wonderwords import RandomWord

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

TRAIN_DATASET = os.path.join(str(RAW_DATA), etlconf.components.train_dataset)
TEST_DATASET = os.path.join(str(RAW_DATA), etlconf.components.test_dataset)


@container_component
def extract(dataset1: Input[Dataset],
            dataset2: Input[Dataset],
            output: Output[Dataset]):
    return ContainerSpec(
        image=IMAGE,
        command=["python"],
        args=["src/etl/components.py",
              "extract",
              "--dataset1", dataset1.path,
              "--dataset2", dataset2.path,
              "--output", output.path])


@container_component
def transform_and_load(data: Input[Dataset],
                       gx_suite_output: Output[Artifact],
                       fg_metadata_output: Output[Artifact]):
    return ContainerSpec(
        image=IMAGE,
        command=["python"],
        args=["src/etl/components.py",
              "transform-and-load",
              "--input", data.path,
              "--gxsuite-output", gx_suite_output.path,
              "--fg-metadata-output", fg_metadata_output.path])


@pipeline(pipeline_root=PIPELINE_ROOT)
def evl_pipeline():
    """EVL pipeline.
    """
    train_set_op = importer(artifact_uri=TRAIN_DATASET,
                            artifact_class=Dataset,
                            reimport=False)
    test_set_op = importer(artifact_uri=TEST_DATASET,
                           artifact_class=Dataset,
                           reimport=False)
    extract_op = extract(dataset1=train_set_op.output,
                         dataset2=test_set_op.output)
    transform_and_load(data=extract_op.output)


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
        compiler.Compiler().compile(pipeline_func=evl_pipeline,
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
        compiler.Compiler().compile(pipeline_func=evl_pipeline,
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
        compiler.Compiler().compile(pipeline_func=evl_pipeline,
                                    package_path=self.template)

    def _get_jobname(self):
        r = RandomWord()
        words = r.random_words(amount=2, word_max_length=10)
        return "-".join(words)


if __name__ == "__main__":
    Fire(Pipeline)
