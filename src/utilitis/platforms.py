"""Module defines supported platforms for pipelines run."""

from typing import Callable, Optional

from google.cloud import aiplatform
from kfp import Client, compiler

from src.utilitis.core import get_logger

logger = get_logger(__name__)


class BasePlatform:
    """Base Platform."""

    def __init__(
        self,
        func: Callable,
        jobname: str,
        template: str,
        root: Optional[str] = None,
        **kwargs
    ) -> None:
        """_summary_

        Args:
            func (Callable): Pipeline function.
            jobname (str): Name of the job.
            template (str, optional): Local path of the pipeline package (the \
                filename should end with one of the following \
                    .tar.gz, .tgz, .zip, .json).
            root (Optional[ str ], optional): Location where to store pipeline's \
                artifacts(outputs). Typically a cloud bucket/S3 on the same \
                    ML platform.
        """
        self._kwargs = kwargs
        self.func = func
        self.jobname = jobname
        self.template = template
        self.root = root


class VertexAI(BasePlatform):
    """Runs the pipeline on Google Cloud Vertex AI ML platform."""

    def __init__(
        self,
        pipeline_name: str,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        **kwargs
    ):
        """_summary_

        Args:
            pipeline_name (str): _description_
            project_id (Optional[ str ], optional): _description_. Defaults to None.
            region (Optional[ str ], optional): _description_. Defaults to None.
        """
        super().__init__(**kwargs)
        self.pipeline_name = pipeline_name
        self.project_id = project_id
        self.region = region

    def run(
        self,
        project: Optional[str] = None,
        region: Optional[str] = None,
    ):
        """Runs the pipeline on Google Cloud Vertex AI.

        Args:
            project (str | None, optional): The project that you want \
                to run this PipelineJob in. Defaults to None.
            region (str | None, optional):Location to create PipelineJob. \
                Defaults to None.Ò
        """
        logger.info("🛠️ Setting up pipeline for Vertex AI...🔄")

        logger.info("🛠️ Compiling pipeline... 🔄")
        compiler.Compiler().compile(
            pipeline_func=self.func,
            package_path=self.template,
        )
        logger.info("🛠️ Compiled pipeline successfully!✅")

        logger.info("🛠️ Initializing pipeline for Vertex AI...🔄")
        aiplatform.init(
            project=(self.project_id if project is None else project),
            location=(self.region if region is None else region),
        )
        run = aiplatform.PipelineJob(
            display_name=self.pipeline_name,
            template_path=self.template,
            job_id=self.jobname,
            pipeline_root=self.root,
            enable_caching=True,
        )

        logger.info("⏳ Submiting to Vertex AI...🔄")
        run.submit()

        logger.info("⌛️ Pipeline submited successfully!✅")


class Kubeflow(BasePlatform):
    """Runs the pipeline on a Kubeflow ML platform."""

    def __init__(
        self, experiment: str, endpoint: Optional[str] = None, **kwargs
    ):
        """The default values are those present in the pipeline YAML \
            config file or defined as environnement variables (.env file).

        Args:
            experiment (str): Experiment name to use. Behaves as \
                namespace.
            endpoint (str | None, optional): URL to the Kubeflow host endpoint.
        """
        super().__init__(**kwargs)
        self.endpoint = endpoint
        self.experiment = experiment

    def run(
        self,
        endpoint: Optional[str] = None,
        experiment: Optional[str] = None,
    ):
        """Runs the pipeline on any Kubeflow ML platform.

        Args:
            endpoint (str | None, optional): URL to the Kubeflow host. \
                Defaults to None.
            experiment (str | None, optional): Experiment name to use. \
                Behaves as namespace. Defaults to None.
        """
        logger.info("🛠️ Setting up pipeline for Kubeflow...🔄")

        logger.info("🛠️ Compiling pipeline...🔄")
        compiler.Compiler().compile(
            pipeline_func=self.func,
            package_path=self.template,
        )
        logger.info("🛠️ Compiled pipeline successfully!✅")

        logger.info("🛠️ Initializing pipeline for Kubeflow...🔄")
        client = Client(host=(self.endpoint if endpoint is None else endpoint))
        exp = client.create_experiment(
            name=(self.experiment if experiment is None else experiment)
        )

        logger.info("⏳ Submiting to Kubeflow...🔄")
        client.run_pipeline(
            experiment_id=exp.experiment_id,
            job_name=self.jobname,
            pipeline_root=self.root,
            pipeline_package_path=self.template,
        )
        logger.info("⌛️ Pipeline submited successfully!✅")
