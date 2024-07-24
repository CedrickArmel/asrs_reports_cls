"""Defines the components of the ETL pipeline."""
import os

from dotenv import load_dotenv
from fire import Fire
import hopsworks
import json
from omegaconf import OmegaConf
import pandas as pd

from src.etl.transformations import encode_cell
from src.etl.validate_features import build_expectation_suite

load_dotenv()

core = OmegaConf.load("conf/base/core.yaml")
etlconf = OmegaConf.load("conf/base/etl.yaml")

COLUMNS = etlconf.components.columns
FG_DESCRIPTION = core.feature_group.description
FG_NAME = core.feature_group.name
FG_ONLINE = core.feature_group.online
FG_VERSION = core.feature_group.version
GX_SUITE_NAME = core.gx_suite.name
LABELS = etlconf.components.labels
PRIMARY_KEY = core.feature_group.primary_key
TARGET = etlconf.components.target

project = hopsworks.login()
fs = project.get_feature_store()


class ETL(object):
    """ETL pipeline components."""
    def extract(self,
                dataset1: str,
                dataset2: str,
                output: str) -> None:
        """Extract the data sets from the data lake (GCS bucket),
        clean them and store them in parquet format.

        Args:
            dataset1 (str): path to the first data set.
            dataset2 (str): path to the 2nd data set.
            output (str): path to where to store the concatenated data sets.
        """
        data1 = pd.read_pickle(dataset1)[0]
        data2 = pd.read_pickle(dataset2)[0]
        data = pd.concat([data1, data2]).reset_index()
        data = data.dropna(axis=0, subset=COLUMNS)
        data = data[COLUMNS]
        data.columns = [col.lower() for col in data.columns]

        if not os.path.exists(os.path.dirname(output)):
            os.makedirs(os.path.dirname(output))

        with open(output, "wb") as f:
            data.to_parquet(f)

    def transform_and_load(self,
                           input: str,
                           gx_suite_output: str,
                           fg_metadata_output: str) -> None:
        """Compute the features, validate them and load them to the \
            feature store.

        Args:
            input (str): Path to the data to load to the feature store after \
                transformation.
            gx_suite_output (str): Location where to store de used \
                Great Expectation suite Artefact.
            fg_metadata_output (str): Location where to store feature \
                group metadata
        """
        with open(input, "rb") as f:
            data = pd.read_parquet(f)

        # Transform the target colum: "label1;...;labeln" => [1,...,1]
        data[TARGET] = data[TARGET].\
            apply(lambda cell: encode_cell(cell, LABELS))

        gxsuite = build_expectation_suite(primary_key=PRIMARY_KEY,
                                          name=GX_SUITE_NAME,
                                          nlabels=len(LABELS),
                                          target=TARGET)
        fg = fs.\
            get_or_create_feature_group(
                name=FG_NAME,
                version=FG_VERSION,
                primary_key=PRIMARY_KEY,
                description=FG_DESCRIPTION,
                expectation_suite=gxsuite,  # validate data before ingestion
                online_enabled=FG_ONLINE)

        fg.insert(data, wait=True)

        if not os.path.exists(os.path.dirname(gx_suite_output)):
            os.makedirs(os.path.dirname(gx_suite_output))

        if not os.path.exists(os.path.dirname(fg_metadata_output)):
            os.makedirs(os.path.dirname(fg_metadata_output))

        with open(gx_suite_output, 'w') as f:
            f.write(json.dumps(gxsuite.to_json_dict()))
        with open(fg_metadata_output, 'w') as f:
            f.write(json.dumps(fg.to_dict()))


if __name__ == "__main__":
    Fire(ETL)
