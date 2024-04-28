
"""Defines the kfp components of the feature engineering pipeline."""
import os
from typing import Dict

from dotenv import load_dotenv
from kfp.dsl import component, Input, Dataset, Output

load_dotenv()
IMAGE = os.getenv("IMAGE")


@component(base_image=IMAGE)
def feature_engineering(train_set: str,
                        test_set: str,
                        train_data_out: Output[Dataset],
                        test_data_out: Output[Dataset]) -> None:
    """KFP component that performs the feature engineering\
        step of the pipeline.

    Args:
        train_set (str): /path/to/training_set/ or \
            gs://bucket/path/to/training_set/
        test_set (str): /path/to/validation_set/ or \
            gs://bucket/path/to/validation_set/
        config (DictConfig): An OmegaConf configuration object.
        train_data_out (Output[Dataset]): KFP component output object \
            available at ${PIPELINE_ROOT}/train_data_out.path.\
                See KFP documentation.
        test_data_out (Output[Dataset]): KFP component output object\
              available at ${PIPELINE_ROOT}/test_data_out.path.\
                See KFP documentation.
    """
    from pathlib import Path
    import sys
    current_dir = str(Path(__file__).parent).split('/')
    for i in range(len(current_dir)+1):
        sys.path.append('/'.join(current_dir[:i]))
    
    from dotenv import load_dotenv
    from omegaconf import OmegaConf
    import pandas as pd
    from src.feature_engineering.nodes import drop_useless, encode_cell

    load_dotenv()
    config = OmegaConf.load("conf/base/feature_engineering.yaml")
    columns = config.components.columns
    labels = config.components.labels
    target = config.components.target

    train_data = pd.read_pickle(train_set)[0]
    test_data = pd.read_pickle(test_set)[0]
    train_data, test_data = drop_useless(columns,
                                         train_data=train_data,
                                         test_data=test_data)
    train_data = train_data.dropna(axis=0,
                                   subset=columns)
    test_data = test_data.dropna(axis=0,
                                 subset=columns)
    train_data[target] = train_data[target].\
        apply(lambda cell: encode_cell(cell, labels))
    test_data[target] = test_data[target].\
        apply(lambda cell: encode_cell(cell, labels))
    train_data.to_parquet(train_data_out.path)
    test_data.to_parquet(test_data_out.path)


@component(base_image=IMAGE)
def make_data_available(source_bucket_name: str,
                        source_directory: str,
                        destination_bucket_name: str,
                        destination_directory: str) -> None:
    """Imports the raw data from the storage location and store them where \
        they can be available for the pipeline job.

    Args:
        source_bucket_name (str): raw-data-location-bucket-name.
        source_directory (str): raw-data-location-directory.
        destination_bucket_name (str): destination-bucket-name.
        destination_directory (OutputPath): destination-directory.
    """
    from pathlib import Path
    import sys
    current_dir = str(Path(__file__).parent).split('/')
    for i in range(len(current_dir)+1):
        sys.path.append('/'.join(current_dir[:i]))

    from dotenv import load_dotenv
    from src.utilitis.gcpstorage import copy_many_blobs

    load_dotenv()

    copy_many_blobs(bucket_name=source_bucket_name,
                    folder_path=source_directory,
                    destination_bucket_name=destination_bucket_name,
                    destination_folder_path=destination_directory)


@component(base_image=IMAGE)
def store_features(train_features: Input[Dataset],
                   test_features: Input[Dataset]) -> None:
    """Load the features datasets and insert them in a hopsworks feature store.

    Args:
        train_features (Input[Dataset]): KFP component input object available\
              at ${PIPELINE_ROOT}/train_features.path.
        test_features (Input[Dataset]): KFP component input object available\
              at ${PIPELINE_ROOT}/test_features.path.
    """
    from pathlib import Path
    import sys
    current_dir = str(Path(__file__).parent).split('/')
    for i in range(len(current_dir)+1):
        sys.path.append('/'.join(current_dir[:i]))

    from dotenv import load_dotenv
    import hopsworks
    import pandas as pd

    load_dotenv()

    project = hopsworks.login()
    fs = project.get_feature_store()

    train_data = pd.read_parquet(train_features.path)
    test_data = pd.read_parquet(test_features.path)

    train_data_fg = fs.\
        get_or_create_feature_group(name="train_features",
                                    description="Features to train the model",
                                    online_enabled=True)
    test_data_fg = fs.\
        get_or_create_feature_group(name="test_features",
                                    description="Features to evaluate the model",
                                    online_enabled=True)
    train_data_fg.insert(train_data)
    test_data_fg.insert(test_data)
