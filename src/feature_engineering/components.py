"""Defines the components of the feature engineering pipeline."""
import argparse
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
import hopsworks
from omegaconf import OmegaConf
import pandas as pd

current_dir = str(Path(__file__).parent).split('/')
for i in range(len(current_dir)+1):
    sys.path.append('/'.join(current_dir[:i]))

from src.feature_engineering.nodes import drop_useless, encode_cell

load_dotenv()


def featureengineering(train_set: str,
                       test_set: str,
                       train_data_out: str,
                       test_data_out: str) -> None:
    """Performs the feature engineering step of the pipeline.

    Args:
        train_set (str): /path/to/training_set/ or \
            gs://bucket/path/to/training_set/
        test_set (str): /path/to/validation_set/ or \
            gs://bucket/path/to/validation_set/
        train_data_out (str): /path/to/output/the/train/features/dataset
        test_data_out (str): /path/to/output/the/test/features/dataset
    """
    config = OmegaConf.load("conf/base/feature_engineering.yaml")
    columns = config.components.columns
    labels = config.components.labels
    target = config.components.target

    train_data = pd.read_pickle(train_set)[0]
    test_data = pd.read_pickle(test_set)[0]

    train_data, test_data = drop_useless(
        columns,
        train_data=train_data,
        test_data=test_data)

    train_data = train_data.dropna(
        axis=0,
        subset=train_data.columns.tolist())

    test_data = test_data.dropna(
        axis=0,
        subset=test_data.columns.tolist())

    train_data[target] = train_data[target].\
        apply(lambda cell: encode_cell(cell, labels))
    test_data[target] = test_data[target].\
        apply(lambda cell: encode_cell(cell, labels))

    if not os.path.exists(os.path.dirname(train_data_out)):
        os.makedirs(os.path.dirname(train_data_out))
    if not os.path.exists(os.path.dirname(test_data_out)):
        os.makedirs(os.path.dirname(test_data_out))

    with open(train_data_out, "wb") as f:
        train_data.to_parquet(f)
    with open(test_data_out, "wb") as f:
        test_data.to_parquet(f)


def storefeatures(train_features: str,
                  test_features: str) -> None:
    """Load the features datasets and insert them in a hopsworks feature store.

    Args:
        train_features (str): /path/to/the/train/dataset.
        test_features (Input[Dataset]): /path/to/the/train/dataset.
    """
    with open(train_features, "rb") as f:
        train_data = pd.read_parquet(f)
    with open(test_features, "rb") as f:
        test_data = pd.read_parquet(f)

    config = OmegaConf.load("conf/base/feature_engineering.yaml")
    primary_key = train_data.columns.tolist()

    project = hopsworks.login()
    fs = project.get_feature_store()

    train_data_fg = fs.\
        get_or_create_feature_group(
            version=config.components.feature_group_version,
            primary_key=primary_key,
            name="train_features",
            description="Features to train the model",
            online_enabled=True)

    test_data_fg = fs.\
        get_or_create_feature_group(
            version=config.components.feature_group_version,
            primary_key=primary_key,
            name="test_features",
            description="Features to evaluate the model",
            online_enabled=True)
    train_data_fg.insert(train_data)
    test_data_fg.insert(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Feature Engineering Pipeline components",
        description="""Programme to call the components of the Feature \
            Engineering Pipeline""")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    fe_parser = subparsers.add_parser(
        "featureengineering",
        help="Performs feature engineering step of the pipeline")
    fe_parser.add_argument(
        "--train-set",
        required=True,
        help="/path/to/training_set/ or gs://bucket/path/to/training_set/")
    fe_parser.add_argument(
        "--test-set",
        required=True,
        help="/path/to/validation_set/ or gs://bucket/path/to/validation_set/")
    fe_parser.add_argument(
        "--train-data-out",
        required=True,
        help="/path/to/output/the/train/features/dataset")
    fe_parser.add_argument(
        "--test-data-out",
        required=True,
        help="/path/to/output/the/test/features/dataset")

    store_features_parser = subparsers.add_parser(
        "storefeatures",
        help="Inserts features in hopsworks feature store")
    store_features_parser.add_argument(
        "--train-features",
        required=True,
        help="/path/to/the/train/dataset")
    store_features_parser.add_argument(
        "--test-features",
        required=True,
        help="/path/to/the/test/dataset")

    args = parser.parse_args()

    if args.subcommand == "featureengineering":
        featureengineering(args.train_set,
                           args.test_set,
                           args.train_data_out,
                           args.test_data_out)
    elif args.subcommand == "storefeatures":
        storefeatures(args.train_features,
                      args.test_features)
    else:
        parser.print_help()
