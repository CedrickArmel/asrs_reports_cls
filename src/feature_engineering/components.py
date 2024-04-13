from kfp.dsl import component, OutputPath, Input, Output, Dataset


@component
def make_data_available(source_bucket_name: str,
                        source_directory: str,
                        destination_bucket_name: str,
                        destination_directory: OutputPath(str)) -> None:
    """Imports the raw data from the storage location and store them where \
        they can be available for the pipeline job.

    Args:
        source_bucket_name (str): raw-data-location-bucket-name.
        source_directory (str): raw-data-location-directory.
        destination_bucket_name (str): destination-bucket-name.
        destination_directory (OutputPath): destination-directory.
    """
    from utilitis.gcpstorage import copy_many_blobs
    copy_many_blobs(bucket_name=source_bucket_name,
                    folder_path=source_directory,
                    destination_bucket_name=destination_bucket_name,
                    destination_folder_path=destination_directory)


@component
def feature_engineering(train_set: Input[Dataset],
                        test_set: Input[Dataset],
                        train_data_out: Output[Dataset],
                        test_data_out: Output[Dataset]) -> None:
    import pandas as pd
    from omegaconf import OmegaConf
    from feature_engineering.nodes import drop_useless, encode_cell

    fe_cfg = OmegaConf.load("../conf/base/feature_engineering.yaml")

    train_data = pd.read_pickle(train_set.path)[0]
    test_data = pd.read_pickle(test_set.path)[0]
    train_data, test_data = drop_useless(fe_cfg.columns_to_keep_from_raw,
                                         train_data=train_data,
                                         test_data=test_data)
    train_data = train_data.dropna(axis=0,
                                   subset=fe_cfg.columns_to_keep_from_raw)
    test_data = test_data.dropna(axis=0, subset=fe_cfg.
                                 columns_to_keep_from_raw)
    train_data[fe_cfg.target] = train_data[fe_cfg.target].apply(
        lambda cell: encode_cell(cell, fe_cfg.labels))
    test_data[fe_cfg.target] = test_data[fe_cfg.target].apply(
        lambda cell: encode_cell(cell, fe_cfg.labels)
    )
    train_data.to_parquet(train_data_out.path)
    test_data.to_parquet(test_data_out.path)


@component
def store_features(train_features: Input[Dataset],
                   test_features: Input[Dataset]) -> None:
    """Load the features datasets and insert them in a hopsworks feature store.

    Args:
        train_features (Input[Dataset]): input/train/features/location.
        test_features (Input[Dataset]): input/test/features/location.
    """

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
