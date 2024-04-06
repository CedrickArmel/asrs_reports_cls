from kfp.dsl import component, InputPath, OutputPath, Input, Output, Dataset


@component
def make_data_available(source_bucket_name: str,
                        source_directory: str,
                        destination_bucket_name: str,
                        destination_directory: OutputPath(str)) -> None:
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

    cfg = OmegaConf.load("../conf/parameters.yaml")
    fe = cfg.feature_engineering

    train_data = pd.read_pickle(train_set.path)[0]
    test_data = pd.read_pickle(test_set.path)[0]
    train_data, test_data = drop_useless(fe.columns_to_keep_from_raw,
                                         train_data=train_data,
                                         test_data=test_data)
    train_data = train_data.dropna(axis=0,
                                   subset=fe.columns_to_keep_from_raw)
    test_data = test_data.dropna(axis=0, subset=fe.columns_to_keep_from_raw)
    train_data[fe.target] = train_data[fe.target].apply(
        lambda cell: encode_cell(cell, fe.labels))
    test_data[fe.target] = test_data[fe.target].apply(
        lambda cell: encode_cell(cell, fe.labels)
    )
    train_data.to_parquet(train_data_out.path)
    test_data.to_parquet(test_data_out.path)
