"""Defines the data transformations to be used in the pipeline components."""
from typing import List, Union

import pandas as pd


def drop_useless(columns: List[str],
                 **kwargs) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """Keep only useful columns in the dataframes \
        (eg. train and testset given as kwargs).

    Args:
        columns (List[str]): The columns, common to the dataset given as\
              kwargs, to return in the outputs.

    Returns:
        Union[List[pd.DataFrame], pd.DataFrame]: The transformed dataframes.
    """

    outputs: List[pd.DataFrame] = []
    for _, item in kwargs.items():
        data: pd.DataFrame = item[columns]
        outputs.append(data)
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def encode_cell(cell: str,
                labels: List[str],
                include_other: bool = False) -> List[int]:
    """Encode the multilabels cell such that the cell content is replaced by \
        a list of same length as labels and containing 0/1.

    Args:
        cell (str): semicolon separated string of labels.
        labels (List[str]): actual list of labels to classify to use \
            to regroup the raw labels in the data.
        include_other (bool, optional): Whether to include optional "other"\
              label. Defaults to False.

    Returns:
        List: Multilabel one-hot encoded list.
    """

    cell_anomalies = [item.strip() for item in cell.split(';')]
    splited_cell_anomalies = {label: 1 if
                              any(item.startswith(label)
                                  for item in cell_anomalies)
                              else 0 for label in labels}
    if include_other:
        splited_cell_anomalies['Other'] = 1 if not any(splited_cell_anomalies.
                                                       values()) else 0
    encodings = list(splited_cell_anomalies.values())
    return encodings
