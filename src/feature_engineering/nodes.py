from typing import List, Union

import pandas as pd


def drop_useless(columns: List[str],
                 **kwargs) -> Union[List[pd.DataFrame], pd.DataFrame]:
    """Keep only useful columns in the dataframe.
    Args:
        data (str): Name of the dataset as defined in the catalog
    """
    outputs: List[pd.DataFrame] = []
    for _, item in kwargs.items():
        data: pd.DataFrame = item[columns]
        outputs.append(data)
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs


def encode_cell(cell: pd.Series,
                labels: List[str],
                include_other: bool = False) -> pd.Series:
    """Encode the multilabels cell such that the cell content is replaced by \n
    a list of same length as labels and containing 0/1.

    Args:
        cell (pd.Series): cell containing the multilabel target
        labels (list): actual list of labels to classify.

    Returns:
        pd.Series: Expand of the cell with number of cols\n
        equal to number of element in labels.
    """
    cell_anomalies = [item.strip() for item in cell.split(';')]
    splited_cell_anomalies = {label: any(item.startswith(label)
                                         for item in cell_anomalies)
                              for label in labels}
    if include_other:
        splited_cell_anomalies['Other'] = not any(splited_cell_anomalies.
                                                  values())
    return pd.Series(splited_cell_anomalies)
