"""Defines the data transformations to be used in the ETL components."""
from typing import List


def encode_cell(cell: str,
                labels: List[str]) -> List[int]:
    """Encode the multilabels cell such that the cell content is replaced by \
        a list of same length as labels and containing 0/1.

    Args:
        cell (str): semicolon separated string of labels.
        labels (List[str]): actual list of labels to classify to use \
            to regroup the raw labels in the data.

    Returns:
        List: Multilabel one-hot encoded list.
    """

    cell_anomalies = [item.strip() for item in cell.split(';')]
    splited_cell_anomalies = {label: 1 if
                              any(item.startswith(label)
                                  for item in cell_anomalies)
                              else 0 for label in labels}

    encodings = list(splited_cell_anomalies.values())
    return encodings
