"""ETL pipeline components tests."""

import os

from dotenv import load_dotenv

from src.etl.components import ETL

# pylint: disable=C0116

load_dotenv()

TEST_DATA = os.getenv("RAW_DATA")

dataset = os.path.join(str(TEST_DATA), "test_dataset.pkl")

components = ETL(fg_name="test_fg")


def test_etl_components(tmp_path):
    output = tmp_path / "output.parquet"
    components.extract(dataset1=dataset, dataset2=dataset, output=output)
    assert os.path.isfile(output)
    gx_suite = tmp_path / "gx_suite.json"
    fg_metadata = tmp_path / "fg_metada.json"
    components.transform_and_load(
        path=output, gx_suite_output=gx_suite, fg_metadata_output=fg_metadata
    )
