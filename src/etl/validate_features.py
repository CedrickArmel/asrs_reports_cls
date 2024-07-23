from great_expectations.core import (ExpectationSuite,
                                     ExpectationConfiguration)


def build_expectation_suite(primary_key: str,
                            name: str,
                            nlabels: int,
                            target: str) -> ExpectationSuite:
    """Return the validation suite for the ETL pipeline.

    Args:
        primary_key (str): Feature to set as primary key.
        name (str): Name of the validation suite.
        nlabels (int): Number of labels in the encoded multiclass target.
        target (str): Target feature.

    Returns:
        ExpectationSuite: The suite to validate the data
    """
    gxsuite = ExpectationSuite(
        expectation_suite_name=name)
    gxsuite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_not_be_null",
            kwargs=dict(column=primary_key,
                        mostly=0.0)))
    gxsuite.add_expectation(
        ExpectationConfiguration(
            expectation_type="expect_column_values_to_match_regex",
            kwargs=dict(
                column=target,
                regex=r"(?:^\[)(?:(?:[01](?:, ){0,1}){len_labels})(?:\])"
                .replace("len_labels", str(nlabels)))))
    return gxsuite
