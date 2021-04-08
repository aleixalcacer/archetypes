import pytest

from sklearn.utils.estimator_checks import check_estimator

from archetypes import AA, BiAA


@pytest.mark.parametrize(
    "Estimator", [BiAA, AA]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator())
