import pytest
from sklearn.utils.estimator_checks import check_estimator

# from archetypes.sklearn import AA


@pytest.mark.parametrize(
    "Estimator",
    [
        # AA,
        # BiAA,
    ],
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator())
