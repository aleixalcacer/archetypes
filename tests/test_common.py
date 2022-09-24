import pytest
from sklearn.utils.estimator_checks import check_estimator

# from archetypes import AA, BiAA  # F401 error
from archetypes import AA


@pytest.mark.parametrize(
    "Estimator",
    [
        AA,
        # BiAA,
    ],
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator())
