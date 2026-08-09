import pytest

from src.evaluation.hedge_quality_rubric import automated_hedge_score


@pytest.mark.parametrize(
    ("hedge", "volatility", "expected_score"),
    [
        ("[CONFIDENT]", "immutable", 5),
        ("[TEMPORAL_HEDGE]", "fast", 5),
        ("[CONFIDENT]", "fast", 2),
        ("[UNKNOWN]", "immutable", 1),
        ("[CONFIDENT]", "unknown", 3),
    ],
)
def test_automated_hedge_score(hedge, volatility, expected_score):
    assert automated_hedge_score(hedge, volatility) == expected_score
