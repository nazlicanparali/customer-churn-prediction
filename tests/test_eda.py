import pandas as pd
import pytest

from src import eda


def test_cramers_v_perfect_association():
    # x, y birebir aynı bilgiyi taşıyor -> Cramér's V ~ 1.0 olmalı
    x = pd.Series(["a", "a", "b", "b"] * 10)
    y = pd.Series([0, 0, 1, 1] * 10)
    assert eda.cramers_v(x, y) == pytest.approx(1.0)


def test_cramers_v_no_association():
    # x rastgele/dengeli dağılmış, y ile ilişkisiz -> V yaklaşık 0
    x = pd.Series(["a", "b"] * 50)
    y = pd.Series(([0, 1] * 25) + ([1, 0] * 25))
    assert eda.cramers_v(x, y) < 0.2


def test_missing_constant_report_flags_constant_column():
    df = pd.DataFrame(
        {
            "sabit_kolon": ["x"] * 100,
            "degisken_kolon": list(range(100)),
        }
    )
    report = eda.missing_constant_report(df, missing_thresh=0.75, constant_thresh=0.90)
    assert bool(report.loc["sabit_kolon", "drop_constant"]) is True
    assert bool(report.loc["degisken_kolon", "drop_constant"]) is False


def test_apply_missing_constant_filter_drops_expected_columns():
    df = pd.DataFrame(
        {
            "cok_eksik": [None] * 90 + list(range(10)),
            "normal": list(range(100)),
        }
    )
    filtered, dropped = eda.apply_missing_constant_filter(df, missing_thresh=0.75)
    assert "cok_eksik" in dropped
    assert "normal" not in dropped
    assert "cok_eksik" not in filtered.columns
