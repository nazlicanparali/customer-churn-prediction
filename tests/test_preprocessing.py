import pandas as pd

from src.preprocessing import ADDON_SERVICE_COLS, engineer_features


def test_engineer_features():
    row = {c: "No" for c in ADDON_SERVICE_COLS}
    row.update(OnlineSecurity="Yes", TechSupport="Yes", tenure=5, MonthlyCharges=20.0, TotalCharges=100.0)
    out = engineer_features(pd.DataFrame([row]))
    assert out["num_addon_services"].iloc[0] == 2
    assert out["tenure_group"].iloc[0] == "0-12m"
    assert out["charge_consistency_ratio"].iloc[0] == 1.0
