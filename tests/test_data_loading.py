import pandas as pd

from src.data_loading import TARGET_COL, basic_clean


def test_basic_clean_fixes_total_charges_for_zero_tenure():
    df = pd.DataFrame(
        {
            "tenure": [0, 5],
            "TotalCharges": [" ", "120.5"],
            "MonthlyCharges": [29.9, 24.1],
            "SeniorCitizen": [0, 1],
            TARGET_COL: ["No", "Yes"],
        }
    )
    cleaned = basic_clean(df)
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["TotalCharges"].iloc[1] == 120.5
    assert cleaned[TARGET_COL].tolist() == [0, 1]
    assert cleaned["SeniorCitizen"].tolist() == ["No", "Yes"]
