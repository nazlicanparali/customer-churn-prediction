"""
Veri yükleme ve ilk temizlik.

Telco Customer Churn veri seti bilinen bir kalite sorunuyla gelir:
`TotalCharges` sütunu sayısal olmasına rağmen CSV'de string olarak
saklanır ve tenure=0 olan 11 müşteri için boşluk (" ") içerir
(henüz ilk faturasını almamış yeni müşteriler).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Proje kök dizinine göre mutlak yol (bu dosya src/data_loading.py'de olduğu için
# bir üst dizin proje kökü olur). Böylece script'i nereden çalıştırırsanız çalıştırın
# (proje kökünden, notebooks/ içinden, vb.) veri dosyası doğru bulunur.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = str(PROJECT_ROOT / "data" / "raw" / "telco_customer_churn.csv")
TARGET_COL = "Churn"
ID_COL = "customerID"


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    """Ham CSV'yi olduğu gibi okur."""
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    İlk temizlik adımları:

    1. TotalCharges'ı sayısala çevir (boşluk stringler NaN olur)
    2. tenure=0 olan (henüz faturalanmamış) müşterilerde NaN olan
       TotalCharges'ı 0 ile doldur - bu müşteriler gerçekten hiç
       ödeme yapmamış, missing değil
    3. Hedefi (Churn) 0/1'e çevir (Yes/No -> 1/0)
    4. customerID'yi ayrı tutulmak üzere bırak (model feature'ı DEĞİL, sadece
       kimlik amaçlı bir kolon)
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    zero_tenure_mask = (df["tenure"] == 0) & (df["TotalCharges"].isna())
    df.loc[zero_tenure_mask, "TotalCharges"] = 0.0

    remaining_na = df["TotalCharges"].isna().sum()
    if remaining_na:
        # Kalan nadir durumlar için median imputation (capping mantığıyla tutarlı:
        # satır silme yok)
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)

    # SeniorCitizen zaten 0/1 int ama diğer kategorik kolonlarla tutarlı olması
    # için kategori tipine çeviriyoruz (encoding aşamasında ayrım kolaylaşır)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    return df


def load_clean(path: str = RAW_PATH) -> pd.DataFrame:
    return basic_clean(load_raw(path))


if __name__ == "__main__":
    data = load_clean()
    print(data.shape)
    print(data[TARGET_COL].value_counts(normalize=True))
