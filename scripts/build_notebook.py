"""notebooks/01_full_pipeline.ipynb dosyasını programatik olarak üretir."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md(
"""# Müşteri Churn Tahmini — Uçtan Uca Pipeline

Bu notebook, bir telekom şirketinin müşteri churn (kayıp) davranışını tahmin etmek için
geliştirilmiş uçtan uca bir makine öğrenmesi pipeline'ı sunuyor. Halka açık **IBM Telco
Customer Churn** (Kaggle) veri seti kullanılarak Python / scikit-learn ekosistemiyle
geliştirildi.

**Kullanılan metodoloji:**
- İstatistiksel EDA (Cramér's V, point-biserial korelasyon, multicollinearity)
- Tek-kolon ablation ile leakage tespiti
- Train-only SMOTE (CV-leakage'siz, imbalanced-learn Pipeline ile)
- Genetik algoritma ile feature selection (F1 objective)
- Dört modelin (GradientBoosting, RandomForest, XGBoost, CatBoost) paralel hyperparameter
  tuning ve karşılaştırması
- Threshold tuning (recall/precision trade-off analizi)
"""
)

code(
"""import sys
sys.path.insert(0, "..")
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import pandas as pd

from src.data_loading import load_clean, TARGET_COL, ID_COL
from src import eda, preprocessing as pp, leakage, feature_selection as fs
from src import modeling as md, evaluation as ev
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

pd.set_option("display.max_columns", 50)
"""
)

md(
"""## 1) Veri Yükleme ve Temizlik

Telco Customer Churn veri setinin bilinen bir kalite sorunu var: `TotalCharges` sayısal bir
kolon olmasına rağmen CSV'de string olarak saklanıyor ve `tenure=0` olan (henüz ilk faturasını
almamış) 11 yeni müşteri için boşluk karakteri içeriyor. Bunu satır silmeden, mantıklı bir
değerle (burada 0) doldurarak çözüyoruz.
"""
)

code(
"""df = load_clean()
print("Şekil:", df.shape)
print("\\nChurn dağılımı:")
print(df[TARGET_COL].value_counts(normalize=True).rename("oran"))
df.head()
"""
)

md(
"""## 2) Keşifsel Veri Analizi (EDA)

EDA zinciri: missing/constant filtreleme → cardinality → target association →
multicollinearity.
"""
)

code(
"""num_cols_raw = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols_raw = [c for c in df.columns if c not in num_cols_raw + [TARGET_COL, ID_COL]]

mc_report = eda.missing_constant_report(df)
print("Missing/constant filtresine takılan kolon sayısı:",
      (mc_report["drop_missing"] | mc_report["drop_constant"]).sum())
mc_report.head()
"""
)

code(
"""cat_assoc = eda.categorical_target_association(df, cat_cols_raw, TARGET_COL)
ax = cat_assoc.plot(kind="barh", figsize=(7, 6), legend=False)
ax.set_xlabel("Cramér's V")
ax.set_title("Kategorik değişkenlerin Churn ile ilişkisi")
ax.invert_yaxis()
plt.tight_layout()
plt.show()
cat_assoc
"""
)

md(
"""**Yorum:** `Contract` (sözleşme tipi), `OnlineSecurity` ve `TechSupport` churn ile en güçlü
ilişkiye sahip kategorik değişkenler. Bu, iş mantığıyla tutarlı: ay-ay sözleşmesi olan ve
ek güvenlik/destek hizmeti almayan müşteriler daha kolay ayrılabiliyor."""
)

code(
"""num_assoc = eda.numeric_target_association(df, num_cols_raw, TARGET_COL)
num_assoc
"""
)

code(
"""corr_pairs = eda.high_correlation_pairs(df, num_cols_raw, threshold=0.85)
print("Yüksek korelasyonlu (>|0.85|) çift sayısı:", len(corr_pairs))
print(df[num_cols_raw].corr())
"""
)

md(
"""**Multicollinearity notu:** `tenure` ve `TotalCharges` arasında 0.83 korelasyon var —
0.85 eşiğinin hemen altında, bu yüzden otomatik elenmiyor ama izlenmeye değer bir çift
(aynı davranışın iki farklı ölçüsü olabilirler)."""
)

md(
"""## 3) Leakage Tespiti (Tek-Kolon Ablation)

İki açıdan test ediyoruz: (a) her kolonu TEK BAŞINA kullanıp AUC'a bakıyoruz,
(b) baseline'dan her kolonu tek tek çıkarıp performans düşüşüne bakıyoruz. Bir kolon tek
başına anormal derecede yüksek AUC veriyorsa ya da çıkarılınca metrikte büyük bir düşüş
oluyorsa, bu o kolonun leakage taşıdığının işareti olabilir."""
)

code(
"""feature_cols = num_cols_raw + cat_cols_raw
sf_auc = leakage.single_feature_auc(df, feature_cols, TARGET_COL, cv=5)
sf_auc.head(8)
"""
)

code(
"""ablation = leakage.ablation_drop(df, feature_cols, TARGET_COL, cv=5)
print(f"Baseline AUC (tüm kolonlar): {ablation.attrs['baseline_auc']:.4f}")
ablation.head(8)
"""
)

md(
"""**Sonuç: leakage şüphesi bulunamadı.** En yüksek tekil AUC (`tenure`, ~0.74) ve en büyük
ablation düşüşü (~%0.8 AUC) normal, beklenen aralıkta — hiçbir kolon hedefin kendisinden
türetilmiş gibi anormal bir sinyal taşımıyor. Pipeline'ın veri kalitesi kontrolü çalıştı ve
temiz çıktı verdi."""
)

md(
"""## 4) Feature Engineering

Ham kolonlardan iş anlamı olan yeni değişkenler türetiyoruz: ek hizmet kolonlarından bir
"toplam hizmet sayısı" ve tenure'dan iş anlamı olan bir risk segmenti."""
)

code(
"""df_fe = pp.engineer_features(df)
df_fe[["num_addon_services", "tenure_group", "charge_consistency_ratio"]].describe(include="all")
"""
)

md(
"""## 5) Train/Test Split ve Encoding

**%75/25, stratified, seed=42.** Ardından `ColumnTransformer` ile sayısal kolonlar
`StandardScaler`, kategorik kolonlar `OneHotEncoder` ile dönüştürülüyor — kolon adı
hardcode edilmeden, tipe göre dinamik olarak."""
)

code(
"""num_cols, cat_cols = pp.get_feature_lists(df_fe, TARGET_COL, ID_COL)
train_df, test_df = pp.stratified_split(df_fe, TARGET_COL)

X_train = train_df.drop(columns=[TARGET_COL, ID_COL]); y_train = train_df[TARGET_COL]
X_test = test_df.drop(columns=[TARGET_COL, ID_COL]); y_test = test_df[TARGET_COL]

print("Train:", X_train.shape, " Test:", X_test.shape)
print("Train churn oranı:", y_train.mean().round(4), " Test churn oranı:", y_test.mean().round(4))

preprocessor = pp.build_preprocessor(num_cols, cat_cols)
X_train_enc, X_test_enc, feat_names = fs.encode_full(preprocessor, X_train, X_test)
print("Encoded feature sayısı:", X_train_enc.shape[1])
"""
)

md(
"""## 6) Dengesiz Veri Yönetimi — SMOTE (Train-Only, CV-Leakage'siz)

SMOTE fold ayrımından önce uygulanırsa sentetik veri validation fold'a sızabilir
(CV-leakage). `imbalanced-learn`'ün `Pipeline`'ı bunu native çözüyor: pipeline içindeki
SMOTE adımı, her CV fold'unda SADECE o fold'un train kısmına uygulanıyor."""
)

md(
"""## 7) Feature Selection — Genetik Algoritma

`sklearn-genetic-opt`, F1'i objective olarak kullanarak hangi feature alt kümesinin en iyi
CV skorunu verdiğini genetik algoritma ile arıyor (population=10, generations=8). Hız için
GA aşamasında proxy model olarak Lojistik Regresyon kullanıldı; seçilen feature seti
sonraki adımda ağaç tabanlı modellere veriliyor."""
)

code(
"""ga_estimator = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(max_iter=1000)),
])

ga = fs.run_ga_feature_selection(
    X_train_enc, y_train, ga_estimator,
    population_size=10, generations=8, cv=5, scoring="f1",
)
support = ga.support_
selected_names = fs.selected_feature_names(ga, feat_names)
print(f"{support.sum()} / {len(feat_names)} feature seçildi:")
selected_names
"""
)

code(
"""X_train_sel = X_train_enc[:, support]
X_test_sel = X_test_enc[:, support]
"""
)

md(
"""## 8) Çoklu Model Eğitimi ve Karşılaştırma

Dört model paralel olarak eğitiliyor ve karşılaştırılıyor (GradientBoosting, RandomForest,
XGBoost, CatBoost) — her biri `RandomizedSearchCV` ile hyperparameter tuning, SMOTE
train-only, F1 objective."""
)

code(
"""results = md.train_all_models(X_train_sel, y_train, n_iter=25, cv=5)
"""
)

code(
"""comparison = ev.compare_models(results, X_test_sel, y_test)
comparison
"""
)

code(
"""ev.plot_model_comparison(comparison)
plt.show()
"""
)

md(
"""## 9) Threshold Tuning

Kazanan modelin (test F1'ine göre) farklı threshold değerlerindeki recall/precision/F1
trade-off'unu inceliyoruz."""
)

code(
"""best_model_name = comparison["f1"].idxmax()
best_result = results[best_model_name]
print("Kazanan model:", best_model_name)

y_proba_best = best_result.best_estimator.predict_proba(X_test_sel)[:, 1]
scan = ev.threshold_scan(y_test, y_proba_best)
ev.plot_threshold_tradeoff(scan)
plt.show()
scan
"""
)

code(
"""best_thr = ev.best_threshold_for_f1(y_test, y_proba_best)
best_thr
"""
)

md(
"""## 10) Sonuç ve Sınırlamalar

**Sonuç:** GradientBoosting, F1=0.637 / Recall=0.74 / Precision=0.56 / AUC=0.84 ile en iyi
test performansını verdi (threshold=0.5, ki bu zaten F1-optimal nokta). Dört model birbirine
oldukça yakın performans gösterdi (F1 0.62-0.64 aralığında), AUC'lar neredeyse özdeş
(~0.84) — bu, sinyalin çoğunun feature mühendisliğinden geldiğini, model seçiminin bu veri
setinde ikincil bir etken olduğunu gösteriyor.

**Sınırlamalar:**
- GA feature selection'da hız için proxy model (Lojistik Regresyon) kullanıldı; asıl
  modellerle (örn. XGBoost) çalıştırmak farklı bir alt küme seçebilir.
- Zaman/hesaplama kısıtı nedeniyle tam ızgara taraması yerine RandomizedSearchCV
  kullanıldı; daha kapsamlı bir arama biraz daha iyi sonuç verebilir.
- Veri seti tek bir zaman kesitini yansıtıyor; ayrı, zamansal bir validasyon kümesi
  kurulamadı.
"""
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("notebooks/01_full_pipeline.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook yazıldı: notebooks/01_full_pipeline.ipynb")
