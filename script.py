import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
df=pd.read_excel("Online Retail.xlsx")
print("###-Head")
print(df.head())
print("###-Info")
print(df.info())
print("###-Missing Values")
print(df.isnull().sum())
print("Doublons :", df.duplicated().sum())
print("Quantités négatives :", (df["Quantity"] < 0).sum())
print("Quantités nulles :", (df["Quantity"] == 0).sum())
print("Prix négatifs :", (df["UnitPrice"] < 0).sum())
print("Prix nuls :", (df["UnitPrice"] == 0).sum())
print(
    "Factures commençant par C :",
    df["InvoiceNo"].astype(str).str.startswith("C").sum()
)
print("Date minimale :", df["InvoiceDate"].min())
print("Date maximale :", df["InvoiceDate"].max())
duplicates=df[df.duplicated(keep=False)]
print(duplicates.sort_values(
    by=["InvoiceNo", "StockCode", "InvoiceDate"]
).head(20))
negative_qty = df[df["Quantity"] < 0]

print(negative_qty.head(10))

print("Factures commençant par C :")
print(
    negative_qty[
        negative_qty["InvoiceNo"].astype(str).str.startswith("C")
    ].head(10)
)
invalid_price = df[df["UnitPrice"] <= 0]

print(invalid_price.head(20))

print("Répartition des prix invalides :")
print(invalid_price["UnitPrice"].value_counts())

print("Factures concernées :")
print(invalid_price["InvoiceNo"].value_counts().head(10))

### Remove duplicates
df_original=df.copy()
df=df.drop_duplicates()
print("Duplicates:",df.duplicated().sum())

###RFM
sales=df[
    (df["CustomerID"].notna())&(df["Quantity"]>0)&(df["UnitPrice"]>0)
].copy()
print("Sales", len(sales))
print("Customers [SALES]:", sales["CustomerID"].nunique())
print("invoices [SALES]:", sales["InvoiceNo"].nunique())
sales["CA"]=sales["Quantity"] * sales["UnitPrice"]
print(sales.head())
ref_date=pd.Timestamp("2011-12-10")
rfm=sales.groupby("CustomerID").agg(
    Recency=("InvoiceDate",lambda x:(ref_date-x.max()).days),
    Frequency=("InvoiceNo","nunique"),
    Monetary=("CA","sum")
)
print(rfm.head())
print("Nombre de clients :", len(rfm))
print(rfm.head())
##Missing VALUES
print("Missing values: ",rfm.isnull())
print("Min values: ",rfm.min())
print("Max values: ",rfm.max())

rfm.describe()

rfm.isnull().sum()

rfm.min()

rfm.max()

for col in rfm.columns:
  q1=rfm[col].quantile(0.25)
  q3=rfm[col].quantile(0.75)
  iqr=q3-q1
  lower=q1-1.5*iqr
  upper=q3+1.5*iqr
  outliers=rfm[(rfm[col]<lower)|(rfm[col]>upper)]
  print(outliers.shape)
rfm.shape

print(rfm.skew())

rfm_log = rfm[["Recency", "Frequency", "Monetary"]].copy()

rfm_log["Frequency"] = np.log1p(rfm_log["Frequency"])
rfm_log["Monetary"] = np.log1p(rfm_log["Monetary"])
rfm_log["Recency"] = np.log1p(rfm["Recency"])

cols = ["Recency", "Frequency", "Monetary"]

print("Avant transformation :")
print(rfm[cols].skew())

print("\nAprès transformation :")
print(rfm_log[cols].skew())

import numpy as np
from sklearn.preprocessing import StandardScaler

rfm_model = rfm[["Recency", "Frequency", "Monetary"]].copy()

rfm_model["Frequency"] = np.log1p(rfm_model["Frequency"])
rfm_model["Monetary"] = np.log1p(rfm_model["Monetary"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(rfm_model)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertias = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertias, marker="o")
plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Inertie")
plt.title("Méthode du coude - Elbow Method")
plt.xticks(list(k_values))
plt.grid(True)
plt.show()

from sklearn.metrics import silhouette_score

silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

    print(f"K = {k} | Silhouette score = {score:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(
    list(range(2, 11)),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Nombre de clusters (K)")
plt.ylabel("Silhouette score")
plt.title("Silhouette Score selon K")
plt.xticks(list(range(2, 11)))
plt.grid(True)
plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(X_scaled)

print(rfm["Cluster"].value_counts().sort_index())

cluster_profile = rfm.groupby("Cluster").agg(
    Nombre_clients=("Recency", "size"),
    Recency_moyenne=("Recency", "mean"),
    Frequency_moyenne=("Frequency", "mean"),
    Monetary_moyen=("Monetary", "mean")
).round(2)

print(cluster_profile)

cluster_profile_robust = rfm.groupby("Cluster").agg(
    Nombre_clients=("Recency", "size"),
    Recency_mediane=("Recency", "median"),
    Frequency_mediane=("Frequency", "median"),
    Monetary_mediane=("Monetary", "median"),

    Monetary_Q1=("Monetary", lambda x: x.quantile(0.25)),
    Monetary_Q3=("Monetary", lambda x: x.quantile(0.75))
).round(2)

print(cluster_profile_robust)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"],
    index=rfm.index
)

df_pca["Cluster"] = rfm["Cluster"].values

plt.figure(figsize=(10, 6))

for cluster in sorted(df_pca["Cluster"].unique()):
    subset = df_pca[df_pca["Cluster"] == cluster]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=f"Cluster {cluster}",
        alpha=0.5,
        s=15
    )

plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Segmentation des clients RFM avec K-Means")
plt.legend()
plt.grid(True)
plt.show()

print(
    "Variance expliquée par PC1 et PC2 :",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

print("Variance PC1 :", round(pca.explained_variance_ratio_[0] * 100, 2), "%")
print("Variance PC2 :", round(pca.explained_variance_ratio_[1] * 100, 2), "%")

print(
    "Variance totale expliquée :",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

print(rfm["Cluster"].value_counts().sort_index())





from sklearn.metrics import silhouette_score, silhouette_samples
score_global = silhouette_score(X_scaled, rfm["Cluster"])
print(f"Silhouette globale:{score_global:.4f}")

scores = silhouette_samples(X_scaled, rfm["Cluster"])

silhouette_df = pd.DataFrame({
    "Cluster": rfm["Cluster"].values,
    "Silhouette": scores
}, index=rfm.index)

evaluation = silhouette_df.groupby("Cluster").agg(
    Score_moyen=("Silhouette", "mean"),
    Score_min=("Silhouette", "min"),
    Clients_score_negatif=("Silhouette", lambda x: (x < 0).sum())
).round(3)

print(evaluation)

clients_ambigus = silhouette_df[
    silhouette_df["Silhouette"] < 0
].copy()

clients_ambigus = clients_ambigus.join(
    rfm[["Recency", "Frequency", "Monetary"]]
)

clients_ambigus = clients_ambigus.sort_values(
    "Silhouette"
)

print(clients_ambigus)

import pandas as pd
from sklearn.metrics import pairwise_distances

distances = kmeans.transform(X_scaled)

distance_df = pd.DataFrame(
    distances,
    index=rfm.index,
    columns=[
        f"Distance_cluster_{i}"
        for i in range(kmeans.n_clusters)
    ]
)

distance_df["Cluster_actuel"]=rfm["Cluster"]

ambiguous_ids = silhouette_df[
    silhouette_df["Silhouette"]<0
].index

ambiguous_clients = distance_df.loc[ambiguous_ids].copy()

distance_columns = [
    f"Distance_cluster_{i}"
    for i in range(kmeans.n_clusters)
]
ambiguous_clients["Cluster_plus_proche"] = (
    ambiguous_clients[distance_columns].values.argmin(axis=1)
)
ambiguous_clients = ambiguous_clients.join(
    rfm[["Recency", "Frequency", "Monetary"]]
)
print(ambiguous_clients.round(3))
business_profile = rfm.groupby("Cluster").agg(
    Nombre_clients=("Recency", "size"),
    Recency_mediane=("Recency", "median"),
    Frequency_mediane=("Frequency", "median"),
    Monetary_mediane=("Monetary", "median"),
    Monetary_total=("Monetary", "sum")
)

business_profile["Part_clients_pct"] = (
    business_profile["Nombre_clients"]
    / len(rfm)
    * 100
).round(2)

business_profile["Part_Monetary_pct"] = (
    business_profile["Monetary_total"]
    / rfm["Monetary"].sum()
    * 100
).round(2)

print(business_profile.round(2))

import matplotlib.pyplot as plt

part_monetary = business_profile["Part_Monetary_pct"].sort_values(
    ascending=False
)

part_monetary.plot(kind="bar", rot=0)

plt.title("Contribution de chaque cluster aux achats")
plt.xlabel("Cluster")
plt.ylabel("Part du montant total des achats (%)")
plt.tight_layout()
plt.show()

comparison = business_profile[
    ["Part_clients_pct", "Part_Monetary_pct"]
]

comparison.plot(kind="bar", figsize=(9, 5), rot=0)

plt.title("Part des clients vs part des achats")
plt.xlabel("Cluster")
plt.ylabel("Pourcentage (%)")
plt.legend(["Part des clients", "Part des achats"])
plt.tight_layout()
plt.show()

print(
    business_profile[
        [
            "Nombre_clients",
            "Recency_mediane",
            "Frequency_mediane",
            "Monetary_mediane"
        ]
    ].round(2)
)

cluster_names = {
    0: "Clients occasionnels",
    1: "Clients inactifs / à risque",
    2: "Clients à forte valeur"
}

rfm["Segment"] = rfm["Cluster"].map(cluster_names)

rfm[["Recency", "Frequency", "Monetary", "Cluster", "Segment"]].head()

import pandas as pd

business_profile = rfm.groupby("Segment").agg(
    Nombre_clients=("Recency", "size"),
    Recency_mediane=("Recency", "median"),
    Frequency_mediane=("Frequency", "median"),
    Monetary_mediane=("Monetary", "median"),
    Monetary_total=("Monetary", "sum")
)

business_profile["Part_clients_pct"] = (
    business_profile["Nombre_clients"] / len(rfm) * 100
).round(2)

business_profile["Part_Monetary_pct"] = (
    business_profile["Monetary_total"] / rfm["Monetary"].sum() * 100
).round(2)


import matplotlib.pyplot as plt

metrics = ["Recency", "Frequency", "Monetary"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, metric in zip(axes, metrics):
    rfm.boxplot(
        column=metric,
        by="Segment",
        ax=ax,
        grid=False
    )

    ax.set_title(metric)
    ax.set_xlabel("Segment")
    ax.tick_params(axis="x", rotation=25)

plt.suptitle("Distribution des indicateurs RFM par segment")
plt.tight_layout()
plt.show()

contribution = business_profile.sort_values(
    "Part_Monetary_pct",
    ascending=False
)

plt.figure(figsize=(9, 5))

plt.bar(
    contribution.index,
    contribution["Part_Monetary_pct"]
)

plt.title("Contribution au montant total des achats par segment")
plt.xlabel("Segment")
plt.ylabel("Part du montant total (%)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()
from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

rfm.to_csv(
    output_dir / "clients_segmentes.csv",
    index=True
)

business_profile.to_csv(
    output_dir / "profils_segments.csv"
)

print("Exports terminés.")
import joblib

joblib.dump(kmeans, output_dir / "kmeans_model.joblib")
joblib.dump(scaler, output_dir / "standard_scaler.joblib")

print("Modèle et scaler sauvegardés.")