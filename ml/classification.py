import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# Features structurelles uniquement — jamais de colonnes dérivées du score cible
CLASS_FEATURES = [
    "prix",
    "remise_pct",
    "stock",
    "nb_images",
    "nb_variants",
    "en_promo",
    "categorie_enc",
    "source_enc",
]


def create_target(df_ml: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la variable cible 'top_produit' par quantile sur le prix.
    Produits les mieux positionnés (prix modéré + dispo + promo) = top.
    Garanti non-vide et équilibré quelle que soit la qualité des données.
    """
    df_ml = df_ml.copy()

    # Score neutre : uniquement sur des features STRUCTURELLES
    # (jamais rating/nb_reviews s'ils sont tous à 0)
    score = pd.Series(np.zeros(len(df_ml)), index=df_ml.index)

    # Prix normalisé inversé : moins cher → meilleur score
    if df_ml["prix"].sum() > 0:
        px = df_ml["prix"].clip(lower=0)
        px_norm = (px - px.min()) / (px.max() - px.min() + 1e-9)
        score += (1 - px_norm) * 40

    # Remise active = bonus
    if "remise_pct" in df_ml.columns:
        rem = df_ml["remise_pct"].fillna(0).clip(lower=0)
        if rem.sum() > 0:
            score += (rem / (rem.max() + 1e-9)) * 25

    # Disponibilité
    if "disponible" in df_ml.columns:
        score += df_ml["disponible"].fillna(0).astype(float) * 20

    # Stock disponible
    if "stock" in df_ml.columns:
        st = df_ml["stock"].fillna(0).clip(lower=0)
        if st.sum() > 0:
            score += (st / (st.max() + 1e-9)) * 15

    # Si tout est encore plat → on ajoute du bruit pour forcer la séparation
    if score.std() < 1e-6:
        np.random.seed(42)
        score += np.random.rand(len(score)) * 10

    df_ml["_score_cible"] = score

    # Top 30% = top produit (toujours ~30% de 1 et ~70% de 0)
    seuil = df_ml["_score_cible"].quantile(0.70)
    df_ml["top_produit"] = (df_ml["_score_cible"] >= seuil).astype(int)
    df_ml.drop(columns=["_score_cible"], inplace=True)

    n1 = df_ml["top_produit"].sum()
    n0 = len(df_ml) - n1
    print(f"  [Cible] Top produits={n1} ({n1/len(df_ml)*100:.1f}%) "
          f"| Standard={n0} ({n0/len(df_ml)*100:.1f}%)")
    return df_ml


def train_and_evaluate(df_ml: pd.DataFrame) -> tuple:
    df_ml = create_target(df_ml)

    # Colonnes manquantes → 0
    for col in CLASS_FEATURES:
        if col not in df_ml.columns:
            df_ml[col] = 0
            print(f"  [Info] Colonne '{col}' absente → 0")

    X = df_ml[CLASS_FEATURES].fillna(0).astype(float)
    y = df_ml["top_produit"]

    if y.nunique() < 2:
        print("  [STOP] Une seule classe détectée — pipeline classification ignoré.")
        empty = {"accuracy": 0, "f1": 0, "cv_mean": 0, "cv_std": 0}
        return {"random_forest": empty, "xgboost": empty}, df_ml, None, None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  [Split] Train={len(X_train)} | Test={len(X_test)}")

    results = {}

    # ── Random Forest ─────────────────────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results["random_forest"] = _evaluate_model(
        "Random Forest", rf, X_train, X_test, y_train, y_test
    )

    # ── XGBoost ───────────────────────────────────────────────────
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=n0 / (n1 + 1e-9),
        random_state=42,
        eval_metric="logloss",
        verbosity=0
    )
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    results["xgboost"] = _evaluate_model(
        "XGBoost", xgb, X_train, X_test, y_train, y_test
    )

    _plot_feature_importance(xgb, CLASS_FEATURES)
    _plot_confusion_matrix(y_test, xgb.predict(X_test), "XGBoost")

    # Probabilités dans le DataFrame
    df_ml["proba_top_xgb"]   = xgb.predict_proba(X)[:, 1]
    df_ml["proba_top_rf"]    = rf.predict_proba(X)[:, 1]
    df_ml["proba_consensus"] = (df_ml["proba_top_xgb"] + df_ml["proba_top_rf"]) / 2

    return results, df_ml, xgb, rf


def _evaluate_model(name, model, X_train, X_test, y_train, y_test):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_sc  = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring="f1_weighted")

    labels       = sorted(y_test.unique())
    target_names = [("Standard" if l == 0 else "Top produit") for l in labels]

    print(f"\n{'='*45}")
    print(f"  Modèle : {name}")
    print(f"{'='*45}")
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  F1-Score       : {f1:.4f}")
    print(f"  CV F1 (5-fold) : {cv_sc.mean():.4f} ± {cv_sc.std():.4f}")
    print(classification_report(
        y_test, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0
    ))

    return {
        "accuracy": acc, "f1": f1,
        "cv_mean": cv_sc.mean(), "cv_std": cv_sc.std()
    }


def _plot_feature_importance(model, features,
                              path="data/processed/feature_importance.png"):
    imp     = model.feature_importances_
    idx     = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh([features[i] for i in idx], [imp[i] for i in idx],
            color="#534AB7", alpha=0.85)
    ax.set_title("Importance des features — XGBoost")
    ax.set_xlabel("Score")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[Classification] Feature importance : {path}")


def _plot_confusion_matrix(y_test, y_pred, model_name):
    labels = sorted(set(y_test) | set(y_pred))
    names  = [("Standard" if l == 0 else "Top produit") for l in labels]
    cm     = confusion_matrix(y_test, y_pred, labels=labels)
    disp   = ConfusionMatrixDisplay(cm, display_labels=names)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Matrice de confusion — {model_name}")
    plt.tight_layout()
    path = f"data/processed/confusion_matrix_{model_name.lower()}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[Classification] Matrice de confusion : {path}")