"""
Stacking ensemble ML model for market direction prediction.

Architecture:
  Level-0: XGBoost + RandomForest + Calibrated Logistic Regression
  Level-1: Logistic Regression meta-learner on out-of-fold predictions

Features: 25 engineered features covering price, volume, time, category,
          sentiment, technical indicators, and order book proxies.
"""

import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.pipeline import Pipeline

from kalshi_bot.config import MODEL_PATH, SYNTHETIC_TRAIN_SAMPLES
from kalshi_bot.signals.technical import extract_features as _tech_features
from kalshi_bot.signals.orderbook import extract_features as _ob_features

logger = logging.getLogger(__name__)

NONE_SIGNAL = {"signal": "NONE", "edge": 0.0, "confidence": 0.0}

CATEGORIES = [
    "politics", "economics", "sports", "crypto", "weather",
    "entertainment", "science", "finance", "unknown",
]
_cat_encoder = LabelEncoder().fit(CATEGORIES)


def _encode_category(cat: str) -> int:
    cat = (cat or "unknown").lower()
    if cat not in CATEGORIES:
        cat = "unknown"
    return int(_cat_encoder.transform([cat])[0])


def _market_to_features(market: dict) -> np.ndarray:
    """Build 25-feature vector from a market dict."""
    yp = market.get("yes_price", 50)
    np_ = market.get("no_price", 100 - yp)
    vol = market.get("volume", 0)
    yes_bid = market.get("yes_bid", yp - 1)
    yes_ask = market.get("yes_ask", yp + 1)
    spread = yes_ask - yes_bid
    htc = min(market.get("hours_to_close", 24), 720)
    cat = _encode_category(market.get("category", "unknown"))
    sentiment = market.get("sentiment_score", 0.0)
    momentum = market.get("price_momentum", 0.0)
    implied_prob = yp / 100.0

    # Technical features
    tech = _tech_features(market)
    rsi = tech.get("rsi", 50.0)
    bb_pos = tech.get("bb_position", 0.5)
    bb_width = tech.get("bb_width", 0.2)
    price_vs_ma20 = tech.get("price_vs_ma20", 0.0)
    price_vs_ma5 = tech.get("price_vs_ma5", 0.0)
    volatility = tech.get("volatility", 0.05)
    mom_fast = tech.get("mom_fast", 0.0)
    mean_rev_z = tech.get("mean_rev_zscore", 0.0)

    # Order book proxy features
    ob = _ob_features(market)
    imbalance = ob.get("imbalance", 0.0)
    micro_vs_mid = ob.get("microprice_vs_mid", 0.0)
    depth_ratio = ob.get("depth_ratio", 0.0)
    rel_spread = ob.get("relative_spread", 0.02)

    # Derived features
    log_vol = np.log1p(vol)
    log_htc = np.log1p(htc)
    logit_prob = np.log(implied_prob / (1 - implied_prob + 1e-9) + 1e-9)
    spread_vol = spread * log_vol / 20.0  # interaction term

    return np.array([
        # --- Price / probability ---
        yp,                    # 0
        implied_prob,          # 1
        logit_prob,            # 2
        # --- Market structure ---
        log_vol,               # 3
        spread,                # 4
        rel_spread,            # 5
        log_htc,               # 6
        float(cat),            # 7
        # --- Sentiment ---
        sentiment,             # 8
        # --- Momentum ---
        momentum,              # 9
        mom_fast,              # 10
        # --- Technical ---
        rsi,                   # 11
        bb_pos,                # 12
        bb_width,              # 13
        price_vs_ma20,         # 14
        price_vs_ma5,          # 15
        volatility,            # 16
        mean_rev_z,            # 17
        # --- Order book proxies ---
        imbalance,             # 18
        micro_vs_mid,          # 19
        depth_ratio,           # 20
        # --- Interactions ---
        spread_vol,            # 21
        sentiment * implied_prob,  # 22
        rsi * bb_pos,              # 23
        abs(mean_rev_z) * volatility,  # 24
    ], dtype=np.float32)


FEATURE_NAMES = [
    "yes_price", "implied_prob", "logit_prob",
    "log_volume", "spread", "relative_spread", "log_hours_to_close", "category",
    "sentiment_score",
    "price_momentum", "mom_fast",
    "rsi", "bb_position", "bb_width", "price_vs_ma20", "price_vs_ma5",
    "volatility", "mean_reversion_z",
    "ob_imbalance", "microprice_vs_mid", "depth_ratio",
    "spread_vol_interaction", "sentiment_x_prob", "rsi_x_bb", "rev_x_vol",
]


# ── Synthetic training data ───────────────────────────────────────────────────

def generate_training_data(n: int = SYNTHETIC_TRAIN_SAMPLES) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    yes_price = rng.integers(10, 91, size=n).astype(float)
    implied_prob = yes_price / 100.0
    logit_prob = np.log(implied_prob / (1 - implied_prob + 1e-9) + 1e-9)
    volume = rng.integers(500, 100_000, size=n).astype(float)
    spread = rng.integers(1, 15, size=n).astype(float)
    hours_to_close = rng.uniform(1, 720, size=n)
    cat_idx = rng.integers(0, len(CATEGORIES), size=n).astype(float)
    sentiment = rng.uniform(-1, 1, size=n)
    momentum = rng.uniform(-20, 20, size=n)
    mom_fast = rng.uniform(-10, 10, size=n)
    rsi = rng.uniform(10, 90, size=n)
    bb_pos = rng.uniform(0, 1, size=n)
    bb_width = rng.uniform(0.05, 0.50, size=n)
    vs_ma20 = rng.uniform(-15, 15, size=n)
    vs_ma5 = rng.uniform(-8, 8, size=n)
    volatility = rng.uniform(0.01, 0.20, size=n)
    mean_rev_z = rng.uniform(-3, 3, size=n)
    imbalance = rng.uniform(-0.8, 0.8, size=n)
    micro_vs_mid = rng.uniform(-2, 2, size=n)
    depth_ratio = rng.uniform(-0.5, 0.5, size=n)
    rel_spread = spread / (yes_price + 1e-6)
    spread_vol = spread * np.log1p(volume) / 20.0

    # Ground-truth probability with multiple factors contributing
    true_prob = np.clip(
        implied_prob
        + 0.10 * sentiment
        + 0.005 * momentum
        + 0.003 * mom_fast
        - 0.002 * (rsi - 50)       # oversold → bullish
        - 0.10 * (bb_pos - 0.5)    # upper band → bearish
        - 0.05 * mean_rev_z        # high z → mean revert (bearish)
        + 0.03 * imbalance          # buy pressure → bullish
        + 0.01 * micro_vs_mid,
        0.05, 0.95,
    )
    labels = rng.binomial(1, true_prob).astype(int)

    return pd.DataFrame({
        "yes_price": yes_price, "implied_prob": implied_prob, "logit_prob": logit_prob,
        "log_volume": np.log1p(volume), "spread": spread, "relative_spread": rel_spread,
        "log_hours_to_close": np.log1p(hours_to_close), "category": cat_idx,
        "sentiment_score": sentiment,
        "price_momentum": momentum, "mom_fast": mom_fast,
        "rsi": rsi, "bb_position": bb_pos, "bb_width": bb_width,
        "price_vs_ma20": vs_ma20, "price_vs_ma5": vs_ma5,
        "volatility": volatility, "mean_reversion_z": mean_rev_z,
        "ob_imbalance": imbalance, "microprice_vs_mid": micro_vs_mid,
        "depth_ratio": depth_ratio,
        "spread_vol_interaction": spread_vol,
        "sentiment_x_prob": sentiment * implied_prob,
        "rsi_x_bb": rsi * bb_pos,
        "rev_x_vol": np.abs(mean_rev_z) * volatility,
        "label": labels,
    })


# ── Stacking ensemble ────────────────────────────────────────────────────────

class StackingEnsemble:
    """
    Level-0: XGBoost, RandomForest, GradientBoosting (calibrated)
    Level-1: LogisticRegressionCV meta-learner on OOF predictions
    """

    def __init__(self):
        self.base_models: list = []
        self.meta_model = None
        self.scaler = StandardScaler()
        self._trained = False
        self._n_folds = 5

    def _build_base_models(self):
        xgb = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.75,
            min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            max_features="sqrt", random_state=42, n_jobs=-1,
        )
        gb = GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.06,
            subsample=0.8, min_samples_leaf=5, random_state=42,
        )
        # Calibrate each base model with Platt scaling
        return [
            CalibratedClassifierCV(xgb, method="sigmoid", cv=3),
            CalibratedClassifierCV(rf, method="sigmoid", cv=3),
            CalibratedClassifierCV(gb, method="isotonic", cv=3),
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> dict:
        self.base_models = self._build_base_models()
        kf = StratifiedKFold(n_splits=self._n_folds, shuffle=True, random_state=42)

        oof_preds = np.zeros((len(X), len(self.base_models)))

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            for m_idx, model in enumerate(self.base_models):
                import copy
                m_clone = copy.deepcopy(model)
                m_clone.fit(X_tr, y_tr)
                oof_preds[val_idx, m_idx] = m_clone.predict_proba(X_val)[:, 1]

        # Train base models on full dataset
        for model in self.base_models:
            model.fit(X, y)

        # Train meta-learner on OOF predictions
        self.meta_model = LogisticRegressionCV(
            Cs=10, cv=5, max_iter=500, random_state=42
        )
        self.meta_model.fit(oof_preds, y)
        self._trained = True

        # Evaluate on OOF
        meta_preds = self.meta_model.predict_proba(oof_preds)[:, 1]
        auc = roc_auc_score(y, meta_preds)
        acc = accuracy_score(y, (meta_preds > 0.5).astype(int))
        ll = log_loss(y, meta_preds)
        return {"auc": auc, "accuracy": acc, "log_loss": ll}

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns P(YES) for each sample."""
        base_preds = np.column_stack([
            m.predict_proba(X)[:, 1] for m in self.base_models
        ])
        return self.meta_model.predict_proba(base_preds)[:, 1]


# ── MLModel wrapper ──────────────────────────────────────────────────────────

class MLModel:
    def __init__(self):
        self._ensemble: StackingEnsemble | None = None
        self._trained = False
        self._feature_importance: dict | None = None
        self._load()

    def _load(self):
        path = Path(MODEL_PATH)
        if path.exists():
            try:
                with open(path, "rb") as f:
                    saved = pickle.load(f)
                self._ensemble = saved["ensemble"]
                self._feature_importance = saved.get("feature_importance")
                self._trained = True
                logger.info(f"Stacking ensemble loaded from {MODEL_PATH}")
            except Exception as exc:
                logger.warning(f"Could not load model: {exc}")

    def train(self, df: pd.DataFrame | None = None) -> dict:
        if df is None:
            df = generate_training_data()

        feature_cols = [c for c in df.columns if c != "label"]
        X = df[feature_cols].values.astype(np.float32)
        y = df["label"].values.astype(int)

        logger.info(f"Training stacking ensemble on {len(X)} samples, {X.shape[1]} features...")
        self._ensemble = StackingEnsemble()
        metrics = self._ensemble.fit(X, y)

        # Extract XGBoost feature importances
        try:
            xgb_model = self._ensemble.base_models[0].calibrated_classifiers_[0].estimator
            importances = xgb_model.feature_importances_
            self._feature_importance = dict(zip(feature_cols, importances.tolist()))
        except Exception:
            self._feature_importance = None

        self._trained = True
        self.save_model()

        logger.info(
            f"Ensemble trained — AUC={metrics['auc']:.4f} "
            f"Acc={metrics['accuracy']:.4f} LogLoss={metrics['log_loss']:.4f}"
        )
        return metrics

    def save_model(self):
        path = Path(MODEL_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "ensemble": self._ensemble,
                "feature_importance": self._feature_importance,
            }, f)
        logger.info(f"Ensemble saved to {MODEL_PATH}")

    def predict(self, market: dict) -> dict:
        if not self._trained or self._ensemble is None:
            return NONE_SIGNAL

        features = _market_to_features(market).reshape(1, -1)
        yes_prob = float(self._ensemble.predict_proba(features)[0])
        no_prob = 1.0 - yes_prob

        implied_yes = market.get("yes_price", 50) / 100.0
        yes_edge = yes_prob - implied_yes
        no_edge = no_prob - (1.0 - implied_yes)

        if abs(yes_edge) < 0.025 and abs(no_edge) < 0.025:
            return NONE_SIGNAL

        if yes_edge >= no_edge:
            signal, edge, confidence = "YES", yes_edge, yes_prob
        else:
            signal, edge, confidence = "NO", no_edge, no_prob

        return {
            "signal": signal,
            "edge": round(max(edge, 0.0), 4),
            "confidence": round(confidence, 4),
        }

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        if not self._feature_importance:
            return []
        sorted_fi = sorted(
            self._feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_fi[:n]


# ── Module-level singleton ────────────────────────────────────────────────────

_model_instance: MLModel | None = None


def get_model() -> MLModel:
    global _model_instance
    if _model_instance is None:
        _model_instance = MLModel()
        if not _model_instance._trained:
            logger.info("No saved model found — training stacking ensemble on synthetic data...")
            _model_instance.train()
    return _model_instance


def analyze(market: dict) -> dict:
    return get_model().predict(market)
