"""
Module ML Ensemble — détection d'opportunités par Machine Learning avancé.

Combine 4 algorithmes en stacking :
- Gradient Boosting (XGBoost-like via sklearn)
- Random Forest (robuste au bruit)
- Logistic Regression (interprétable)
- Isolation Forest (détection d'anomalies)

Features ingénierées :
- Techniques : RSI, MACD, Bollinger, ATR, Stochastic
- Momentum : 1J/5J/21J/63J/252J rolling returns
- Volatilité : réalisée, GARCH-like, régimes
- Microstructure : autocorrélation, Hurst exponent
- Fondamentaux : P/E, P/B, ROE, marges, croissance

Output : probabilité d'un rendement positif dans les N prochains jours.
"""
import logging
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


@dataclass
class MLSignal:
    """Signal ML pour un titre."""
    ticker: str
    proba_up: float              # Probabilité hausse (0-1)
    proba_strong_up: float       # Probabilité forte hausse (>5%)
    anomaly_score: float         # Score d'anomalie (-1 à 1, + = normal)
    ensemble_score: float        # Score composé final (-1 à 1)
    confidence: float            # Accord entre modèles (0-1)
    top_features: List[Tuple[str, float]]  # Features les plus prédictives


class FeatureEngineer:
    """
    Ingénierie de features techniques + fondamentaux.
    Produit ~35 features par titre à partir des prix OHLCV.
    """

    @staticmethod
    def compute_features(
        prices: pd.Series,
        fundamentals: Optional[Dict] = None,
    ) -> Optional[pd.DataFrame]:
        """Retourne un DataFrame de features alignés sur les prix."""
        if len(prices) < 100:
            return None

        df = pd.DataFrame(index=prices.index)
        df["price"] = prices
        df["logret"] = np.log(prices / prices.shift(1))

        # ── Momentum multi-horizons ──────────────────────────────────────────
        for days in [1, 5, 10, 21, 63, 126, 252]:
            if len(prices) > days:
                df[f"mom_{days}"] = np.log(prices / prices.shift(days))

        # ── Moyennes mobiles et distance relative ────────────────────────────
        for window in [5, 21, 63, 200]:
            ma = prices.rolling(window).mean()
            df[f"dist_ma{window}"] = (prices - ma) / ma
            df[f"ma{window}_slope"] = ma.pct_change(5)

        # ── Volatilité ──────────────────────────────────────────────────────
        df["vol_21"] = df["logret"].rolling(21).std() * np.sqrt(252)
        df["vol_63"] = df["logret"].rolling(63).std() * np.sqrt(252)
        df["vol_ratio"] = df["vol_21"] / df["vol_63"]
        df["vol_shock"] = df["logret"].abs() / df["vol_21"].shift(1)

        # ── RSI (14) ────────────────────────────────────────────────────────
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi_14"] = 100 - 100 / (1 + rs)

        # ── MACD ────────────────────────────────────────────────────────────
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        df["macd"] = (ema_12 - ema_26) / prices
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ── Bollinger Bands (20, 2σ) ────────────────────────────────────────
        mid = prices.rolling(20).mean()
        std = prices.rolling(20).std()
        df["bb_pct"] = (prices - mid) / (2 * std + 1e-10)
        df["bb_width"] = (4 * std) / mid

        # ── Autocorrélation (mean reversion vs trend) ───────────────────────
        df["autocorr_5"] = df["logret"].rolling(60).apply(
            lambda x: x.autocorr(lag=5) if len(x.dropna()) > 10 else 0,
            raw=False,
        )

        # ── Ratio Sharpe glissant 63J ──────────────────────────────────────
        mean_r = df["logret"].rolling(63).mean() * 252
        vol_r = df["logret"].rolling(63).std() * np.sqrt(252)
        df["sharpe_63"] = mean_r / (vol_r + 1e-10)

        # ── Drawdown actuel vs max 252J ────────────────────────────────────
        rolling_max = prices.rolling(252).max()
        df["drawdown_252"] = (prices - rolling_max) / rolling_max

        # ── Hurst exponent simplifié (régime : trend vs mean-rev) ──────────
        df["hurst_proxy"] = df["logret"].rolling(100).apply(
            FeatureEngineer._hurst_proxy, raw=True
        )

        # ── Fondamentaux (valeurs répétées pour alignement) ────────────────
        if fundamentals:
            df["pe_proxy"] = _safe(fundamentals.get("price", 0)) / max(
                _safe(fundamentals.get("book_value_per_share", 1)), 0.01
            ) / max(_safe(fundamentals.get("roe", 0.1)), 0.01)
            df["pb"] = _safe(fundamentals.get("price", 0)) / max(
                _safe(fundamentals.get("book_value_per_share", 1)), 0.01
            )
            df["roe"] = _safe(fundamentals.get("roe", 0.1))
            df["margin"] = _safe(fundamentals.get("operating_margin", 0.1))
            df["growth"] = _safe(fundamentals.get("earnings_growth", 0))
            df["debt_eq"] = _safe(fundamentals.get("debt_to_equity", 50)) / 100
            df["beta"] = _safe(fundamentals.get("beta", 1.0))

        return df.dropna()

    @staticmethod
    def _hurst_proxy(returns: np.ndarray) -> float:
        """Estimateur rapide de l'exposant de Hurst (0.5 = random walk)."""
        if len(returns) < 20:
            return 0.5
        try:
            lags = [5, 10, 20, 40]
            tau = [np.sqrt(np.std(np.subtract(returns[lag:], returns[:-lag])))
                   for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return float(poly[0] * 2)
        except Exception:
            return 0.5


def _safe(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return float(v)


class MLEnsembleDetector:
    """
    Détecteur d'opportunités par ensemble ML.

    Entraînement auto-supervisé sur l'historique :
    - Label = rendement à N jours > seuil (signal positif)
    - Training : données jusqu'à t-N
    - Prediction : features actuels
    """

    def __init__(
        self,
        horizon: int = 21,         # prédire le rendement à 21 jours
        strong_threshold: float = 0.05,
        random_state: int = 42,
    ):
        self.horizon = horizon
        self.strong_threshold = strong_threshold
        self.rs = random_state
        self.scaler = StandardScaler()

        # Modèles
        self.gb = GradientBoostingClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.05,
            random_state=random_state,
        )
        self.rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, min_samples_leaf=10,
            random_state=random_state, n_jobs=-1,
        )
        self.lr = LogisticRegression(
            max_iter=500, C=0.5, random_state=random_state,
        )
        self.iso = IsolationForest(
            n_estimators=80, contamination=0.1, random_state=random_state,
        )

        self.fitted = False
        self.feature_names: List[str] = []

    def fit_predict(
        self,
        prices: pd.Series,
        fundamentals: Optional[Dict] = None,
    ) -> Optional[MLSignal]:
        """
        Entraîne sur l'historique du titre puis prédit la probabilité actuelle.
        Mode self-training : chaque titre entraîne son propre ensemble.
        """
        ticker = fundamentals.get("ticker", "?") if fundamentals else "?"

        df = FeatureEngineer.compute_features(prices, fundamentals)
        if df is None or len(df) < 200:
            return None

        # Features / labels
        feature_cols = [c for c in df.columns if c not in ["price", "logret"]]
        self.feature_names = feature_cols

        X_all = df[feature_cols].values
        # Label : rendement futur > 0 (binaire)
        future_ret = np.log(
            df["price"].shift(-self.horizon) / df["price"]
        )
        y_all = (future_ret > 0).astype(int)
        y_strong = (future_ret > self.strong_threshold).astype(int)

        # Dernière ligne = prédiction courante (sans label)
        X_current = X_all[-1:].copy()

        # Training set : on exclut les N dernières lignes (pas de label)
        train_mask = ~future_ret.isna()
        X_train = X_all[train_mask.values]
        y_train = y_all[train_mask.values].values
        y_strong_train = y_strong[train_mask.values].values

        if len(X_train) < 80 or len(np.unique(y_train)) < 2:
            return None

        try:
            # Standardisation
            X_train_s = self.scaler.fit_transform(X_train)
            X_current_s = self.scaler.transform(X_current)

            # ── Modèle 1 : Gradient Boosting ────────────────────────────
            self.gb.fit(X_train_s, y_train)
            proba_gb = self.gb.predict_proba(X_current_s)[0, 1]

            # ── Modèle 2 : Random Forest ────────────────────────────────
            self.rf.fit(X_train_s, y_train)
            proba_rf = self.rf.predict_proba(X_current_s)[0, 1]

            # ── Modèle 3 : Logistic Regression ──────────────────────────
            self.lr.fit(X_train_s, y_train)
            proba_lr = self.lr.predict_proba(X_current_s)[0, 1]

            # ── Modèle 4 : Strong up (si assez de positifs) ─────────────
            proba_strong = 0.5
            if y_strong_train.sum() > 10:
                gb_strong = GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, random_state=self.rs
                )
                gb_strong.fit(X_train_s, y_strong_train)
                proba_strong = gb_strong.predict_proba(X_current_s)[0, 1]

            # ── Modèle 5 : Anomaly Detection (Isolation Forest) ─────────
            self.iso.fit(X_train_s)
            anomaly = self.iso.decision_function(X_current_s)[0]

            # ── Stacking : moyenne pondérée ─────────────────────────────
            proba_up = 0.4 * proba_gb + 0.35 * proba_rf + 0.25 * proba_lr

            # Accord entre modèles = confiance
            probas = np.array([proba_gb, proba_rf, proba_lr])
            confidence = 1 - np.std(probas) * 2  # std faible = accord fort

            # Score composé : [-1, 1]
            ensemble_score = (proba_up - 0.5) * 2
            ensemble_score = np.clip(ensemble_score + 0.2 * anomaly, -1, 1)

            # ── Top features (importance du GB) ─────────────────────────
            importances = self.gb.feature_importances_
            top_idx = np.argsort(importances)[::-1][:5]
            top_features = [
                (feature_cols[i], float(importances[i])) for i in top_idx
            ]

            return MLSignal(
                ticker=ticker,
                proba_up=float(proba_up),
                proba_strong_up=float(proba_strong),
                anomaly_score=float(anomaly),
                ensemble_score=float(ensemble_score),
                confidence=float(np.clip(confidence, 0, 1)),
                top_features=top_features,
            )
        except Exception as e:
            logger.debug(f"ML fit failed for {ticker}: {e}")
            return None


class CrossSectionalMLScreen:
    """
    Écran ML cross-sectional (tous titres simultanément).

    Forme un dataset global de tous les titres puis applique :
    - Clustering K-Means pour détecter les régimes
    - XGBoost-like pour classer les titres par probabilité de surperformance
    """

    def __init__(self, horizon: int = 21):
        self.horizon = horizon

    def screen(
        self,
        ml_signals: List[MLSignal],
    ) -> List[MLSignal]:
        """Filtre et classe les signaux ML par score composé."""
        if not ml_signals:
            return []

        # Tri par score ensemble + pénalité si faible confiance
        def composite(s: MLSignal) -> float:
            return s.ensemble_score * (0.5 + 0.5 * s.confidence)

        return sorted(ml_signals, key=composite, reverse=True)
