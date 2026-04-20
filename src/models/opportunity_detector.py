"""
Détecteur d'opportunités ultra-avancé.

Combine :
1. CPA (Composite Predictive Alpha) — signal quantitatif fondamental + technique
2. ML Ensemble — probabilité ML sur 21 jours (GB + RF + LR + IsoForest)
3. Détection d'anomalies — ruptures de régime statistiques
4. Filtres de qualité — liquidité, momentum, volatilité

Score Opportunity = 0.5·CPA + 0.35·ML + 0.15·Régime
Seuil opportunité : Score > 0.15 ET confiance > 50%
"""
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List

from .cpa import CPAResult
from .ml_ensemble import MLEnsembleDetector, MLSignal

logger = logging.getLogger(__name__)


@dataclass
class Opportunity:
    """Opportunité détectée avec tous les signaux agrégés."""
    ticker: str
    score: float                       # Score composé (-1 à 1)
    action: str                         # STRONG_BUY / BUY / HOLD / SELL
    confidence: float                   # 0-1
    price: Optional[float] = None
    target_price: Optional[float] = None
    upside_pct: Optional[float] = None

    # Signaux sources
    cpa_alpha: Optional[float] = None
    ml_proba_up: Optional[float] = None
    ml_proba_strong: Optional[float] = None

    # Explication humaine
    primary_reason: str = ""
    secondary_reasons: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)

    # Position sizing
    kelly_position: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    sector: str = ""
    universe: str = ""


class OpportunityDetector:
    """
    Orchestre CPA + ML pour générer les opportunités les plus fiables.
    """

    def __init__(
        self,
        w_cpa: float = 0.50,
        w_ml: float = 0.35,
        w_regime: float = 0.15,
        min_score: float = 0.15,
        min_confidence: float = 0.50,
    ):
        self.w_cpa = w_cpa
        self.w_ml = w_ml
        self.w_regime = w_regime
        self.min_score = min_score
        self.min_confidence = min_confidence
        self.ml = MLEnsembleDetector(horizon=21)

    def detect(
        self,
        cpa_result: CPAResult,
        prices: pd.Series,
        fundamentals: Dict,
    ) -> Optional[Opportunity]:
        """Analyse un titre et retourne une opportunité ou None."""
        if cpa_result.confidence < 0.3:
            return None

        # 1. Signal CPA
        cpa_score = np.tanh(cpa_result.alpha * 3)  # normaliser [-1,1]

        # 2. Signal ML
        ml_signal = self.ml.fit_predict(prices, fundamentals)
        ml_score = ml_signal.ensemble_score if ml_signal else 0.0
        ml_conf = ml_signal.confidence if ml_signal else 0.5

        # 3. Régime (qualité statistique)
        regime_score = self._regime_score(prices, cpa_result)

        # Score composite
        final_score = (
            self.w_cpa * cpa_score
            + self.w_ml * ml_score
            + self.w_regime * regime_score
        )

        # Confiance globale
        confidence = (
            0.4 * cpa_result.confidence
            + 0.4 * ml_conf
            + 0.2 * min(1.0, regime_score + 0.5)
        )

        # Seuil d'acceptation
        if abs(final_score) < self.min_score or confidence < self.min_confidence:
            return None

        # Construire l'opportunité
        opp = Opportunity(
            ticker=cpa_result.ticker,
            score=float(final_score),
            action=self._decide_action(final_score),
            confidence=float(confidence),
            price=cpa_result.price,
            target_price=cpa_result.intrinsic_value,
            upside_pct=cpa_result.upside_pct,
            cpa_alpha=cpa_result.alpha,
            ml_proba_up=ml_signal.proba_up if ml_signal else None,
            ml_proba_strong=ml_signal.proba_strong_up if ml_signal else None,
            kelly_position=cpa_result.kelly_position,
            sector=cpa_result.sector,
            universe=cpa_result.universe,
        )

        # Stop / Take Profit
        if cpa_result.price:
            vol = self._realized_vol(prices)
            opp.stop_loss = cpa_result.price * (1 - 2 * vol)
            opp.take_profit = cpa_result.price * (1 + 3 * vol)

        # Raisons claires
        opp.primary_reason, opp.secondary_reasons = self._build_reasons(
            cpa_result, ml_signal, regime_score
        )
        opp.risk_flags = self._risk_flags(prices, fundamentals, cpa_result)

        return opp

    def _decide_action(self, score: float) -> str:
        if score > 0.35:
            return "STRONG_BUY"
        elif score > 0.15:
            return "BUY"
        elif score > -0.15:
            return "HOLD"
        elif score > -0.35:
            return "SELL"
        else:
            return "STRONG_SELL"

    def _regime_score(
        self, prices: pd.Series, cpa: CPAResult
    ) -> float:
        """Score de qualité du régime de marché pour ce titre."""
        if len(prices) < 63:
            return 0.0

        returns = np.log(prices / prices.shift(1)).dropna()
        score = 0.0

        # Tendance positive sur 3 mois
        ret_3m = float(returns.tail(63).sum())
        score += np.clip(ret_3m * 2, -0.5, 0.5)

        # Volatilité raisonnable (pas de panique)
        vol = float(returns.tail(21).std() * np.sqrt(252))
        if vol < 0.30:
            score += 0.2
        elif vol > 0.60:
            score -= 0.3

        # Pas de crash récent
        dd = (prices.tail(63) / prices.tail(63).cummax() - 1).min()
        if dd > -0.10:
            score += 0.15

        return float(np.clip(score, -1, 1))

    def _realized_vol(self, prices: pd.Series, window: int = 21) -> float:
        """Volatilité réalisée journalière."""
        returns = np.log(prices / prices.shift(1)).dropna()
        if len(returns) < window:
            return 0.02
        return float(returns.tail(window).std())

    def _build_reasons(
        self, cpa: CPAResult, ml: Optional[MLSignal], regime: float
    ) -> tuple:
        """Construit l'explication claire et actionnable."""
        primary = ""
        secondary = []

        # Raison principale : le signal le plus fort
        components = {
            "value": (cpa.value_gap or 0, "sous-évaluée vs fondamentaux"),
            "factor": (cpa.factor_premia or 0, "facteurs de qualité favorables"),
            "mean_rev": (cpa.mean_reversion or 0, "correction excessive (rebond attendu)"),
            "info": (cpa.info_flow or 0, "momentum positif solide"),
        }
        dominant = max(components, key=lambda k: abs(components[k][0]))
        val, label = components[dominant]
        if val > 0:
            primary = f"Action {label}"
        else:
            primary = f"Risque : {label.replace('favorables', 'défavorables')}"

        # Raisons secondaires
        if ml and ml.proba_up > 0.65:
            secondary.append(
                f"IA donne {ml.proba_up*100:.0f}% de chance de hausse sur 21 jours"
            )
        if ml and ml.proba_strong_up > 0.50:
            secondary.append(
                f"Forte probabilité de +5% minimum ({ml.proba_strong_up*100:.0f}%)"
            )
        if cpa.upside_pct and cpa.upside_pct > 15:
            secondary.append(f"Potentiel théorique de {cpa.upside_pct:+.0f}%")
        if regime > 0.3:
            secondary.append("Régime de marché favorable")

        return primary, secondary

    def _risk_flags(
        self, prices: pd.Series, fundamentals: Dict, cpa: CPAResult
    ) -> List[str]:
        """Flags de risque à signaler."""
        flags = []
        returns = np.log(prices / prices.shift(1)).dropna()

        # Volatilité extrême
        if len(returns) >= 21:
            vol = returns.tail(21).std() * np.sqrt(252)
            if vol > 0.6:
                flags.append(f"⚠️ Haute volatilité ({vol*100:.0f}% ann.)")

        # Drawdown important
        if len(prices) >= 252:
            dd = float((prices.tail(252) / prices.tail(252).cummax() - 1).min())
            if dd < -0.30:
                flags.append(f"⚠️ Drawdown récent {dd*100:.0f}%")

        # Levier élevé
        debt_eq = fundamentals.get("debt_to_equity") or 0
        if debt_eq and debt_eq > 200:
            flags.append(f"⚠️ Dette élevée (D/E {debt_eq:.0f}%)")

        # Marge faible
        op_margin = fundamentals.get("operating_margin") or 0.1
        if op_margin < 0:
            flags.append("⚠️ Rentabilité opérationnelle négative")

        return flags
