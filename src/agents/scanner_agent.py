"""
Agent Scanner — CPA + ML Ensemble + Détection d'opportunités.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from src.data.universe import get_universe
from src.data.fetcher import fetch_prices, fetch_fundamentals, fetch_fama_french_factors
from src.models.cpa import CPACalculator, CPAResult
from src.models.opportunity_detector import OpportunityDetector, Opportunity
from config.settings import (
    DATA_PERIOD, TOP_N_SIGNALS, ALPHA_THRESHOLD,
    W1, W2, W3, W4, LAMBDA_RISK, RISK_FREE_RATE, KELLY_FRACTION,
)

logger = logging.getLogger(__name__)
MAX_WORKERS = 6
BATCH_SIZE = 20


class ScannerAgent:
    """Scanner avec pipeline ML intégré."""

    def __init__(self, universe: str = "SP500"):
        self.universe = universe
        self.calculator = CPACalculator(
            w1=W1, w2=W2, w3=W3, w4=W4,
            lambda_risk=LAMBDA_RISK,
            risk_free=RISK_FREE_RATE,
            kelly_fraction=KELLY_FRACTION,
        )
        self.detector = OpportunityDetector()
        self.ff_factors: Optional[pd.DataFrame] = None
        self.results: List[CPAResult] = []
        self.opportunities: List[Opportunity] = []

    def run(self, max_tickers: Optional[int] = None) -> List[CPAResult]:
        start = datetime.now()
        logger.info(f"[ScannerAgent] Démarrage {self.universe}")

        tickers = get_universe(self.universe)
        if max_tickers:
            tickers = tickers[:max_tickers]
        logger.info(f"[ScannerAgent] {len(tickers)} tickers")

        self.ff_factors = fetch_fama_french_factors()
        all_prices = self._fetch_prices_batched(tickers)
        benchmark = self._get_benchmark()

        logger.info("[ScannerAgent] Pipeline CPA + ML en parallèle...")
        self.results, self.opportunities = self._analyze_parallel(
            tickers, all_prices, benchmark
        )

        self.results.sort(key=lambda r: r.alpha, reverse=True)
        self.opportunities.sort(key=lambda o: o.score, reverse=True)

        elapsed = (datetime.now() - start).seconds
        logger.info(
            f"[ScannerAgent] {elapsed}s — "
            f"{len(self.results)} CPA / {len(self.opportunities)} opportunités"
        )
        return self.results

    def top_signals(self, n: int = TOP_N_SIGNALS, threshold: float = ALPHA_THRESHOLD):
        return [r for r in self.results if r.alpha >= threshold][:n]

    def top_opportunities(self, n: int = 10) -> List[Opportunity]:
        """Meilleures opportunités (acheter) — tri descendant par score."""
        return sorted(
            [o for o in self.opportunities if o.score > 0.10],
            key=lambda o: o.score, reverse=True,
        )[:n]

    def all_universe_opportunities(self) -> List[Opportunity]:
        """Toutes les opportunités positives + négatives triées."""
        return sorted(self.opportunities, key=lambda o: o.score, reverse=True)

    def _fetch_prices_batched(self, tickers: List[str]) -> Dict[str, pd.Series]:
        all_prices = {}
        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i:i + BATCH_SIZE]
            try:
                prices_df = fetch_prices(batch, period=DATA_PERIOD)
                for t in batch:
                    if t in prices_df.columns:
                        s = prices_df[t].dropna()
                        if len(s) > 30:
                            all_prices[t] = s
            except Exception as e:
                logger.warning(f"Batch error: {e}")
            time.sleep(0.2)
        logger.info(f"[ScannerAgent] Prix : {len(all_prices)}/{len(tickers)}")
        return all_prices

    def _analyze_parallel(
        self, tickers, all_prices, benchmark
    ):
        cpa_results = []
        opportunities = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(
                    self._analyze_one,
                    t,
                    all_prices.get(t, pd.Series(dtype=float)),
                    benchmark,
                ): t
                for t in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    cpa_res, opp = future.result(timeout=60)
                    if cpa_res:
                        cpa_results.append(cpa_res)
                    if opp:
                        opportunities.append(opp)
                except Exception as e:
                    logger.debug(f"{ticker}: {e}")
        return cpa_results, opportunities

    def _analyze_one(self, ticker, prices, benchmark):
        try:
            fundamentals = fetch_fundamentals(ticker)
            if "error" in fundamentals or not fundamentals.get("price"):
                return None, None

            # CPA
            cpa_result = self.calculator.compute(
                ticker=ticker,
                prices=prices,
                fundamentals=fundamentals,
                ff_factors=self.ff_factors,
                benchmark_prices=benchmark,
                universe=self.universe,
            )
            cpa_result.sector = fundamentals.get("sector", "")

            # ML + Opportunity (seulement si CPA confiant)
            opp = None
            if cpa_result.confidence >= 0.3 and len(prices) >= 200:
                try:
                    opp = self.detector.detect(cpa_result, prices, fundamentals)
                except Exception as e:
                    logger.debug(f"Opportunity detection {ticker}: {e}")

            return cpa_result, opp
        except Exception as e:
            logger.debug(f"{ticker} fail: {e}")
            return None, None

    def _get_benchmark(self):
        bm = {"SP500": "SPY", "NASDAQ100": "QQQ", "EUROSTOXX50": "FEZ"}
        ticker = bm.get(self.universe, "SPY")
        try:
            prices = fetch_prices([ticker], period=DATA_PERIOD)
            if ticker in prices.columns:
                return prices[ticker].dropna()
        except Exception:
            pass
        return None
