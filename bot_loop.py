"""
Bot Loop — Boucle autonome du scanner CPA.

Tourne en continu, exécute des scans périodiques, envoie des messages pro
sur Telegram à intervalles réguliers.

Usage :
    python bot_loop.py                # Boucle normale (scan toutes les 4h)
    python bot_loop.py --demo         # Mode démo (scan rapide toutes les 5 min)
    python bot_loop.py --once         # Un seul scan puis stop
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Charger .env.local si présent
env_file = Path(__file__).parent / ".env.local"
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import UNIVERSES, TOP_N_SIGNALS
from src.agents.scanner_agent import ScannerAgent
from src.notifications.telegram_bot import TelegramNotifier
from src.notifications.pro_messages import ProMessageBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("bot_loop")


class AlphaForgeBot:
    """Bot en boucle qui scan et envoie des signaux pro sur Telegram."""

    def __init__(self, interval_seconds: int = 14400, test_mode: bool = False):
        self.interval = interval_seconds
        self.test_mode = test_mode
        self.notifier = TelegramNotifier()
        self.msg = ProMessageBuilder()
        self.iteration = 0
        self.start_time = datetime.now()

        if not self.notifier.token or not self.notifier.chat_id:
            logger.error(
                "❌ TELEGRAM_BOT_TOKEN ou TELEGRAM_CHAT_ID manquant !\n"
                "   Lance : python tools/get_chat_id.py <TOKEN>"
            )
            sys.exit(1)

    def run(self, once: bool = False):
        """Boucle principale."""
        self._send(self.msg.startup())
        logger.info("🚀 AlphaForge Bot démarré")

        try:
            while True:
                self.iteration += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"ITÉRATION #{self.iteration}")
                logger.info(f"{'='*60}")

                try:
                    self._run_scan_cycle()
                except Exception as e:
                    logger.error(f"Erreur cycle: {e}", exc_info=True)
                    self.notifier.send_message(
                        f"⚠️ <b>Erreur pendant scan</b>\n<code>{str(e)[:200]}</code>"
                    )

                if once:
                    logger.info("Mode --once : arrêt")
                    break

                next_run = datetime.now() + timedelta(seconds=self.interval)
                logger.info(f"💤 Prochain scan : {next_run.strftime('%H:%M:%S')}")
                time.sleep(self.interval)

        except KeyboardInterrupt:
            logger.info("\n⛔ Arrêt manuel")
            self._send(
                f"⏸ <b>Bot arrêté</b>\n"
                f"Itérations : {self.iteration}\n"
                f"Durée : {datetime.now() - self.start_time}"
            )

    def _run_scan_cycle(self):
        """Un cycle complet : scan + envoi des messages pro."""
        # Bannière d'ouverture
        self._send(self.msg.market_open_banner())

        max_tickers = 15 if self.test_mode else None
        results_by_universe = {}

        for universe in UNIVERSES:
            logger.info(f"🔍 Scan {universe}...")
            try:
                scanner = ScannerAgent(universe=universe)
                results = scanner.run(max_tickers=max_tickers)
                results_by_universe[universe] = results
                logger.info(f"  ✅ {len(results)} résultats")
            except Exception as e:
                logger.error(f"  ❌ {e}")
                results_by_universe[universe] = []

        # Envoyer le résumé global
        summary = self.msg.market_summary(results_by_universe)
        self._send(summary)

        # Envoyer le top par univers (messages séparés pour lisibilité)
        for universe, results in results_by_universe.items():
            if results:
                top_msg = self.msg.top_signals(results, universe, top_n=TOP_N_SIGNALS)
                self._send(top_msg)
                time.sleep(1)  # évite le rate limit

        # Alertes flash pour signaux très forts
        all_results = [r for rs in results_by_universe.values() for r in rs]
        strong_signals = sorted(
            [r for r in all_results if abs(r.alpha) > 0.20],
            key=lambda r: abs(r.alpha),
            reverse=True,
        )[:3]

        for r in strong_signals:
            time.sleep(1)
            reason = self._dominant_reason(r)
            flash = self.msg.alert_flash(
                ticker=r.ticker,
                alpha=r.alpha,
                reason=reason,
                price=r.price,
                upside=r.upside_pct,
            )
            self._send(flash)

        # Footer
        self._send(self.msg.footer())

    def _send(self, text: str):
        """Envoie un message et log."""
        ok = self.notifier.send_chunk(text)
        if ok:
            logger.info(f"📤 Message envoyé ({len(text)} chars)")
        else:
            logger.error(f"❌ Échec envoi : {text[:100]}")
        time.sleep(0.5)  # rate limit Telegram

    @staticmethod
    def _dominant_reason(r) -> str:
        """Raison textuelle du signal."""
        components = {
            "Sous-évaluation fondamentale (RIM+Bayes)": r.value_gap or 0,
            "Primes de facteurs favorables (FF5+MOM)": r.factor_premia or 0,
            "Retour à la moyenne statistique (OU)": r.mean_reversion or 0,
            "Flux d'information positif (Kalman)": r.info_flow or 0,
        }
        return max(components, key=lambda k: abs(components[k]))


def parse_args():
    p = argparse.ArgumentParser(description="AlphaForge Bot Loop")
    p.add_argument("--demo", action="store_true",
                   help="Mode démo (15 tickers, scan toutes les 5 min)")
    p.add_argument("--once", action="store_true",
                   help="Un seul scan puis stop")
    p.add_argument("--interval", type=int, default=14400,
                   help="Intervalle en secondes (défaut: 4h = 14400)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.demo:
        bot = AlphaForgeBot(interval_seconds=300, test_mode=True)
    else:
        bot = AlphaForgeBot(interval_seconds=args.interval, test_mode=False)
    bot.run(once=args.once)


if __name__ == "__main__":
    main()
