"""
Bot Loop — scanne CPA + ML et envoie les opportunités sur Telegram.
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

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
    def __init__(self, interval_seconds: int = 14400, test_mode: bool = False):
        self.interval = interval_seconds
        self.test_mode = test_mode
        self.notifier = TelegramNotifier()
        self.msg = ProMessageBuilder()
        self.iteration = 0
        self.start_time = datetime.now()

        if not self.notifier.token or not self.notifier.chat_id:
            logger.error("❌ TELEGRAM_BOT_TOKEN/CHAT_ID manquants")
            sys.exit(1)

    def run(self, once: bool = False):
        self._send(self.msg.startup())
        logger.info("🚀 AlphaForge Bot démarré")

        try:
            while True:
                self.iteration += 1
                logger.info(f"\n{'='*60}\nITÉRATION #{self.iteration}\n{'='*60}")
                try:
                    self._run_cycle()
                except Exception as e:
                    logger.error(f"Erreur: {e}", exc_info=True)
                    self.notifier.send_message(f"⚠️ Erreur: {str(e)[:200]}")

                if once:
                    break

                next_run = datetime.now() + timedelta(seconds=self.interval)
                logger.info(f"💤 Prochain scan: {next_run.strftime('%H:%M:%S')}")
                time.sleep(self.interval)

        except KeyboardInterrupt:
            self._send("⏸ Bot arrêté")

    def _run_cycle(self):
        self._send(self.msg.market_open_banner())

        max_tickers = 15 if self.test_mode else None
        all_opportunities = []
        total_analyzed = 0

        for universe in UNIVERSES:
            logger.info(f"🔍 Scan {universe}...")
            try:
                scanner = ScannerAgent(universe=universe)
                scanner.run(max_tickers=max_tickers)
                opps = scanner.all_universe_opportunities()
                all_opportunities.extend(opps)
                total_analyzed += len(scanner.results)
                logger.info(f"  ✅ {len(opps)} opportunités / {len(scanner.results)} analysés")

                if opps:
                    msg = self.msg.opportunities(opps, universe, top_n=TOP_N_SIGNALS)
                    self._send(msg)
                    time.sleep(1)
            except Exception as e:
                logger.error(f"  ❌ {e}")

        # Résumé
        self._send(self.msg.market_summary(all_opportunities, total_analyzed))

        # Flash sur les 3 meilleures opportunités
        strong = sorted(
            [o for o in all_opportunities if abs(o.score) > 0.35],
            key=lambda o: abs(o.score), reverse=True,
        )[:3]
        for o in strong:
            time.sleep(1)
            self._send(self.msg.alert_flash(o))

        self._send(self.msg.footer())

    def _send(self, text: str):
        ok = self.notifier.send_chunk(text)
        if ok:
            logger.info(f"📤 Envoyé ({len(text)} chars)")
        time.sleep(0.5)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=int, default=14400)
    return p.parse_args()


def main():
    args = parse_args()
    bot = AlphaForgeBot(
        interval_seconds=300 if args.demo else args.interval,
        test_mode=args.demo,
    )
    bot.run(once=args.once)


if __name__ == "__main__":
    main()
