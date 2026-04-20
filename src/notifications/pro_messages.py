"""
Messages Telegram — simples, clairs, actionnables.
Format : action + ticker + prix + potentiel + UNE raison claire.
"""
from datetime import datetime
from typing import List, Dict, Optional


class ProMessageBuilder:
    DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━"

    @staticmethod
    def startup() -> str:
        now = datetime.now().strftime("%d/%m/%Y — %H:%M")
        return (
            f"🟢 <b>AlphaForge — EN LIGNE</b>\n"
            f"{ProMessageBuilder.DIVIDER}\n"
            f"📅 {now}\n"
            f"🧠 Moteur : CPA + ML Ensemble (GB+RF+LR+IsoForest)\n"
            f"<i>Scan des opportunités en cours…</i>"
        )

    @staticmethod
    def market_open_banner() -> str:
        return (
            f"🔔 <b>OPPORTUNITÉS DU JOUR</b>\n"
            f"{ProMessageBuilder.DIVIDER}"
        )

    @staticmethod
    def opportunities(opps: List, universe: str, top_n: int = 10) -> str:
        """
        Format clair :

        #1 🟢🟢 ACHAT FORT | AAPL 185$ (+18%)
        → Sous-évaluée vs fondamentaux
        📊 IA: 72% hausse | Conf: 85%
        """
        top = [o for o in opps if o.universe == universe][:top_n]
        if not top:
            return f"ℹ️ Aucune opportunité {universe}"

        flag = {
            "SP500": "🇺🇸 <b>SP500</b>",
            "NASDAQ100": "💻 <b>NASDAQ 100</b>",
            "EUROSTOXX50": "🇪🇺 <b>EUROSTOXX 50</b>",
        }.get(universe, universe)

        lines = [f"\n{flag}\n"]

        for i, o in enumerate(top, 1):
            action_tag = ProMessageBuilder._action_tag(o.action)
            price_str = f"{o.price:.2f}$" if o.price else "?"

            upside_str = ""
            if o.upside_pct is not None and abs(o.upside_pct) > 1:
                sign = "+" if o.upside_pct > 0 else ""
                upside_str = f" <b>{sign}{o.upside_pct:.0f}%</b>"

            lines.append(
                f"<b>#{i}</b> {action_tag} · <b>{o.ticker}</b> · {price_str}{upside_str}"
            )

            # La raison claire
            lines.append(f"   → <i>{o.primary_reason}</i>")

            # Signaux ML si disponibles
            ml_parts = []
            if o.ml_proba_up is not None:
                ml_parts.append(f"🤖 IA hausse: <b>{o.ml_proba_up*100:.0f}%</b>")
            if o.ml_proba_strong and o.ml_proba_strong > 0.4:
                ml_parts.append(f"💥 Fort gain: {o.ml_proba_strong*100:.0f}%")
            if o.confidence:
                ml_parts.append(f"✅ Conf: {o.confidence*100:.0f}%")
            if ml_parts:
                lines.append(f"   {' · '.join(ml_parts)}")

            # Kelly position
            if o.kelly_position and o.kelly_position > 0.005:
                lines.append(f"   💼 Allocation suggérée: <b>{o.kelly_position*100:.1f}%</b>")

            # Stop/TP
            if o.stop_loss and o.take_profit and o.price:
                lines.append(
                    f"   🎯 TP: {o.take_profit:.2f}$ · 🛑 SL: {o.stop_loss:.2f}$"
                )

            # Risques
            if o.risk_flags:
                lines.append(f"   {o.risk_flags[0]}")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def market_summary(opps_all: List, total_analyzed: int) -> str:
        strong_buy = sum(1 for o in opps_all if o.action == "STRONG_BUY")
        buy = sum(1 for o in opps_all if o.action == "BUY")
        sell = sum(1 for o in opps_all if o.action in ("SELL", "STRONG_SELL"))

        return (
            f"📊 <b>RÉSUMÉ MARCHÉ</b>\n"
            f"{ProMessageBuilder.DIVIDER}\n"
            f"📈 {total_analyzed} actions analysées\n"
            f"🧠 {len(opps_all)} opportunités détectées\n\n"
            f"🚀 <b>{strong_buy}</b> ACHAT FORT\n"
            f"🟢 <b>{buy}</b> ACHAT\n"
            f"🔴 <b>{sell}</b> À ÉVITER"
        )

    @staticmethod
    def alert_flash(opp) -> str:
        action_tag = ProMessageBuilder._action_tag(opp.action)
        msg = [
            f"🚨 <b>ALERTE — {opp.ticker}</b>",
            f"{ProMessageBuilder.DIVIDER}",
            f"{action_tag}",
        ]
        if opp.price:
            msg.append(f"💵 Prix: <b>{opp.price:.2f}$</b>")
        if opp.upside_pct:
            sign = "+" if opp.upside_pct > 0 else ""
            msg.append(f"🎯 Potentiel: <b>{sign}{opp.upside_pct:.0f}%</b>")
        if opp.ml_proba_up:
            msg.append(f"🤖 IA hausse: <b>{opp.ml_proba_up*100:.0f}%</b>")
        msg.append(f"\n💡 <b>Pourquoi ?</b>\n{opp.primary_reason}")
        for r in opp.secondary_reasons[:2]:
            msg.append(f"• {r}")
        if opp.risk_flags:
            msg.append(f"\n{opp.risk_flags[0]}")
        return "\n".join(msg)

    @staticmethod
    def footer() -> str:
        return (
            f"{ProMessageBuilder.DIVIDER}\n"
            f"🤖 AlphaForge · {datetime.now().strftime('%H:%M')}\n"
            f"🧠 CPA + ML Ensemble · <i>Info uniquement, DYOR</i>"
        )

    @staticmethod
    def _action_tag(action: str) -> str:
        tags = {
            "STRONG_BUY": "🟢🟢 <b>ACHAT FORT</b>",
            "BUY": "🟢 <b>ACHAT</b>",
            "HOLD": "⚪ <b>NEUTRE</b>",
            "SELL": "🔴 <b>VENDRE</b>",
            "STRONG_SELL": "🔴🔴 <b>VENDRE FORT</b>",
        }
        return tags.get(action, "⚪ <b>NEUTRE</b>")
