"""
Messages Telegram ultra-pro — formatage HTML riche, emojis, mise en page.
"""
from datetime import datetime
from typing import List, Dict, Optional


class ProMessageBuilder:
    """Construit des messages Telegram de niveau institutionnel."""

    DIVIDER = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    SOFT_DIVIDER = "┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈"

    @staticmethod
    def startup() -> str:
        """Message de démarrage du bot."""
        now = datetime.now().strftime("%d/%m/%Y — %H:%M")
        return (
            f"🟢 <b>AlphaForge Bot — ACTIF</b>\n"
            f"{ProMessageBuilder.DIVIDER}\n\n"
            f"📅 {now}\n"
            f"🔬 Moteur : Composite Predictive Alpha (CPA)\n"
            f"📊 Univers : S&P 500 · Nasdaq 100 · Euro Stoxx 50\n"
            f"⚙️  Modèles : RIM+Bayes · FF5+MOM · OU-MLE · Kalman\n\n"
            f"<i>Scan en cours...</i>"
        )

    @staticmethod
    def market_open_banner() -> str:
        """Bannière ouverture marché."""
        return (
            f"🔔 <b>MARKET OPEN — PRE-MARKET SCAN</b>\n"
            f"{ProMessageBuilder.DIVIDER}\n"
            f"📈 Analyse des signaux alpha quotidiens\n"
            f"⚡ 500+ titres analysés en parallèle\n"
        )

    @staticmethod
    def top_signals(results: List, universe: str, top_n: int = 10) -> str:
        """Formate le top N signaux d'un univers avec design premium."""
        top = sorted(results, key=lambda r: r.alpha, reverse=True)[:top_n]
        if not top:
            return f"ℹ️  <i>Aucun signal {universe} détecté</i>"

        flag = {"SP500": "🇺🇸", "NASDAQ100": "🇺🇸💻", "EUROSTOXX50": "🇪🇺"}.get(universe, "🌍")

        lines = [
            f"\n{flag} <b>{universe} — TOP {len(top)} BUY SIGNALS</b>",
            f"{ProMessageBuilder.SOFT_DIVIDER}",
            "",
        ]

        for i, r in enumerate(top, 1):
            # Barre de force visuelle
            bar = ProMessageBuilder._alpha_bar(r.alpha)
            rank_emoji = ["🥇", "🥈", "🥉"][i-1] if i <= 3 else f"<code>#{i:02d}</code>"

            # Ligne principale
            lines.append(
                f"{rank_emoji} <b>{r.ticker}</b>  {bar}  "
                f"α<b>{r.alpha:+.3f}</b>"
            )

            # Détails
            details = []
            if r.upside_pct is not None:
                direction = "📈" if r.upside_pct > 0 else "📉"
                details.append(f"{direction} {r.upside_pct:+.1f}%")
            if r.price:
                details.append(f"💵 {r.price:.2f}")
            if r.confidence:
                stars = "⭐" * int(r.confidence * 4 + 0.5)
                details.append(f"{stars}")

            if details:
                lines.append(f"   {' · '.join(details)}")

            # Composants dominants
            components = {
                "Value": r.value_gap,
                "Factor": r.factor_premia,
                "MeanRev": r.mean_reversion,
                "InfoFlow": r.info_flow,
            }
            valid = {k: v for k, v in components.items() if v is not None}
            if valid:
                dominant = max(valid, key=lambda k: abs(valid[k]))
                lines.append(
                    f"   <i>Signal dominant : {dominant} ({valid[dominant]:+.3f})</i>"
                )

            if r.kelly_position and r.kelly_position > 0:
                lines.append(f"   💼 Kelly suggéré : <b>{r.kelly_position:.1%}</b>")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def market_summary(results_by_universe: Dict) -> str:
        """Résumé marché global."""
        total = sum(len(r) for r in results_by_universe.values())
        strong_buy = sum(
            1 for rs in results_by_universe.values() for r in rs if r.alpha > 0.15
        )
        buy = sum(
            1 for rs in results_by_universe.values() for r in rs
            if 0.05 < r.alpha <= 0.15
        )
        hold = sum(
            1 for rs in results_by_universe.values() for r in rs
            if -0.05 <= r.alpha <= 0.05
        )
        sell = sum(
            1 for rs in results_by_universe.values() for r in rs if r.alpha < -0.05
        )

        return (
            f"\n📊 <b>VUE D'ENSEMBLE MARCHÉ</b>\n"
            f"{ProMessageBuilder.SOFT_DIVIDER}\n\n"
            f"📈 Titres analysés : <b>{total}</b>\n"
            f"🚀 Strong Buy (α &gt; 15%)  : <b>{strong_buy}</b>\n"
            f"🟢 Buy (α 5-15%)         : <b>{buy}</b>\n"
            f"⚪ Hold (α ±5%)          : <b>{hold}</b>\n"
            f"🔴 Sell/Avoid (α &lt; -5%) : <b>{sell}</b>\n"
        )

    @staticmethod
    def alert_flash(ticker: str, alpha: float, reason: str,
                    price: Optional[float] = None,
                    upside: Optional[float] = None) -> str:
        """Alerte flash pour signal très fort."""
        if alpha > 0.20:
            emoji, tag = "🚨🚀", "STRONG BUY"
        elif alpha > 0.10:
            emoji, tag = "📈", "BUY"
        elif alpha < -0.20:
            emoji, tag = "🚨📉", "STRONG SELL"
        elif alpha < -0.10:
            emoji, tag = "📉", "SELL"
        else:
            emoji, tag = "📊", "SIGNAL"

        msg = [
            f"{emoji} <b>FLASH — {tag}</b>",
            f"{ProMessageBuilder.SOFT_DIVIDER}",
            f"🎯 <b>{ticker}</b>  →  α = <b>{alpha:+.4f}</b>",
        ]
        if price:
            msg.append(f"💵 Prix : {price:.2f}")
        if upside:
            msg.append(f"🎁 Potentiel : {upside:+.1f}%")
        msg.append(f"💡 {reason}")
        msg.append(f"\n<i>⏰ {datetime.now().strftime('%H:%M:%S')}</i>")
        return "\n".join(msg)

    @staticmethod
    def footer() -> str:
        """Pied de message."""
        return (
            f"\n{ProMessageBuilder.DIVIDER}\n"
            f"🤖 <b>AlphaForge Bot</b> · CPA Engine v1.0\n"
            f"📚 RIM+Bayes · FF5 · Ornstein-Uhlenbeck · Kalman\n"
            f"⚠️  <i>Info uniquement — pas un conseil financier. DYOR.</i>"
        )

    @staticmethod
    def heartbeat(iteration: int) -> str:
        """Message de heartbeat pour la boucle."""
        now = datetime.now().strftime("%H:%M:%S")
        return (
            f"💓 <b>Heartbeat #{iteration}</b>\n"
            f"🕐 {now}\n"
            f"<i>Bot opérationnel. Prochain scan dans quelques minutes.</i>"
        )

    @staticmethod
    def _alpha_bar(alpha: float, length: int = 10) -> str:
        """Barre visuelle de l'alpha."""
        normalized = max(-1.0, min(1.0, alpha / 0.3))  # normaliser sur ±30%
        if normalized > 0:
            filled = int(abs(normalized) * length)
            return "🟩" * filled + "⬜" * (length - filled)
        else:
            filled = int(abs(normalized) * length)
            return "🟥" * filled + "⬜" * (length - filled)
