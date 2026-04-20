"""
Récupération automatique du Chat ID Telegram.

Usage :
    1. Lancer : python tools/get_chat_id.py
    2. Ouvrir Telegram, trouver ton bot, envoyer /start
    3. Le chat_id s'affiche automatiquement
"""
import os
import sys
import time
import requests
from pathlib import Path


def get_chat_id(token: str, timeout: int = 120) -> str:
    """
    Écoute les messages entrants et retourne le chat_id du premier utilisateur.
    """
    print(f"\n{'='*60}")
    print("  TELEGRAM CHAT ID DISCOVERY")
    print(f"{'='*60}\n")

    # 1. Vérifier le bot
    try:
        info = requests.get(
            f"https://api.telegram.org/bot{token}/getMe", timeout=10
        ).json()
        if not info.get("ok"):
            print(f"❌ Token invalide: {info}")
            sys.exit(1)
        bot = info["result"]
        print(f"✅ Bot connecté : @{bot['username']} ({bot['first_name']})")
    except Exception as e:
        print(f"❌ Connexion échouée : {e}")
        sys.exit(1)

    # 2. Supprimer le webhook (sinon getUpdates ne fonctionne pas)
    requests.post(f"https://api.telegram.org/bot{token}/deleteWebhook", timeout=10)

    # 3. Instructions
    print(f"\n📱 Ouvre Telegram : https://t.me/{bot['username']}")
    print(f"📩 Envoie n'importe quel message au bot (ex: /start)")
    print(f"⏳ En attente (timeout {timeout}s)...\n")

    # 4. Polling jusqu'à recevoir un message
    start = time.time()
    last_update_id = 0

    while time.time() - start < timeout:
        try:
            resp = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"offset": last_update_id + 1, "timeout": 10},
                timeout=15,
            ).json()

            if resp.get("ok") and resp.get("result"):
                for update in resp["result"]:
                    last_update_id = update["update_id"]
                    msg = update.get("message") or update.get("edited_message")
                    if msg and "chat" in msg:
                        chat = msg["chat"]
                        chat_id = str(chat["id"])
                        user = msg.get("from", {})
                        print(f"✅ CHAT ID RÉCUPÉRÉ !")
                        print(f"   Chat ID : {chat_id}")
                        print(f"   Type    : {chat.get('type')}")
                        print(f"   Nom     : {user.get('first_name', '')} {user.get('last_name', '') or ''}")
                        print(f"   Username: @{user.get('username', 'N/A')}")
                        return chat_id
        except Exception as e:
            print(f"⚠️  {e}")
        time.sleep(2)
        print("  ...en attente...", end="\r")

    print("\n❌ Timeout : aucun message reçu.")
    sys.exit(1)


def save_to_env(token: str, chat_id: str):
    """Sauvegarde dans .env.local (jamais commité)."""
    env_path = Path(__file__).parent.parent / ".env.local"
    content = (
        f"# Telegram — NE JAMAIS COMMITER !\n"
        f"TELEGRAM_BOT_TOKEN={token}\n"
        f"TELEGRAM_CHAT_ID={chat_id}\n"
        f"RISK_FREE_RATE=0.045\n"
    )
    env_path.write_text(content, encoding="utf-8")
    print(f"\n💾 Config sauvée dans : {env_path}")
    print(f"   (Ce fichier est dans .gitignore — il ne sera jamais commité)\n")


def send_welcome(token: str, chat_id: str):
    """Envoie un message de bienvenue pour confirmer."""
    message = (
        "🎉 <b>Connexion réussie !</b>\n\n"
        "✅ Bot CPA Alpha Scanner configuré.\n"
        "📊 Tu recevras bientôt les signaux quant.\n\n"
        "<i>— AlphaForge Bot</i>"
    )
    resp = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
        timeout=10,
    ).json()
    if resp.get("ok"):
        print("✅ Message de bienvenue envoyé sur Telegram !")
    else:
        print(f"⚠️  Erreur envoi : {resp}")


if __name__ == "__main__":
    # Récupérer le token depuis l'argument ou l'environnement
    if len(sys.argv) > 1:
        token = sys.argv[1]
    else:
        token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not token:
            print("❌ Usage : python tools/get_chat_id.py <TOKEN>")
            print("   Ou défini la variable TELEGRAM_BOT_TOKEN")
            sys.exit(1)

    chat_id = get_chat_id(token)
    save_to_env(token, chat_id)
    send_welcome(token, chat_id)

    print(f"\n{'='*60}")
    print("  ✅ CONFIGURATION TERMINÉE")
    print(f"{'='*60}")
    print("\nProchaine étape :")
    print("  → python main.py --test      # test rapide")
    print("  → python main.py             # scan complet")
