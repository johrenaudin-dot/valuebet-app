# -*- coding: utf-8 -*-
import streamlit as st
import requests
import os

st.title("🔑 Test API Odds")

# Récupérer la clé API depuis les secrets Streamlit
api_key = st.secrets.get("ODDS_API_KEY", None)

if not api_key:
    st.error("❌ Aucune clé API trouvée dans les secrets. Ajoute ODDS_API_KEY dans Streamlit Cloud.")
else:
    st.success("✅ Clé API trouvée !")

    url = f"https://api.the-odds-api.com/v4/sports/soccer_epl/odds"
    params = {
        "regions": "eu",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "apiKey": api_key
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            st.error(f"❌ Erreur API : {resp.status_code} - {resp.text}")
        else:
            data = resp.json()
            st.success(f"✅ Données reçues : {len(data)} matchs trouvés")
            st.json(data[:2])  # Afficher juste 2 matchs pour test
    except Exception as e:
        st.error(f"⚠️ Erreur lors de l'appel API : {e}")
