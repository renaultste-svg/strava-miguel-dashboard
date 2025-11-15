import json
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration Strava via secrets Streamlit
# --------------------------------------------------

STRAVA_BASE_URL = "https://www.strava.com/api/v3"

CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]
REFRESH_TOKEN = st.secrets["REFRESH_TOKEN"]

# Poids pour estimer les kcal CAP
ATHLETE_WEIGHT_KG = float(st.secrets.get("ATHLETE_WEIGHT_KG", 89.0))


# --------------------------------------------------
# Utilitaires Strava
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def get_access_token():
    """
    Utilise le REFRESH_TOKEN pour obtenir un access_token Strava.
    NOTE : on suppose que le REFRESH_TOKEN reste valide.
    """
    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "grant_type": "refresh_token",
            "refresh_token": REFRESH_TOKEN,
        },
        timeout=15,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Erreur Strava OAuth: {resp.status_code} {resp.text}")

    data = resp.json()
    access_token = data.get("access_token")
    if not access_token:
        raise RuntimeError("Impossible de récupérer l'access_token depuis Strava.")
    return access_token


@st.cache_data(show_spinner=True)
def fetch_activities(days: int = 90):
    """
    Récupère les activités Strava des X derniers jours.
    Retourne la liste brute (JSON).
    """
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    after_ts = int((datetime.utcnow() - timedelta(days=days)).timestamp())
    page = 1
    per_page = 200
    all_activities = []

    while True:
        params = {
            "after": after_ts,
            "page": page,
            "per_page": per_page,
        }
        resp = requests.get(
            f"{STRAVA_BASE_URL}/athlete/activities",
            headers=headers,
            params=params,
            timeout=20,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Erreur API Strava: {resp.status_code} {resp.text}")

        activities = resp.json()
        if not activities:
            break

        all_activities.extend(activities)
        page += 1

    return all_activities


def activities_to_dataframe(activities):
    """
    Transforme la liste d'activités en DataFrame exploitable.
    Estime les calories pour la CAP si besoin.
    """
    rows = []
    for a in activities:
        sport = a.get("sport_type") or a.get("type")

        distance_km = (a.get("distance", 0) or 0) / 1000.0
        moving_time_min = (a.get("moving_time", 0) or 0) / 60.0

        calories = a.get("calories")

        # Vélo : parfois on a seulement les kilojoules
        if calories is None and a.get("kilojoules") is not None:
            calories = a["kilojoules"] * 0.239  # approx kJ -> kcal

        # CAP : estimation ~ 1 kcal / kg / km
        if calories is None and sport in ("Run", "TrailRun"):
            calories = ATHLETE_WEIGHT_KG * distance_km

        start_date = a.get("start_date_local") or a.get("start_date")

        rows.append(
            {
                "id": a.get("id"),
                "name": a.get("name"),
                "sport": sport,
                "distance_km": distance_km,
                "moving_time_min": moving_time_min,
                "start_date": start_date,
                "calories": calories,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["start_date"] = pd.to_datetime(df["start_date"])
    df["calories"] = df["calories"].fillna(0.0)

    iso = df["start_date"].dt.isocalendar()
    df["iso_year"] = iso.year
    df["iso_week"] = iso.week
    df["week_label"] = (
        df["iso_year"].astype(str)
        + "-W"
        + df["iso_week"].astype(str).str.zfill(2)
    )

    return df


# --------------------------------------------------
# Calculs de stats (7j / 30j / variation)
# --------------------------------------------------

def format_pace(min_per_km: float) -> str:
    if pd.isna(min_per_km) or min_per_km <= 0:
        return "N/A"
    mins = int(min_per_km)
    secs = int(round((min_per_km - mins) * 60))
    if secs == 60:
        mins += 1
        secs = 0
    return f"{mins:d}:{secs:02d}/km"


def compute_period_stats(df, start_limit):
    """
    Stats globales et CAP pour toutes les activités à partir de start_limit.
    """
    period_df = df[df["start_date"] >= start_limit].copy()
    if period_df.empty:
        return None

    total_kcal = float(period_df["calories"].sum())
    total_acts = int(len(period_df))
    total_time_h = float(period_df["moving_time_min"].sum()) / 60.0

    run_df = period_df[period_df["sport"] == "Run"].copy()
    run_kcal = float(run_df["calories"].sum())
    run_acts = int(len(run_df))
    run_dist = float(run_df["distance_km"].sum())

    if run_dist > 0:
        run_time_min = float(run_df["moving_time_min"].sum())
        run_pace = run_time_min / run_dist
    else:
        run_pace = None

    return {
        "total_kcal": total_kcal,
        "total_activites": total_acts,
        "total_heures": total_time_h,
        "run_kcal": run_kcal,
        "run_activites": run_acts,
        "run_distance_km": run_dist,
        "run_pace_min_per_km": run_pace,
    }


def compute_weekly_totals(df: pd.DataFrame) -> pd.DataFrame:
    weekly = (
        df.groupby(["iso_year", "iso_week", "week_label"])
        .agg(total_kcal=("calories", "sum"))
        .reset_index()
        .sort_values(["iso_year", "iso_week"])
    )
    return weekly


# --------------------------------------------------
# App Streamlit
# --------------------------------------------------

st.set_page_config(
    page_title="Strava – Rapport Miguel-ready",
    layout="wide",
)

st.title("📊 Strava – Rapport Miguel-ready")
st.caption(f"Poids utilisé pour l’estimation CAP : **{ATHLETE_WEIGHT_KG:.1f} kg**")

st.sidebar.header("⚙️ Paramètres")
days = st.sidebar.slider(
    "Période analysée (jours)",
    min_value=30,
    max_value=180,
    step=30,
    value=90,
    help="Durée sur laquelle on récupère les activités depuis Strava.",
)

if st.sidebar.button("🔁 Rafraîchir les données et générer le rapport"):
    st.cache_data.clear()

st.write("Clique sur le bouton ci-dessous pour récupérer les activités et générer ton rapport.")

if st.button("🚀 Générer le rapport maintenant"):
    with st.spinner("Connexion à Strava et récupération des activités…"):
        activities = fetch_activities(days=days)
        df = activities_to_dataframe(activities)

    if df.empty:
        st.error("Aucune activité trouvée sur cette période.")
        st.stop()

    ref_date = df["start_date"].max().normalize()
    start_7d = ref_date - timedelta(days=7)
    start_14d = ref_date - timedelta(days=14)
    start_30d = ref_date - timedelta(days=30)

    stats_7d = compute_period_stats(df, start_7d)
    stats_30d = compute_period_stats(df, start_30d)
    stats_prev_7d = compute_period_stats(df, start_14d)

    # Volume CAP pour 7j et 7j précédents
    run_dist_7d = stats_7d["run_distance_km"] if stats_7d else 0.0

    mask_prev = (
        (df["start_date"] >= start_14d)
        & (df["start_date"] < start_7d)
        & (df["sport"] == "Run")
    )
    run_dist_prev_7d = float(df.loc[mask_prev, "distance_km"].sum())

    if run_dist_prev_7d > 0:
        delta_run_pct = (run_dist_7d - run_dist_prev_7d) / run_dist_prev_7d * 100.0
    else:
        delta_run_pct = None

    # Indicateur "risque charge"
    if delta_run_pct is None:
        charge_comment = "Pas de comparaison possible (pas de CAP sur la période précédente)."
        risque_charge = "N/A"
    else:
        if delta_run_pct > 30:
            risque_charge = "ÉLEVÉ"
            charge_comment = "⚠ Augmentation > 30% : risque blessure/periostite accru."
        elif delta_run_pct > 20:
            risque_charge = "MODÉRÉ"
            charge_comment = "✓ Augmentation modérée mais à surveiller (> 20%)."
        elif delta_run_pct < -10:
            risque_charge = "BASSE"
            charge_comment = "Baisse sensible de charge, plutôt récup."
        else:
            risque_charge = "FAIBLE"
            charge_comment = "✓ Variation faible, charge stable."

    # Petit résumé en haut
    col1, col2, col3 = st.columns(3)
    if stats_7d:
        col1.metric("CAP 7j (distance)", f"{stats_7d['run_distance_km']:.2f} km")
        col2.metric("CAP 30j (distance)", f"{stats_30d['run_distance_km']:.2f} km" if stats_30d else "N/A")
        col3.metric("Risque charge (CAP)", risque_charge)
    else:
        col1.metric("CAP 7j (distance)", "0 km")
        col2.metric("CAP 30j (distance)", f"{stats_30d['run_distance_km']:.2f} km" if stats_30d else "N/A")
        col3.metric("Risque charge (CAP)", "N/A")

    st.markdown("---")

    # Rapport détaillé (texte)
    lines = []
    lines.append("RAPPORT D’ENTRAÎNEMENT – EXTRACTION STRAVA")
    lines.append("==========================================")
    lines.append(f"Date de référence (dernière activité) : {ref_date.strftime('%Y-%m-%d')}")
    lines.append("")
    lines.append("Périodes analysées :")
    lines.append("  - 7 derniers jours (semaine glissante)")
    lines.append("  - 30 derniers jours")
    lines.append("")

    def add_block(title, stats):
        lines.append(title)
        lines.append("-" * len(title))
        if stats is None:
            lines.append("Aucune activité sur cette période.")
            lines.append("")
            return
        lines.append(f"Total calories           : {stats['total_kcal']:.0f} kcal")
        lines.append(f"Nombre d’activités       : {stats['total_activites']}")
        lines.append(f"Temps total              : {stats['total_heures']:.1f} h")
        lines.append("")
        lines.append("Course à pied :")
        lines.append(f"  - Séances             : {stats['run_activites']}")
        lines.append(f"  - Distance totale     : {stats['run_distance_km']:.2f} km")
        lines.append(f"  - Calories CAP        : {stats['run_kcal']:.0f} kcal")
        lines.append(f"  - Allure moyenne CAP  : {format_pace(stats['run_pace_min_per_km'])}")
        lines.append("")

    add_block("📆 7 DERNIERS JOURS", stats_7d)
    add_block("📆 30 DERNIERS JOURS", stats_30d)

    lines.append("Variation volume CAP 7j vs 7j précédents")
    lines.append("-----------------------------------------")
    lines.append(f"Distance CAP 7j en cours      : {run_dist_7d:.2f} km")
    lines.append(f"Distance CAP 7j précédents    : {run_dist_prev_7d:.2f} km")
    if delta_run_pct is not None:
        lines.append(f"Évolution                     : {delta_run_pct:+.1f} %")
        lines.append(charge_comment)
    else:
        lines.append("Pas de comparaison possible (pas de CAP sur la période précédente).")
    lines.append("")

    # Miguel-ready summary
    lines.append("Miguel-ready summary")
    lines.append("---------------------")
    if stats_7d:
        lines.append(f"- CAP 7j : {stats_7d['run_distance_km']:.2f} km sur {stats_7d['run_activites']} séances.")
        lines.append(f"- Allure moyenne CAP (7j) : {format_pace(stats_7d['run_pace_min_per_km'])}.")
        lines.append(f"- Calories totales 7j (tous sports) : {stats_7d['total_kcal']:.0f} kcal.")
    else:
        lines.append("- Aucune activité sur les 7 derniers jours.")

    if stats_30d:
        lines.append(f"- CAP 30j : {stats_30d['run_distance_km']:.2f} km sur {stats_30d['run_activites']} séances.")
        lines.append(f"- Calories CAP 30j : {stats_30d['run_kcal']:.0f} kcal.")

    lines.append(
        "- Variation volume CAP 7j vs 7j précédents : "
        + (f"{delta_run_pct:+.1f} %" if delta_run_pct is not None else "N/A")
    )
    lines.append(f"- Indicateur de risque de charge CAP : {risque_charge}")
    lines.append("")
    lines.append("Tu peux copier-coller tout ce bloc (ou tout le rapport) dans la discussion avec Miguel.")

    rapport_text = "\n".join(lines)

    st.subheader("📝 Rapport complet")
    st.text(rapport_text)

    st.subheader("🎯 Miguel-ready summary (copie/colle ici dans le chat)")
    miguel_block = "\n".join(lines[lines.index("Miguel-ready summary"):])
    st.code(miguel_block, language="markdown")

    # Petit graphique calories / semaine
    st.subheader("📈 Calories totales par semaine")
    weekly = compute_weekly_totals(df)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(weekly["week_label"], weekly["total_kcal"], marker="o")
    ax.set_xlabel("Semaine")
    ax.set_ylabel("Calories (kcal)")
    ax.set_title("Calories d'entraînement par semaine")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Clique sur le bouton ci-dessus pour générer ton rapport.")
