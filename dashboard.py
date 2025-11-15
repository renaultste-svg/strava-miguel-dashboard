import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# On réutilise tes fonctions existantes
from fetch_activities import fetch_recent_activities, activities_to_dataframe, ATHLETE_WEIGHT_KG


# --------- Fonctions utilitaires --------- #

@st.cache_data(show_spinner=True)
def get_dataframe(days: int) -> pd.DataFrame:
    """
    Récupère les activités Strava sur X jours
    et renvoie un DataFrame prêt à l'emploi.
    Résultat mis en cache pour éviter de re-requêter Strava à chaque fois.
    """
    activities = fetch_recent_activities(days=days)
    df = activities_to_dataframe(activities)

    if df.empty:
        return df

    # Sécuriser la colonne calories
    df["calories"] = df["calories"].fillna(0.0)

    # Libellé ISO semaine pour les graphes
    df["week_label"] = (
        df["iso_year"].astype(int).astype(str)
        + "-W"
        + df["iso_week"].astype(int).astype(str).str.zfill(2)
    )
    return df


def compute_weekly_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Total des calories par semaine (tous sports)."""
    weekly = (
        df.groupby(["iso_year", "iso_week", "week_label"])
        .agg(
            total_kcal=("calories", "sum"),
            nb_activites=("id", "count"),
        )
        .reset_index()
        .sort_values(["iso_year", "iso_week"])
    )
    return weekly


def compute_weekly_by_sport(df: pd.DataFrame) -> pd.DataFrame:
    """Calories par semaine et par sport (tableau pivot)."""
    pivot = (
        df.pivot_table(
            index="week_label",
            columns="sport",
            values="calories",
            aggfunc="sum",
        )
        .fillna(0.0)
        .sort_index()
    )
    return pivot


# --------- Mise en page Streamlit --------- #

st.set_page_config(
    page_title="Calories Analyzer",
    layout="wide",
)

st.title("📊 Calories Analyzer – Tableau de bord Strava perso")
st.caption(f"Poids utilisé pour l’estimation course à pied : **{ATHLETE_WEIGHT_KG:.1f} kg**")

# --- Barre latérale --- #
st.sidebar.header("⚙️ Paramètres")

days = st.sidebar.slider(
    "Période analysée (en jours)",
    min_value=7,
    max_value=180,
    step=7,
    value=90,
    help="Durée sur laquelle on récupère les activités depuis Strava",
)

st.sidebar.markdown("---")
st.sidebar.markdown("🔁 Les données sont mises en cache. "
                    "Clique sur **Menu ▸ Rerun** si tu changes beaucoup de paramètres.")

# --- Récupération des données --- #
with st.spinner("Récupération des activités Strava…"):
    df = get_dataframe(days)

if df.empty:
    st.warning("Aucune activité trouvée sur cette période.")
    st.stop()

weekly = compute_weekly_totals(df)
weekly_sport = compute_weekly_by_sport(df)

# Liste des sports disponibles
sports = sorted(df["sport"].dropna().unique())
default_sports = sports  # par défaut : tout

selected_sports = st.sidebar.multiselect(
    "Sports affichés dans le graphique par sport",
    options=sports,
    default=default_sports,
)

# Filtre éventuel
if selected_sports:
    df_filtered = df[df["sport"].isin(selected_sports)]
else:
    df_filtered = df.copy()

weekly_sport_filtered = weekly_sport[selected_sports] if selected_sports else weekly_sport

# --------- Indicateurs globaux --------- #
total_kcal = df["calories"].sum()
total_activities = len(df)
total_hours = df["moving_time_min"].sum() / 60

col1, col2, col3 = st.columns(3)
col1.metric("🔥 Calories totales", f"{int(total_kcal):,} kcal".replace(",", " "))
col2.metric("📂 Nombre d’activités", f"{total_activities}")
col3.metric("⏱ Temps de mouvement", f"{total_hours:.1f} h")

st.markdown("---")

# --------- Graphique 1 : total par semaine --------- #
st.subheader("📆 Calories totales par semaine")

fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(weekly["week_label"], weekly["total_kcal"], marker="o")
ax1.set_xlabel("Semaine")
ax1.set_ylabel("Calories (kcal)")
ax1.set_title(f"Calories d’entraînement par semaine (sur {days} jours)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
st.pyplot(fig1)

# --------- Graphique 2 : par sport --------- #
st.subheader("🏃‍♂️🚴‍♂️ Calories par semaine et par sport")

if not weekly_sport_filtered.empty:
    fig2, ax2 = plt.subplots(figsize=(8, 4))

    for sport in weekly_sport_filtered.columns:
        ax2.plot(
            weekly_sport_filtered.index,
            weekly_sport_filtered[sport],
            marker="o",
            label=sport,
        )

    ax2.set_xlabel("Semaine")
    ax2.set_ylabel("Calories (kcal)")
    ax2.set_title("Répartition des calories par sport et par semaine")
    ax2.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.info("Aucun sport sélectionné pour ce graphique.")

st.markdown("---")

# --------- Tableau des dernières activités --------- #
st.subheader("📋 Dernières activités")

# On affiche les colonnes les plus utiles
cols_to_show = [
    "start_date",
    "name",
    "sport",
    "distance_km",
    "moving_time_min",
    "calories",
]

df_show = df.sort_values("start_date", ascending=False)[cols_to_show].copy()
df_show["distance_km"] = df_show["distance_km"].round(2)
df_show["moving_time_min"] = df_show["moving_time_min"].round(1)
df_show["calories"] = df_show["calories"].round(0).astype(int)

st.dataframe(
    df_show,
    use_container_width=True,
    height=400,
)

st.caption(
    "💡 Les calories vélo viennent de Strava (ou des kJ), "
    "celles de la course à pied sont estimées à ~1 kcal / kg / km."
)
