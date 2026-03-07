import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
import numpy as np

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="ScoutIQ – Football Intelligence Platform",
    layout="wide"
)

# ===============================
# CHAMPIONS LEAGUE STYLE
# ===============================

st.markdown("""
<style>

.stApp{
background: radial-gradient(circle at top,#001F54,#000814,#000000);
color:white;
}

[data-testid="stSidebar"]{
background:#020617;
}

h1,h2,h3{
color:white;
}

.block-container{
background:rgba(255,255,255,0.05);
padding:2rem;
border-radius:15px;
backdrop-filter:blur(6px);
}

</style>
""",unsafe_allow_html=True)

# ===============================
# TITLE
# ===============================

st.markdown(
"<h1 style='text-align:center;'>⚽ ScoutIQ – Football Intelligence Platform</h1>",
unsafe_allow_html=True
)

st.markdown("---")

# ===============================
# LANGUAGE
# ===============================

language = st.sidebar.selectbox(
    "Language / Langue",
    ["Français","English"]
)

# ===============================
# LOAD DATA
# ===============================

@st.cache_data
def load_data():

    df = pd.read_csv("female_players.zip", compression="zip", low_memory=False)

    df = df[
        (df["overall"] > 0) &
        (df["value_eur"].notna())
    ]

    df["value_eur"] = df["value_eur"].fillna(0)

    df["market_efficiency"] = df["overall"]/(df["value_eur"]+1)

    return df


df = load_data()

# ===============================
# SIDEBAR MENU
# ===============================

menu = st.sidebar.selectbox(
    "Navigation",
    [
        "🏠 Overview",
        "🔎 Scouting",
        "🔄 Transfer Simulator",
        "📊 Market Intelligence",
        "🤖 AI Profiling",
        "🏆 Club Ranking"
    ]
)

# ===============================
# OVERVIEW DASHBOARD
# ===============================

if menu == "🏠 Overview":

    st.subheader("Global Football Overview")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Players",len(df))
    col2.metric("Average Rating",round(df["overall"].mean(),2))
    col3.metric("Total Market Value",
                f"{round(df['value_eur'].sum()/1e9,2)} B€")
    col4.metric("Average Age",
                round(df["age"].mean(),1))

    st.markdown("### Rating Distribution")

    fig = px.histogram(
        df,
        x="overall",
        nbins=30,
        color_discrete_sequence=["#2563eb"]
    )

    st.plotly_chart(fig,use_container_width=True)

    st.markdown("### Market Value vs Overall")

    fig2 = px.scatter(
        df.sample(1500),
        x="overall",
        y="value_eur",
        hover_data=["short_name","club_name"]
    )

    st.plotly_chart(fig2,use_container_width=True)

# ===============================
# SCOUTING
# ===============================

elif menu == "🔎 Scouting":

    st.subheader("Player Scouting")

    player = st.selectbox(
        "Select Player",
        sorted(df["short_name"].unique())
    )

    player_data = df[df["short_name"]==player].iloc[0]

    col1,col2 = st.columns(2)

    col1.metric("Overall",player_data["overall"])
    col1.metric("Age",player_data["age"])

    col2.metric("Market Value",
                f"{int(player_data['value_eur']/1e6)} M€")

    attributes = [
        "pace","shooting","passing",
        "dribbling","defending","physic"
    ]

    radar_df = pd.DataFrame({
        "Attribute":attributes,
        "Value":[player_data[a] for a in attributes]
    })

    fig = px.line_polar(
        radar_df,
        r="Value",
        theta="Attribute",
        line_close=True
    )

    fig.update_traces(fill="toself")

    st.plotly_chart(fig,use_container_width=True)

# ===============================
# TRANSFER SIMULATOR
# ===============================

elif menu == "🔄 Transfer Simulator":

    st.subheader("Transfer Impact Simulator")

    player = st.selectbox(
        "Player",
        sorted(df["short_name"].unique())
    )

    club = st.selectbox(
        "Destination Club",
        sorted(df["club_name"].dropna().unique())
    )

    if st.button("Simulate Transfer"):

        player_rating = df[df["short_name"]==player]["overall"].values[0]

        team_avg = df[df["club_name"]==club]["overall"].mean()

        impact = player_rating - team_avg

        st.markdown(f"### Transfer Analysis : {player} ➜ {club}")

        col1,col2,col3 = st.columns(3)

        col1.metric("Player Rating",player_rating)
        col2.metric("Team Average",round(team_avg,1))
        col3.metric("Impact",round(impact,2))

        if impact > 2:

            st.success("Major improvement for the squad")

        elif impact > 0:

            st.info("Moderate improvement")

        else:

            st.warning("Likely rotation player")

# ===============================
# MARKET INTELLIGENCE
# ===============================

elif menu == "📊 Market Intelligence":

    st.subheader("Market Efficiency")

    top = df.sort_values(
        "market_efficiency",
        ascending=False
    ).head(20)

    st.dataframe(
        top[
            ["short_name","club_name",
             "overall","value_eur",
             "market_efficiency"]
        ]
    )

    fig = px.scatter(
        df.sample(1500),
        x="overall",
        y="value_eur",
        hover_data=["short_name"]
    )

    st.plotly_chart(fig,use_container_width=True)

# ===============================
# AI PROFILING
# ===============================

elif menu == "🤖 AI Profiling":

    st.subheader("AI Player Archetypes")

    features = df[
        ["pace","shooting","passing",
         "dribbling","defending","physic"]
    ].dropna()

    kmeans = KMeans(n_clusters=4,random_state=42)

    clusters = kmeans.fit_predict(features)

    df_cluster = df.loc[features.index].copy()

    df_cluster["cluster"] = clusters

    fig = px.scatter(
        df_cluster,
        x="pace",
        y="shooting",
        color="cluster",
        hover_data=["short_name"]
    )

    st.plotly_chart(fig,use_container_width=True)

# ===============================
# CLUB RANKING
# ===============================

elif menu == "🏆 Club Ranking":

    st.subheader("Club Power Index")

    ranking = (
        df.groupby("club_name")
        .agg(
            avg_rating=("overall","mean"),
            squad_value=("value_eur","sum")
        )
        .sort_values("avg_rating",ascending=False)
        .reset_index()
    )


    st.dataframe(ranking.head(20))
