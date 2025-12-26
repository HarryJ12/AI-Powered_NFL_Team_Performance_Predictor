from prediction import (
    load_data,
    engineer_features,
    run_models,
    predict_matchup,
)

import streamlit as st
import pandas as pd
import numpy as np
import os
import base64

# wide page layout
st.set_page_config(
    page_title="NextDown",
    page_icon="logo2.png",
    layout="wide",
)

# Caches the loaded DataFrame so the CSV is read only once per app run (speeds up in-app performance)
@st.cache_data
def load_df(path: str = "csvs/UseThis.csv"):
    return load_data(path)

# Train all models once and caches them in memory to avoid retraining during in-app usage (speeds up in-app performance)
@st.cache_resource
def train_models(df):
    reg_results, clf_results, meta_results = run_models(engineer_features(df.copy()))

    teams = sorted(df["team"].dropna().unique())
    return {
        "df": df,
        "reg_model": reg_results["model"],
        "clf_model": clf_results["model"],
        "scaler": clf_results["scaler"],
        "reg_features": reg_results["features"],
        "clf_features": clf_results["features"],
        "teams": teams,
        "reg_metrics": reg_results["metrics"],
        "clf_metrics": clf_results["metrics"],
        "meta_metrics": meta_results.get("metrics", {}),
    }

def fmt(x, decimals=2):
    try:
        return f"{float(x):.{decimals}f}"
    except (TypeError, ValueError):
        return str(x)
    
# Render model metrics using HTML
def display_metrics(data):
    lines = []

    rm = data["reg_metrics"]
    lines.append(f"<div style='font-size:12px;'>Linear Regression RMSE: {fmt(rm.get('rmse'))}</div>")
    lines.append(f"<div style='font-size:12px;'>Linear Regression R²: {fmt(rm.get('r2'))}</div>")
    lines.append("<div style='height:8px;'></div>")
    
    cm = data["clf_metrics"]
    lines.append(f"<div style='font-size:12px;'>Logistic Regression Accuracy: {fmt(cm.get('accuracy'))}</div>")
    lines.append(f"<div style='font-size:12px;'>Logistic Regression F1: {fmt(cm.get('f1'))}</div>")
    lines.append(f"<div style='font-size:12px;'>Logistic Regression Brier: {fmt(cm.get('brier'))}</div>")
    lines.append(f"<div style='font-size:12px;'>Logistic Regression ROC-AUC: {fmt(cm.get('roc_auc'))}</div>")
    lines.append("<div style='height:8px;'></div>")

    mm = data["meta_metrics"]
    lines.append(f"<div style='font-size:13px;'>Meta RMSE: {fmt(mm.get('rmse'))}</div>")
    lines.append(f"<div style='font-size:13px;'>Meta R²: {fmt(mm.get('r2'))}</div>")

    content_html = f"<div style='padding:0; margin-top:-10px; margin-bottom:6px;'>\n{''.join(lines)}\n</div>"
    st.markdown(content_html, unsafe_allow_html=True)

# Main Streamlit application
#   - Renders logo and header using HTML
#   - Loads data and trains models (cached for performance)
#   - Provides UI for predicting matchups between two teams 
def app():

    # Logo
    logo_html = ""
    with open("logo2.png", "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    logo_html = (
        "<img src='data:image/png;base64," + img_b64 + "' "
        "style='width:96px; height:auto; margin-right:18px; display:block;' />"
    )

    # Header
    header_html = f"""
    <div style='display:flex; align-items:center; gap:12px; padding-top:6px;'>
      {logo_html}
      <div>
        <h1 style='margin:0; padding:0;'>NextDown: AI-Driven NFL Team Performance Predictor</h1>
      </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # Load data and train models
    df = load_df()
    data = train_models(df)

    # Tabs for different functionalities
    tabs = st.tabs([
        "Home: Predict Matchups",
        "Create What-If Scenarios",
        "Recent Matchup Statistics",
    ])

    # Home tab — prediction UI
    with tabs[0]:

        # Prediction subheader
        st.markdown(
            "<div style='color:#bfc6c9; margin-top:6px; margin-bottom:17px; font-size:17px;'>Pick two teams to predict score and win probabilities</div>",
            unsafe_allow_html=True,
        )

        # Team selection (dropdowns) & prediction button that triggers and outputs predictions
        teams = data["teams"]
        col1, col2 = st.columns(2)

        with col1:
            team1 = st.selectbox("Team 1", teams, index=0, key="predict_team1")
        with col2:
            team2 = st.selectbox("Team 2", teams, index=1 if len(teams) > 1 else 0, key="predict_team2")

        if team1 == team2:
            st.warning("Please select two different teams.")
        else:
            if st.button("Predict", key="predict_button"):
                matchup = predict_matchup (
                    team1,
                    team2,
                    data["df"],
                    data["reg_model"],
                    data["clf_model"],
                    data["scaler"],
                    data["reg_features"],
                    data["clf_features"],
                )

                scores = matchup.get("predicted_scores", {})
                probs = matchup.get("win_probabilities", {})
                c1, c2 = st.columns(2)
                
                with c1:
                    st.markdown(f"### {team1}")
                    st.metric(label="Predicted Score", value=scores.get(team1, "—"))
                    st.markdown(f"**Win Probability:** {probs.get(team1, '—')}")
                with c2:
                    st.markdown(f"### {team2}")
                    st.metric(label="Predicted Score", value=scores.get(team2, "—"))
                    st.markdown(f"**Win Probability:** {probs.get(team2, '—')}")
        
        st.markdown("<div style='margin-top:3px;'></div>", unsafe_allow_html=True)

        # Model metrics dropdown
        with st.expander("Model Metrics"):
            display_metrics(data)
    
        # Footer with links
        with st.expander("More Information"):
            lines2 = []
            lines2.append(f"<div style='font-size:13px;'><a href='https://github.com/HarryJ12' target='_blank' style='; text-decoration:none;'>GitHub Project Reopsitory</a></div>")
            lines2.append(f"<div style='font-size:13px;'><a href='https://www.kaggle.com/datasets/cviaxmiwnptr/nfl-team-stats-20022019-espn/data' target='_blank' style='; text-decoration:none;'>Kaggle NFL Team Stats Dataset</a></div>")
            lines2.append("<div style='height:8px;'></div>")
            lines2.append(f"<div style='font-size:13px;'>Created By <a href='https://www.linkedin.com/in/harryjoshi1/' target='_blank'>Harry Joshi</a>, CS Student @ Umass Lowell</div>" )
            content_html2 = f"<div style='padding:0; margin-top:-10px; margin-bottom:6px;'>\n{''.join(lines2)}\n</div>"
            st.markdown(content_html2, unsafe_allow_html=True)

    # What-if tab placeholder
    with tabs[1]:
        st.markdown(
            "<div style='color:#bfc6c9; margin-top:6px; margin-bottom:17px; font-size:17px;'>Create what-if scenarios by adjusting team statistics to see predicted outcomes</div>",
            unsafe_allow_html=True,
        )
        st.info("This feature is under development and will be available soon.")

    # Recent matchup statistics tab
    with tabs[2]:

        # Recent matchup statistics subheader
        st.markdown(
            "<div style='color:#bfc6c9; margin-top:6px; margin-bottom:17px; font-size:17px;'>Pick two teams to retrieve their 2022-2024 matchup statistics (2025 season coming soon)</div>",
            unsafe_allow_html=True,
        )

        # Team selection (dropdowns) for recent matchup statistics that retrieves and 
        # displays last three seasons of relevant game data between two teams
        teams = sorted(df["team"].dropna().unique())
        c1, c2 = st.columns(2)

        with c1:
            hist_team = st.selectbox("Team 1", teams, index=0, key="hist_team")
        with c2:
            hist_opponent = st.selectbox("Team 2", teams, index=1 if len(teams) > 1 else 0, key="hist_opponent")

        if hist_team == hist_opponent:
            st.warning("Please select two different teams.")
        else:
            if st.button("Retrieve", key="retrieve_history"):
                opp_col = None
                for candidate in ["opponent"]:
                    if candidate in df.columns:
                        opp_col = candidate
                        break
                if opp_col is None:
                    st.error("Could not find an 'opponent' column in the dataset. Expected an opponent column.")
                else: # selects matchup regardless of home/away status
                    mask = (
                        ((df["team"] == hist_team) & (df[opp_col] == hist_opponent))
                        | ((df["team"] == hist_opponent) & (df[opp_col] == hist_team))
                    )

                    requested_cols = [
                        "season",
                        "week",
                        "win_numeric",
                        "score_for",
                        "score_against",
                        "first_downs",
                        "first_downs_pass",
                        "first_downs_rush",
                        "third_down_comp",
                        "third_down_att",
                        "fourth_down_comp",
                        "fourth_down_att",
                        "plays",
                        "drives",
                        "yards",
                        "pass_comp",
                        "pass_att",
                        "pass_yards",
                        "sacks_num",
                        "sacks_yards",
                        "rush_att",
                        "rush_yards",
                        "pen_num",
                        "redzone_comp",
                        "redzone_att",
                        "fumbles_lost",
                        "fumbles_forced",
                        "interceptions_thrown",
                        "interceptions_forced",
                        "turnovers_forced",
                        "turnovers_commited",
                        "turnover_diff_pct",
                        "def_st_td",
                        "possession_time",
                        "redzone_efficiency",
                        "third_down_efficiency",
                        "yards_per_play",
                        "rush_yards_per_attempt",
                        "pass_completion_pct",
                    ]

                    # Resolve columns during display
                    available_cols = [c for c in requested_cols if c in df.columns]

                    context_cols = ["team", opp_col]
                    for date_candidate in ("date", "game_date", "match_date"):
                        if date_candidate in df.columns:
                            context_cols.insert(0, date_candidate)
                            break

                    cols_to_show = context_cols + available_cols
                    matches = df.loc[mask, cols_to_show]

                    # No matches found handling 
                    if matches.empty:
                        st.info(
                            f"No games from seasons 2022–2024 between {hist_team} and {hist_opponent}."
                        )
                        st.stop()

                    # Group same games pairs (home/away) together
                    matches_sorted = matches.sort_index()
                    idx = matches_sorted.index.to_series()
                    game_ids = (idx.diff().ne(1)).cumsum()

                    # Winner labeling
                    if {"team", opp_col, "win_numeric"}.issubset(matches_sorted.columns):
                        matches_sorted["winner"] = np.where(
                            matches_sorted["win_numeric"].astype(int) == 1,
                            matches_sorted["team"],
                            matches_sorted[opp_col],
                        )
                        matches_sorted.drop(columns="win_numeric", inplace=True)

                    # Insert blank rows between games
                    groups = []
                    for _, g in matches_sorted.groupby(game_ids, sort=False):
                        groups.append(g)
                        groups.append(pd.DataFrame("", index=[0], columns=g.columns))

                    matches_sorted = pd.concat(groups[:-1], ignore_index=True)

                    # Presentation cleanup
                    if "team" in matches_sorted.columns:
                        matches_display = matches_sorted.set_index("team")
                    else:
                        matches_display = matches_sorted

                    matches_display.columns = matches_display.columns.str.replace("_", " ")
                    st.dataframe(matches_display)

if __name__ == "__main__":
    app()
    