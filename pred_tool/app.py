# Here will be the code for the frontend


from enum import Enum, auto

import pandas as pd
import streamlit as st
from streamlit import session_state as ss
import plotly.express as px
import data
import plotly.graph_objects as go

# --- Streamlit Configuration ---
st.set_page_config(page_title="Sprottenflotte prediction model", page_icon="🚲", layout="wide")

def reset_app():
    """Resets the app."""

    # clear session state
    ss.clear()
    initialize_session_state()

    # clear cache
    st.cache_data.clear()
    st.cache_resource.clear()


# --- App Stages ---
class AppStage(Enum):
    """Enum for the different stages of the app."""

    START = auto()
    ANALYSIS_START = auto()

    # define greater or equal
    def __ge__(self, other):
        return self.value >= other.value


# --- Session State Initialization ---
def initialize_session_state():
    """Initializes the session state."""
    if "app_stage" not in ss:
        ss["app_stage"] = AppStage.START
    if "df" not in ss:
        ss["df"] = pd.DataFrame()
    if "analysis_started" not in ss:
        ss["analysis_started"] = False
    if "analysis_done" not in ss:
        ss["analysis_done"] = False
    if "base_rule_expr" not in ss:
        ss["base_rule_expr"] = ""
    if "generated_rules" not in ss:
        ss["generated_rules"] = False
    if "edit_table_id" not in ss:
        ss["edit_table_id"] = 0
    if "download_enabled" not in ss:
        ss["download_enabled"] = False
    if "show_visuals" not in ss:
        ss["show_visuals"] = False  # Default to False (using full dataset)


initialize_session_state()


# --- Helper Functions ---
def get_table_height(rows, max_rows=20, offset=18):
    """Calculates the height of a table."""

    def calc(rows, offset):
        return int(35 + offset + rows * 35)

    return calc(rows, offset) if rows <= max_rows else calc(max_rows, offset)


def increment_edit_table_id():
    """Increments the edit table ID."""
    ss["edit_table_id"] = ss["edit_table_id"] + 1


# --- Main App Logic ---
st.title("Sprottenflotte prediction model 🚲 x 🤖")
st.write(
    "Thank you for using the Sprottenflotte prediciton model! This model is still in beta - We are happy to hear your feedback. Please report any issues to Claas Resow."
)
tab1, tab2, tab3 = st.tabs(["Tabellenansicht", "Kartenansicht", "Historische_Analyse"])

# Load data into a DataFrame
df = data.get_predictions()

with tab1:
    # Show data preview
    st.write("### Vorhersage")
    st.dataframe(df.head())
    
with tab2:
    # Coordinates of Kiel
    latitude = 54.3233
    longitude = 10.1228

    # Create a DataFrame with coordinates (optional for more points)
    data = pd.DataFrame({
        'City': ['Kiel'],
        'Latitude': [latitude],
        'Longitude': [longitude]
    })

    # Plot the map
    fig = px.scatter_mapbox(
        data, 
        lat='Latitude', 
        lon='Longitude', 
        hover_name='City', 
        zoom=10,
        height=600
    )

    # Set the Mapbox style (requires an internet connection)
    fig.update_layout(mapbox_style="open-street-map")

    # Show the map
    st.plotly_chart(fig)


    if "actual" in df.columns and "predicted" in df.columns:
        # Calculate errors
        df["error"] = df["actual"] - df["predicted"]

        # Actual vs Predicted Plot
        st.write("### Actual vs Predicted Plot")
        fig_actual_vs_predicted = go.Figure()
        fig_actual_vs_predicted.add_trace(go.Scatter(x=df.index, y=df["actual"], mode="lines+markers", name="Actual"))
        fig_actual_vs_predicted.add_trace(go.Scatter(x=df.index, y=df["predicted"], mode="lines+markers", name="Predicted"))
        fig_actual_vs_predicted.update_layout(
            title="Actual vs Predicted Values",
            xaxis_title="Index",
            yaxis_title="Values",
            legend_title="Legend"
        )
        st.plotly_chart(fig_actual_vs_predicted)

        # Error Distribution Plot
        st.write("### Error Distribution Plot")
        fig_error_dist = px.histogram(df, x="error", nbins=10, title="Error Distribution")
        fig_error_dist.update_layout(
            xaxis_title="Error",
            yaxis_title="Frequency"
        )
        st.plotly_chart(fig_error_dist)

    else:
        st.error("The uploaded file must contain 'actual' and 'predicted' columns.")


with tab3:
    st.write("### Historische Analyse")




st.button("Reset App", on_click=reset_app)
