# Here will be the code for the frontend


from enum import Enum, auto

import pandas as pd
import streamlit as st
from streamlit import session_state as ss
import plotly.express as px
import data as data
import predictions as predictions
import plotly.graph_objects as go
import numpy as np


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
    if "stations" not in ss:
        ss["stations"] = list()  # Default to False (using full dataset)
    if "subareas" not in ss:
        ss["subareas"] = list()  # Default to False (using full dataset)


# initialize_session_state() # needs to be here?


# --- Helper Functions ---
def print_message(message_type, message_text):
    if message_type == 'info':
        return st.info(message_text)
    elif message_type == 'success':
        return st.success(message_text)
    elif message_type == 'error':
        return st.error(message_text)


@st.cache_data
def make_dataframe_of_subarea(selected_option, stations_df):
    """Creates a DataFrame for the selected subarea based on the 'subarea' column."""
    if selected_option == 'Alle':
        subarea_df = stations_df.copy()
    else:
        subarea_df = stations_df[stations_df['subarea'] == selected_option]

    subarea_df = subarea_df.sort_values(
        ['Prio', 'Delta'],
        ascending=[False, False],  # Sortiere 'Prio' absteigend und 'Delta' in Bezug auf den absoluten Wert
        key=lambda col: (col if col.name != 'Delta' else abs(col))
    ).reset_index(drop=True)
    subarea_df.index += 1  # Setze den Index auf 1 beginnend
    return subarea_df


def make_subareas_dataframe(stations_df):
    """Creates a DataFrame for the subareas, mean delta, and sort"""
    result_df = stations_df.groupby('subarea')['Delta'].apply(lambda x: x.abs().mean()).reset_index(name='mean_absolute_Delta')
    result_df = result_df.sort_values('mean_absolute_Delta', ascending=False).reset_index(drop=True)
    result_df.index += 1
    return result_df


def get_latest_available_bikes(stations_df):
    # Sortiere die Daten nach prediction_time_utc, um sicherzustellen, dass der letzte Wert genommen wird
    stations_df_sorted = stations_df.sort_values(by='time_utc', ascending=True)

    # Gruppiere nach entityId und nehme den letzten Wert für availableBikeNumber
    latest_available_bikes = stations_df_sorted.groupby('entityId')['availableBikeNumber'].last()

    return latest_available_bikes


def add_current_capacity_to_stations_df(stations_df, data_df, color_map):
    """
    Adds current capacity, delta, priority, and color information to the stations DataFrame.

    Parameters:
    - stations_df: DataFrame containing information about stations.
    - data_df: DataFrame containing the latest data used to compute the current capacity.

    Returns:
    - Updated stations_df with added columns: 'current_capacity', 'Delta', 'Prio', 'color_info', and 'color'.
    """
    # Get the latest capacity values from data_df
    latest_available_bikes = get_latest_available_bikes(data_df)

    # Add the current capacity values to the stations_df
    stations_df['current_capacity'] = stations_df['entityId'].map(latest_available_bikes).round()

    # Calculate the Delta to max_capacity
    stations_df['Delta'] = stations_df['current_capacity'] - stations_df['maximum_capacity']

    # Define conditions for priority calculation
    conditions = [
        (stations_df['current_capacity'] > 0.9 * stations_df['maximum_capacity']) | 
        (stations_df['current_capacity'] < 0.1 * stations_df['maximum_capacity']),  # Very high or very low
        (stations_df['current_capacity'] > 0.8 * stations_df['maximum_capacity']) | 
        (stations_df['current_capacity'] < 0.2 * stations_df['maximum_capacity'])   # High or low
    ]

    # Define priority choices according to conditions
    choices = ['❗️❗️', '❗️']

    # Assign priority to stations
    stations_df['Prio'] = np.select(conditions, choices, default='')

    # Add a new column to indicate color based on station conditions
    stations_df['color_info'] = stations_df.apply(
        lambda row: 'no data' if pd.isna(row['current_capacity'])
                    else 'überfüllt' if row['current_capacity'] >= 0.8 * row['maximum_capacity']
                    else 'zu leer' if row['current_capacity'] <= 0.2 * row['maximum_capacity'] 
                    else 'okay',
        axis=1
    )

    # Map the colors to a new column
    stations_df['color'] = stations_df['color_info'].map(color_map)

    return stations_df


def add_predictions_to_stations_df(stations_df, predictions_df, color_map_predictions):
    """Adds 5 prediction columns to stations_df for each prediction time."""
    
    # Make sure both DataFrames have the necessary columns
    if 'entityId' not in stations_df.columns:
        raise ValueError("stations_df must contain entityId column")
    if 'entityId' not in predictions_df.columns or 'prediction_time_utc' not in predictions_df.columns or 'prediction_availableBikeNumber' not in predictions_df.columns:
        raise ValueError("predictions_df must contain 'entityId', 'prediction_time_utc', and 'prediction_availableBikeNumber' columns")
    
    # Pivot the predictions_df so that each prediction time gets its own column
    predictions_pivot = predictions_df.pivot(index='entityId', columns='prediction_time_utc', values='prediction_availableBikeNumber')
    
    # Round the pivoted prediction values to the nearest integer
    predictions_pivot = predictions_pivot.round(0)

    # Reset the column names of the pivoted DataFrame for clarity
    predictions_pivot.columns = [f'prediction_{i+1}h' for i in range(len(predictions_pivot.columns))]
    
    # Merge the predictions with the stations_df
    stations_df = stations_df.merge(predictions_pivot, how='left', on='entityId')

    # Define a function to determine the color based on current and future capacity status
    def determine_color(row):
        current = row['current_capacity']
        future = row['prediction_5h'] # Just example: using the 5th prediction column for simplicity

         # Check for NaN values
        if pd.isna(current) or pd.isna(future):
            return 'no data'  # Return 'no data' which you can map to grey
        
        condition_current_full = current >= 0.8 * row['maximum_capacity']
        condition_current_empty = current <= 0.2 * row['maximum_capacity']
        condition_current_okay = not (condition_current_full or condition_current_empty)
        
        condition_future_full = future >= 0.8 * row['maximum_capacity']
        condition_future_empty = future <= 0.2 * row['maximum_capacity']
        condition_future_okay = not (condition_future_full or condition_future_empty)
        
        if condition_current_empty:
            if condition_future_empty: return 'zu leer - zu leer'
            elif condition_future_okay: return 'zu leer - okay'
            elif condition_future_full: return 'zu leer - überfüllt'
        elif condition_current_full:
            if condition_future_empty: return 'überfüllt - zu leer'
            elif condition_future_okay: return 'überfüllt - okay'
            elif condition_future_full: return 'überfüllt - überfüllt'
        elif condition_current_okay:
            if condition_future_empty: return 'okay - zu leer'
            elif condition_future_okay: return 'okay - okay'
            elif condition_future_full: return 'okay - überfüllt'
        return 'no data'  # default fallback color

    # Apply the determine_color function
    stations_df['color_info_predictions'] = stations_df.apply(determine_color, axis=1)
    
    stations_df['color_predictions'] = stations_df['color_info_predictions'].map(color_map_predictions)
    
    return stations_df


def get_full_df_per_station(stations_df, predictions_df, subarea_df):
    # Concatenate die letzten 24h und die nächsten 5h zu einem DataFrame
    stations_df['time_utc'] = pd.to_datetime(stations_df['time_utc'])
    predictions_df['time_utc'] = pd.to_datetime(predictions_df['prediction_time_utc'])
    predictions_df['availableBikeNumber'] = predictions_df['prediction_availableBikeNumber']

    full_df = pd.concat([stations_df[['entityId','time_utc','availableBikeNumber']], predictions_df[['entityId','time_utc','availableBikeNumber']]], ignore_index=True)
    full_df = full_df.sort_values(by=['entityId','time_utc']).reset_index(drop=True)
    full_df = full_df.merge(subarea_df[['entityId', 'subarea', 'station_name']], on='entityId', how='left')
    full_df['deutsche_timezone'] = full_df['time_utc'] + pd.Timedelta(hours=1)

    return full_df


# Berechnet absolute Prio - Muss noch in relative prio umberechnet werden
def measures_prio_of_subarea(stations_df:pd.DataFrame, predictions_df:pd.DataFrame, subareas_df) -> pd.DataFrame:
    full_df = get_full_df_per_station(stations_df, predictions_df, subareas_df)
    # result_df = pd.DataFrame(columns=['subarea', 'Station', 'Prio'])
    first_iteration = True  # Flag zur Überwachung der ersten Iteration

    stations = full_df['entityId'].unique()
    for station in stations:
        prio = 0

        teilbereich = full_df[full_df['entityId'] == station]['subarea'].unique()[0]
        max_capacity = subareas_df[subareas_df['subarea'] == teilbereich]['maximum_capacity'].unique()[0]
        station_data = full_df[full_df['entityId'] == station]
        availableBikes = station_data['availableBikeNumber']

        if availableBikes.iloc[-1] >= (0.8 * max_capacity):
            prio += 0.5
        if len(availableBikes) >= 5 and availableBikes.iloc[-5:].mean() >= (0.8 * max_capacity):  # Mean of last 5 values
            prio += 0.5
        if len(availableBikes) >= 9 and availableBikes.iloc[-9:].mean() >= (0.8 * max_capacity):  # Mean of last 9 values
            prio += 0.5
        if len(availableBikes) >= 25 and availableBikes.iloc[-25:].mean() >= (0.8 * max_capacity):  # Mean of last 25 values
            prio += 1
        
        if availableBikes.iloc[-1] <= (0.2 * max_capacity):
            prio += 0.5
        if len(availableBikes) >= 5 and availableBikes.iloc[-5:].mean() <= (0.2 * max_capacity):  # Mean of last 5 values
            prio += 0.5
        if len(availableBikes) >= 9 and availableBikes.iloc[-9:].mean() <= (0.2 * max_capacity):  # Mean of last 9 values
            prio += 0.5
        if len(availableBikes) >= 25 and availableBikes.iloc[-25:].mean() <= (0.2 * max_capacity):  # Mean of last 25 values
            prio += 1
    
        temp_df = pd.DataFrame({'subarea': [teilbereich], 'Station': [station], 'Prio': [prio]})

        # Überprüfen, ob es die erste Iteration ist
        if first_iteration:
            result_df = temp_df  # Setze result_df direkt für die erste Iteration
            first_iteration = False
        else:
            result_df = pd.concat([result_df, temp_df], ignore_index=True)
    
    result_df = result_df.groupby('subarea')['Prio'].apply(lambda x: x.mean()).reset_index(name='subarea_prio')
    result_df = result_df.sort_values('subarea_prio', ascending=False).reset_index(drop=True)
    result_df.index += 1
    return result_df





# --- Main App Logic ---
def main():
    # Check for first load or reset action
    if 'initialized' not in ss or st.button("Reset App", on_click=reset_app):
        reset_app()  # Reset the app if not initialized or reset button is clicked
        ss['initialized'] = True

    stations_filename = "data/stations.csv"
    stations_df = pd.read_csv(stations_filename)

    data_df, data_message_type, data_message_text = data.update_station_data()
    predictions_df, pred_message_type, pred_message_text = predictions.update_predictions(data_df) # use data_df weil in der function sonst eine veraltete version von den daten eingelesen wird, wichtig bei stundenänderung
    
    if predictions_df == None:
        predictions_df = pd.read_csv('data/predictions.csv')
    
    full_df = get_full_df_per_station(data_df, predictions_df, stations_df)
    
    # Define a color map
    color_map = {
        'überfüllt': 'blue',
        'zu leer': 'red',
        'okay': 'green',
        'no data': 'grey'
    }

    # add current capacity and color to stations_df
    stations_df = add_current_capacity_to_stations_df(stations_df, data_df, color_map)

    # Map the colors based on a predefined color map
    color_map_predictions = {
        'zu leer - zu leer': 'red',
        'zu leer - okay': 'green',
        'zu leer - überfüllt': 'blue',

        'überfüllt - zu leer': 'red',
        'überfüllt - okay': 'green',
        'überfüllt - überfüllt': 'blue',

        'okay - zu leer': 'red',
        'okay - okay': 'green',
        'okay - überfüllt': 'blue',

        'no data': 'grey'
    }

    # add the 5 predictions to stations_df
    stations_df = add_predictions_to_stations_df(stations_df, predictions_df, color_map_predictions)

    # --- initialise ---
    # Initialise Streamlit Interface
    st.title("Sprottenflotte prediction model 🚲 x 🤖")
    st.write("""Thank you for using the Sprottenflotte prediciton model! This model is still in beta
              - We are happy to hear your feedback.
             Please report any issues to Claas Resow.""")
    
    prio_df = measures_prio_of_subarea(data_df, predictions_df, stations_df)

    ss['subareas'] = prio_df['subarea'].tolist()
    ss['subareas'].append('Alle')  # Option hinzufügen

    selected_option = st.selectbox("Wähle ein Teilgebiet aus:", ss['subareas'], index=0)

    # initialise tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Tabellenansicht", "Kartenansicht", "Predictions", "Testebene"])

    # --- tab 1 ---
    with tab1:
        st.write("### Vorhersage - Teilgebiete nach Handlungsbedarf")

        with st.expander("ℹ️ Mehr Informationen zu der Berechnung der Prio anzeigen"):
            st.write("""Die Prio der Subareas wird wie folgt berechnet: """)

        st.dataframe(prio_df, use_container_width=True)

        # st.info('st.info')
        # st.success('st.success')
        # st.error('st.error')
        # st.warning('st.warning')
        # st.button('Show Info', help='helping', icon='ℹ️', disabled=True)
        # st.radio('Show Info', options=[], help='helping for sure')
    
    # --- tab 2 ---
    with tab2:
        st.write("### Historische Analyse")
        print_message(data_message_type, data_message_text)

        with st.expander("ℹ️ Mehr Informationen zur Karte anzeigen"):
            st.write("""
                     Als Default ist hier das Teilgebiet ausgewählt, welches die höchste Prio hat. Die restlichen Teilgebiete sind nach absteigender Prio sortiert.

                     **Die Farben bedeuten:**
                     - **rot** - zu leer - weniger als 20% der maximalen Kapazität
                     - **grün** - okay - zwischen 20% und 80% der maximalen Kapazität
                     - **blau** - überfüllt - mehr als 80% der maximalen Kapazität
                     - **grau** - no data - keine aktuellen Kapazitätsdaten verfügbar
                    """)

        subarea_df = make_dataframe_of_subarea(selected_option, stations_df)

        # --- Easter Egg --->
        # Set random bike position in session state 
        if 'random_bike' not in ss or data_message_type == 'success': 
            random_subarea, new_lat, new_lon = data.update_random_bike_location(stations_df)
            ss['random_bike'] = {'subarea': random_subarea, 'latitude': new_lat, 'longitude': new_lon}
        # anzeigen des punktes als dataframe
        # bike_df = pd.DataFrame([ss['random_bike']])
        # st.dataframe(bike_df)
        # <--- Easter Egg ---

        # Plot the map
        fig = px.scatter_mapbox(
            subarea_df, 
            lat='latitude', 
            lon='longitude',
            title=f"Teilgebiet: {selected_option}",
            hover_name='station_name',
            hover_data={
                'current_capacity':True,
                'maximum_capacity': True,
                'Delta': True,
                'latitude': False,  # Disable latitude hover
                'longitude': False,  # Disable longitude hover
                'color_info': True,
                'color': False
            },
            color='color_info',  # Use the new column for colors
            color_discrete_map=color_map,
            zoom=10.2,
            height=600,
            labels={
                'color_info': 'Station Info'  # Change title of the legend
            }
        )

        # --- Easter Egg --->
        # Danach den neuen Punkt hinzufügen
        if ss.get('random_bike') and selected_option != 'Alle' and ss['random_bike']['subarea'] == selected_option:
            bike_df = pd.DataFrame([ss['random_bike']])
            hover_text = '🚲 Easter Egg Bike 🚲<br><br>' + \
                         'Latitude: ' + bike_df['latitude'].round(1).astype(str) + '°N<br>' + \
                         'Longitude: ' + bike_df['longitude'].round(1).astype(str) + '°E'

            fig.add_scattermapbox(
                lat = bike_df['latitude'], 
                lon = bike_df['longitude'], 
                text = 'Easter Egg Bike', # 🚲
                mode = 'markers', #+text
                showlegend = False,
                textposition='top center',
                marker = dict(color='#9ec044'),
                name='Easter Egg Bike',
                hovertext = hover_text,
                hoverinfo = 'text'
            )
        # <--- Easter Egg ---

        # Set the Mapbox style (requires an internet connection)
        fig.update_layout(mapbox_style="open-street-map")

        # Adjust the hoverlabel color # bgcolor=subarea_df['color'],
        fig.update_traces(marker=dict(size=12),
                        hoverlabel=dict(#font_family='Serif',
                                        font_size=12,
                                        font_color='#31333F',
                                        bgcolor='#FCFEF6',
                                        bordercolor='#9ec044'))

        # Show the map
        st.plotly_chart(fig)

        selected_station = st.selectbox("Wähle eine Station aus:", subarea_df['station_name'])
        station_data = subarea_df[subarea_df['station_name'] == selected_station].iloc[0]

        # Create a Google Maps URL
        google_maps_url = f"https://www.google.com/maps/search/?api=1&query={station_data['latitude']},{station_data['longitude']}"
        st.markdown(f"[Klicken Sie hier, um {selected_station} in Google Maps zu öffnen]({google_maps_url})")

        st.write("***")
        st.write("Historische Daten:")

        if data_df is not None:
            data_df['time_utc'] = pd.to_datetime(data_df['time_utc'])
            data_df['deutsche_timezone'] = data_df['time_utc'] + pd.Timedelta(hours=1)
            st.dataframe(data_df, use_container_width=True)
        else:
            st.error("Failed to load historical data.")

        st.write("Stations Data:")

        columns_to_show = ['subarea', 'station_name', 'current_capacity', 'maximum_capacity',  'Delta', 'Prio']
        st.dataframe(subarea_df[columns_to_show], use_container_width=True)

        st.dataframe(subarea_df, use_container_width=True)

    # --- tab 3 ---
    with tab3:
        st.write("### Predictions")
        print_message(pred_message_type, pred_message_text)

        with st.expander("ℹ️ Mehr Informationen zur Karte anzeigen"):
            st.write("""
                     Als Default ist hier das Teilgebiet ausgewählt, welches die höchste Prio hat. Die restlichen Teilgebiete sind nach absteigender Prio sortiert.

                     In Zukunft bedeutet bei Stunde 5 der Predictions.

                     **Die Farben bedeuten:**
                     - **rot** - in Zukunft zu leer - 'zu leer - zu leer', 'okay - zu leer', 'überfüllt - zu leer'
                     - **grün** - in Zukunft okay - 'zu leer - okay', 'okay - okay', 'überfüllt - okay'
                     - **blau** - in Zukunft überfüllt - 'zu leer - überfüllt', 'okay - überfüllt', 'überfüllt - überfüllt'
                     - **grau** - no data - keine Daten verfügbar
                    """)
            
        subarea_df = make_dataframe_of_subarea(selected_option, stations_df)

        # Plot the map
        fig = px.scatter_mapbox(
            subarea_df, 
            lat='latitude', 
            lon='longitude',
            title=f"Teilgebiet: {selected_option}",
            hover_name='station_name',
            hover_data={
                'current_capacity':True,
                'maximum_capacity': True,
                'Delta': False,
                'latitude': False,  # Disable latitude hover
                'longitude': False,  # Disable longitude hover
                'color_info_predictions': True,
                'color_predictions': False,
                'prediction_1h': True,
                'prediction_2h': True,
                'prediction_3h': True,
                'prediction_4h': True,
                'prediction_5h': True
            },
            color='color_info_predictions',  # Use the new column for colors
            color_discrete_map=color_map_predictions,
            zoom=10.2,
            height=600,
            labels={
                'color_info_predictions': 'Station Info'  # Change title of the legend
            }
        )

        # Set the Mapbox style (requires an internet connection)
        fig.update_layout(mapbox_style="open-street-map")

        # Adjust the hoverlabel color # bgcolor=subarea_df['color'],
        fig.update_traces(marker=dict(size=12),
                        hoverlabel=dict(#font_family='Serif',
                                        font_size=12,
                                        font_color='#31333F',
                                        bgcolor='#FCFEF6',
                                        bordercolor='#9ec044'))

        # Show the map
        st.plotly_chart(fig)

        st.write("***")
        st.write("Daten:")

        if predictions_df is not None:
            predictions_df['time_utc'] = pd.to_datetime(predictions_df['time_utc'])
            predictions_df['deutsche_timezone'] = predictions_df['time_utc'] + pd.Timedelta(hours=1)
            predictions_df = predictions_df.merge(stations_df[['entityId', 'station_name']], on='entityId', how='left')
            st.dataframe(predictions_df[['entityId', 'station_name', 'deutsche_timezone', 'availableBikeNumber']], use_container_width=True)
            pivot_df = predictions_df.pivot(index='station_name', columns='deutsche_timezone', values='prediction_availableBikeNumber')
            st.dataframe(pivot_df, use_container_width=True)
        else:
            st.error("Failed to load prediction data.")

    # --- tab 4 ---
    with tab4:
        if selected_option == 'Alle':
            subarea_df = full_df
        else:
            subarea_df = full_df[full_df['subarea'] == selected_option]

        fig = px.line(
            subarea_df,
            x='deutsche_timezone',
            y='availableBikeNumber',
            color='station_name',
            title="Bike Availability Over Time by Station",
            labels={
                "deutsche_timezone": "Uhrzeit",
                "availableBikeNumber": "Available Bikes",
                "station": "Station"
            }
        )

        # Customize the layout
        fig.update_layout(
            xaxis_title="Uhrzeit",
            yaxis_title="Available Bikes",
            legend_title="Station",
            template="plotly_white"
    )
        fig.add_vline(x=f"{subarea_df['deutsche_timezone'].iloc[-6]}", line_width=2, line_dash="dash", line_color="blue")  

        # Show the plot
        st.plotly_chart(fig)

        st.dataframe(subarea_df[['entityId', 'station_name', 'availableBikeNumber', 'deutsche_timezone']], use_container_width=True)

    st.button("Reset App", on_click=reset_app, key="reset_button")

# --- Entry Point ---
if __name__ == "__main__":
    main()

