# Here will be the code for the frontend


from enum import Enum, auto

import pandas as pd
import streamlit as st
from streamlit import session_state as ss
import plotly.express as px
import app_functions as app_functions
import data as data
import predictions as predictions
import predictions_test as predictions_test
import plotly.graph_objects as go
import numpy as np


# --- Streamlit Configuration ---
st.set_page_config(page_title="Sprottenflotte prediction model", page_icon="🚲", layout="wide")

def reset_app():
    """Resets the app."""

    # clear session state
    ss.clear()

    # clear cache
    st.cache_data.clear()
    st.cache_resource.clear()


# --- Main App Logic ---
def main():
    # --- initialise ---
    # Initialise Streamlit Interface
    st.title("Sprottenflotte prediction model 🚲 x 🤖")
    st.write("""Thank you for using the Sprottenflotte prediciton model! This model is still in beta
              - We are happy to hear your feedback.
             Please report any issues to Claas Resow.""")
    
    st.write("""Stündliche Daten. Nur nach neuer vollen Stunde neu laden.""")
    
    # initialise tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Tabellenansicht", "Kartenansicht", "Predictions", "Testebene"])


    stations_filename = "data/stations.csv"
    stations_df = pd.read_csv(stations_filename)

    # Check for first load or reset action
    if 'initialized' not in ss: # or st.button("Reset App", on_click=reset_app)
        reset_app()

        weather_data_df, weather_data_message_type, weather_data_message_text = data.update_weather_data()
        data_df, data_message_type, data_message_text = data.update_station_data()
        test_df_cool = predictions_test.make_dataframe_for_prediction_model(data_df, weather_data_df, stations_df)
        # predictions_df, pred_message_type, pred_message_text = predictions_test.update_predictions(data_df, weather_data_df, stations_df)
        test_data = predictions_test.update_predictions(data_df, weather_data_df, stations_df)
        # st.dataframe(test_data, use_container_width=True)
        st.write(test_data)


        predictions_df, pred_message_type, pred_message_text = predictions.update_predictions(data_df) # use data_df weil in der function sonst eine veraltete version von den daten eingelesen wird, wichtig bei stundenänderung
        
        ss['test_df_cool'] = test_df_cool

        ss['weather_data_df'] = weather_data_df
        ss['data_df'] = data_df
        ss['predictions_df'] = predictions_df

        ss['initialized'] = True
        # if data_message_type == 'success':
        # --- Easter Egg --->
        # Set random bike position in session state 
        # if 'random_bike' not in ss: # or data_message_type == 'success': 
        random_subarea, new_lat, new_lon = data.update_random_bike_location(stations_df)
        ss['random_bike'] = {'subarea': random_subarea, 'latitude': new_lat, 'longitude': new_lon}
        # anzeigen des punktes als dataframe
        bike_df = pd.DataFrame([ss['random_bike']])
        st.dataframe(bike_df)
        # <--- Easter Egg ---

    else:
        weather_data_df = ss.get('weather_data_df')
        weather_data_message_type =  None # 'info'
        weather_data_message_text =  None # 'Es sind bereits Daten für alle Wetterstationen vorhanden.'

        data_df = ss.get('data_df')
        data_message_type = None # 'info'
        data_message_text = None # 'Es sind bereits Daten für alle Stationen vorhanden.'

        predictions_df = ss.get('predictions_df')
        pred_message_type = None # 'info'
        pred_message_text = None # 'Es sind bereits Predictions für alle Stationen vorhanden.'

        test_df_cool = ss.get('test_df_cool')
        
        # st.write("Data is already initialized, no need to reload. Only reload nach voller Stunde")

    if predictions_df is None:
        predictions_df = pd.read_csv('data/predictions.csv')
        st.error("predictions_df is None")
    
    full_df = app_functions.get_full_df_per_station(data_df, predictions_df, stations_df)
    
    # Define a color map
    color_map = {
        'überfüllt': 'blue',
        'zu leer': 'red',
        'okay': 'green',
        'no data': 'grey'
    }

    # add current capacity and color to stations_df
    stations_df = app_functions.add_current_capacity_to_stations_df(stations_df, data_df, color_map)

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
    stations_df = app_functions.add_predictions_to_stations_df(stations_df, predictions_df, color_map_predictions)

    # # --- initialise ---
    # # Initialise Streamlit Interface
    # st.title("Sprottenflotte prediction model 🚲 x 🤖")
    # st.write("""Thank you for using the Sprottenflotte prediciton model! This model is still in beta
    #           - We are happy to hear your feedback.
    #          Please report any issues to Claas Resow.""")
    
    # with st.expander(""):
    #         st.write("""Stündliche Daten. Nur nach neuer vollen Stunde neu laden.""")
    
    prio_df = app_functions.measures_prio_of_subarea(data_df, predictions_df, stations_df)

    ss['subareas'] = prio_df['subarea'].tolist()
    ss['subareas'].append('Alle')  # Option hinzufügen

    selected_option = st.selectbox("Wähle ein Teilgebiet aus:", ss['subareas'], index=0)

    # # initialise tabs
    # tab1, tab2, tab3, tab4 = st.tabs(["Tabellenansicht", "Kartenansicht", "Predictions", "Testebene"])

    # --- tab 1 ---
    with tab1:
        st.write("### Vorhersage - Teilgebiete nach Handlungsbedarf")

        with st.expander("ℹ️ Mehr Informationen zu der Berechnung der Prio anzeigen"):
            st.write("""Die Prio der Subareas wird wie folgt berechnet: """)

        st.dataframe(prio_df, use_container_width=True)

        st.dataframe(test_df_cool, use_container_width=True)

        # st.info('st.info')
        # st.success('st.success')
        # st.error('st.error')
        # st.warning('st.warning')
        # st.button('Show Info', help='helping', icon='ℹ️', disabled=True)
        # st.radio('Show Info', options=[], help='helping for sure')
    
    # --- tab 2 ---
    with tab2:
        st.write("### Historische Analyse")
        app_functions.print_message(weather_data_message_type, weather_data_message_text)
        app_functions.print_message(data_message_type, data_message_text)

        with st.expander("ℹ️ Mehr Informationen zur Karte anzeigen"):
            st.write("""
                     Als Default ist hier das Teilgebiet ausgewählt, welches die höchste Prio hat. Die restlichen Teilgebiete sind nach absteigender Prio sortiert.

                     **Die Farben bedeuten:**
                     - **rot** - zu leer - weniger als 20% der maximalen Kapazität
                     - **grün** - okay - zwischen 20% und 80% der maximalen Kapazität
                     - **blau** - überfüllt - mehr als 80% der maximalen Kapazität
                     - **grau** - no data - keine aktuellen Kapazitätsdaten verfügbar
                    """)

        subarea_df = app_functions.make_dataframe_of_subarea(selected_option, stations_df)

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
        st.plotly_chart(fig, config={"scrollZoom": True})

        selected_station = st.selectbox("Wähle eine Station aus:", subarea_df['station_name'])
        station_data = subarea_df[subarea_df['station_name'] == selected_station].iloc[0]

        # Create a Google Maps URL
        google_maps_url = f"https://www.google.com/maps/search/?api=1&query={station_data['latitude']},{station_data['longitude']}"
        st.markdown(f"[Klicken Sie hier, um {selected_station} in Google Maps zu öffnen]({google_maps_url})")

        st.write("***")
        # st.write("Historische Daten:")

        # if data_df is not None:
        #     data_df['time_utc'] = pd.to_datetime(data_df['time_utc'])
        #     data_df['deutsche_timezone'] = data_df['time_utc'] + pd.Timedelta(hours=1)
        #     st.dataframe(data_df[['entityId','deutsche_timezone','availableBikeNumber']], use_container_width=True)
        # else:
        #     st.error("Failed to load historical data.")

        st.write(f"Daten der Stationen von {selected_option}")

        columns_to_show = ['subarea', 'station_name', 'current_capacity', 'maximum_capacity',  'Delta', 'color_info']
        st.dataframe(subarea_df[columns_to_show], use_container_width=True)

        # st.dataframe(subarea_df, use_container_width=True)

        # st.write("Wetterstation Data:")
        # st.dataframe(weather_data_df, use_container_width=True)

    # --- tab 3 ---
    with tab3:
        st.write("### Predictions")
        app_functions.print_message(pred_message_type, pred_message_text)

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
            
        subarea_df = app_functions.make_dataframe_of_subarea(selected_option, stations_df)

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
        st.plotly_chart(fig, config={"scrollZoom": True})

        st.write("***")
        st.write(f"Daten der Stationen von {selected_option}")

        columns_to_show = ['subarea', 'station_name', 'current_capacity', 'prediction_1h', 'prediction_2h', 'prediction_3h', 'prediction_4h', 'prediction_5h', 'maximum_capacity', 'color_info_predictions']
        st.dataframe(subarea_df[columns_to_show], use_container_width=True)

        # st.write("Daten:")

        # if predictions_df is not None:
        #     predictions_df['time_utc'] = pd.to_datetime(predictions_df['time_utc'])
        #     predictions_df['deutsche_timezone'] = predictions_df['time_utc'] + pd.Timedelta(hours=1)
        #     predictions_df = predictions_df.merge(stations_df[['entityId', 'station_name']], on='entityId', how='left')
        #     st.dataframe(predictions_df[['entityId', 'station_name', 'deutsche_timezone', 'availableBikeNumber']], use_container_width=True)
        #     pivot_df = predictions_df.pivot(index='station_name', columns='deutsche_timezone', values='prediction_availableBikeNumber')
        #     st.dataframe(pivot_df, use_container_width=True)
        # else:
        #     st.error("Failed to load prediction data.")

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
            title=f"Bike Availability Over Time by Station in {selected_option}",
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
            template="plotly_white",
            yaxis=dict(showgrid=True, gridcolor='lightgrey', gridwidth=1, griddash='dot')
    )
        fig.add_vline(x=f"{subarea_df['deutsche_timezone'].iloc[-6]}", line_width=2, line_dash="dash", line_color="black")  
        # Add annotation for the vertical line
        fig.add_annotation(
            x=f"{subarea_df['deutsche_timezone'].iloc[-6]}",
            y=max(subarea_df['availableBikeNumber']),  # Adjust y position as necessary
            text=" Predictions",
            showarrow=False,
            xanchor="left"  # Align text to the left of the vertical line
        )
        # Show the plot
        st.plotly_chart(fig)

        st.write("***")
        st.write(f"Zeitdaten der Stationen von {selected_option}")
        # st.dataframe(subarea_df[['entityId', 'station_name', 'availableBikeNumber', 'deutsche_timezone']], use_container_width=True)
        st.dataframe(subarea_df.pivot(index='station_name', columns='deutsche_timezone', values='availableBikeNumber'), use_container_width=True)

    st.button("Reset App/Reload", on_click=reset_app, key="reset_button")

# --- Entry Point ---
if __name__ == "__main__":
    main()

