import requests
import streamlit as st

from PIL import Image
from streamlit_func import streamlit_mapper

# Streamlit data output is in string, whilst our dataset is not
# Hence, it is necessary to map the values

# Create and Load Mapper
# weekday_map = config["weekday_map"]
# season_map = config["season_map"]
# weathersit_map = config["weathersit_map"]
# binary_map = {0: "no", 1:"yes"}

# Set header
header_images = Image.open("img/streamlit_images/hero_header.jpg")
st.image(header_images)

st.title("Bike Share Prediction")
st.subheader("Submet the following information")

with st.form(key="data_form"):
    weekday = st.selectbox(
        label="What day is today?",
        options=(
            "Sunday",
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday"
        )
    )
    
    dteday = st.date_input(
        label="What is today's date?"
    )

    dtetime = st.time_input(
        label="What time is it now?"
    )

    workingday = st.selectbox(
        label="Is today a workingday?",
        options=(
            "Yes",
            "No"
        )
    )

    holiday = st.selectbox(
        label="Is today a holiday?",
        options=(
            "Yes",
            "No"
        )
    )

    season = st.selectbox(
        label="What Kind of Season Right Now?",
        options=(
            "Winter",
            "Spring",
            "Summer",
            "Fall",
        )
    )

    weathersit = st.selectbox(
        label = "How is The Weather?",
        options = (
            "Clear",
            "Mist",
            "Light Snow or Rain",
            "Heavy Rain"
        )
    )

    temp = st.slider(
        label="What is today temperature?",
        min_value=0.02,
        max_value=1.0,
        help = "Enter the normalized temperature in celcius"
    )

    hum = st.slider(
        label="What is today humidity?",
        min_value=0.0,
        max_value=1.0,
        help="Enter the normalized humidity" 
    )

    windspeed = st.slider(
        label="How is the windspeed today?",
        min_value=0.0,
        max_value=0.8507,
        help=""
    )
    submitted = st.form_submit_button("Predict")

    if submitted:
        # Mapping submitted values
        map_season = streamlit_mapper(season.lower(), "season")
        map_weekday = streamlit_mapper(weekday.lower(), "weekday")
        map_weathersit = streamlit_mapper(weathersit.lower(), "weathersit")
        map_holiday = streamlit_mapper(holiday.lower(), "holiday")
        map_workingday = streamlit_mapper(workingday.lower(), "workingday") 

        map_year = 0 if dteday.year % 2 else 1

        # print(map_season)
        # print(map_weekday)
        # print(map_weathersit)
        # print(map_holiday)
        # print(map_workingday)
        # print(map_year)

        raw_data = {
            "dteday": dteday.day,
            "season": map_season,
            "yr": map_year,
            "mnth": dteday.month,
            "hr": dtetime.hour,
            "holiday": map_holiday,
            "weekday": map_weekday,
            "workingday": map_workingday,
            "weathersit": map_weathersit,
            "temp": temp,
            "hum": hum,
            "windspeed": windspeed
        }

        # raw_data = {
        #     "dteday": dteday.day,
        #     "season": season.lower(),
        #     "yr": dteday.year,
        #     "mnth": dteday.month,
        #     "hr": dtetime.hour,
        #     "holiday": holiday.lower(),
        #     "weekday": weekday.lower(),
        #     "workingday": workingday.lower(),
        #     "weathersit": weathersit.lower(),
        #     "temp": temp,
        #     "hum": hum,
        #     "windspeed": windspeed
        # }

        with st.spinner("Sending data to prediction server ..."):
            res = requests.post("http://localhost:8088/predict", json=raw_data).json()

        # st.success(res)
        if res["error_msg"] != "":
            st.error("Error occurs while predicting: {}".format(res["error_msg"]))
        else:
            st.success(f"The number of bike rent: {res['res']}")