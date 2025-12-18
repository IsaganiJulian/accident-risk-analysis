from sqlalchemy import between
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import date
today = date.today()


from dotenv import load_dotenv
import os
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")

from langchain_community.tools.tavily_search import TavilySearchResults
tavily=TavilySearchResults()



from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize LLM
llm = ChatGroq(model='gemma2-9b-it', groq_api_key=GROQ_API_KEY)

# Page config
st.set_page_config(page_title="Accident Precaution App", layout="wide")






# Load data
@st.cache_data
def load_data():
    # use_cols = [
    #     "Severity", "Start_Time", "End_Time", "Start_Lat", "Start_Lng", "Description", "Street", "City", "County", "State",
    #     "Humidity(%)", "Weather_Condition", "Start_Hour", "Start_DayOfWeek", "Start_Month", "Visibility(km)",
    #     "Pressure(mm)", "Wind_Speed(kmh)", "Precipitation(mm)", "Temperature(C)", "Wind_Chill(C)",
    #     "Start_Year", "Start_Weekday", "Visibility_Category"
    # ]
    # df = pd.read_csv("result_data.csv", usecols=use_cols,nrows=500000)
    df = pd.read_csv("data/result_data.csv")
    return df

df = load_data()
# Sidebar inputs
st.sidebar.header("Enter Travel Conditions")
city = st.sidebar.text_input("City", "Denton")
state = st.sidebar.text_input("State (2-letter code)", "Texas")
hour = st.sidebar.slider("Hour (0-23)", 0, 23, 8)
weekday = st.sidebar.selectbox("Weekday", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
show_advanced = st.sidebar.checkbox("🔍 Show Advanced Insights")

# Filter data
filtered = df[
    (df["City"].str.contains(city, case=False, na=False)) &
    (df["State"].str.upper() == state.upper()) &
    (df["Start_Hour"].between(hour - 2, hour + 2, inclusive="both")) &
    (df["Start_Weekday"] == weekday) 
]

# Title
st.title("🚦 Accident Precaution Assistant")
st.markdown(f"Showing past **accidents** under **similar conditions** in **{city.title()}, {state.upper()}** at hour **{hour}** on a **{weekday}**")

# Show results
st.subheader("🔎 Matching Historical Accident Records")
if filtered.empty:
    st.warning("No records found for this combination. Conditions might be low-risk or underreported.")
else:
    st.success(f"{len(filtered)} records found under similar conditions.")
    st.dataframe(filtered[[
        "Start_Time", "End_Time", "Severity", "Weather_Condition",
        "Start_Lat", "Start_Lng", "Description"
    ]].sort_values("Start_Time", ascending=False))

    st.map(
    filtered[["Start_Lat", "Start_Lng"]]
    .dropna()
    .rename(columns={"Start_Lat": "latitude", "Start_Lng": "longitude"})
)

    # --- Visualizations on matched results ---
    st.markdown("---")
    st.header("📊 Trends in Similar Accidents")

    # Bar chart: Accident Severity
    st.subheader("Severity Levels Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered, x="Severity", ax=ax1)
    ax1.set_title("Severity Distribution of Matching Accidents")
    st.pyplot(fig1)

    # Bar chart: Top 5 Streets
    if "Street" in filtered.columns:
        st.subheader("Top Streets with Accidents under Similar Conditions")
        top_streets = filtered["Street"].value_counts().head(5)
        fig2, ax2 = plt.subplots()
        sns.barplot(x=top_streets.values, y=top_streets.index, ax=ax2)
        ax2.set_title("Top 5 Streets (if available)")
        ax2.set_xlabel("Accident Count")
        st.pyplot(fig2)

    # Hour histogram from filtered set (Optional)
    st.subheader("Distribution by Hour (within matched)")
    fig3, ax3 = plt.subplots()
    sns.histplot(filtered["Start_Hour"], bins=24, ax=ax3, kde=True)
    ax3.set_title("Hour Distribution for Matching Records")
    ax3.set_xlabel("Hour")
    st.pyplot(fig3)


if show_advanced:
    st.markdown("---")
    st.header("📈 Advanced Insights")
    response = tavily.invoke(
        f"Give me today's weather report for {city}, {state} including Visibility (in km), Pressure (in mm), Wind Speed (in km/h), Precipitation (in mm), Temperature (in Celsius), and Wind Chill (in Celsius)."
    )

    import re

    def f_to_c(f):
        return round((f - 32) * 5 / 9, 1)

    def mph_to_kmph(mph):
        return round(mph * 1.60934, 1)

    def miles_to_km(mi):
        return round(mi * 1.60934, 1)

    def inHg_to_mmHg(inhg):
        return round(inhg * 25.4, 1)

    # Your full content string
    content = response[0]["content"]  # paste your full string here

    # Extract temperature
    temp_f_match = re.search(r'Temp\.\s+(\d+\.?\d*)Â?°F', content)
    temperature_c = f_to_c(float(temp_f_match.group(1))) if temp_f_match else None

    # Extract wind speed
    wind_match = re.search(r'Wind\s+(\d+\.?\d*) mph', content)
    wind_kmh = mph_to_kmph(float(wind_match.group(1))) if wind_match else None

    # Extract visibility

    vis_match = re.search(r'Vis\.\s*\|\s*(\d+\.?\d*)', content)

    visibility_km = miles_to_km(float(vis_match.group(1)))if vis_match else None

    # Extract pressure (inHg)
    pressure_match = re.search(r'Alt\.\s+(\d+\.?\d*) inHg', content)
    pressure_inhg = float(pressure_match.group(1)) if pressure_match else None
    pressure_mmhg = inHg_to_mmHg(pressure_inhg) if pressure_inhg else None

    # Extract humidity
    humidity_match = re.search(r'Rel\. Humidity\s*\|\s*(\d+)%', content)
    humidity = int(humidity_match.group(1)) if humidity_match else None

    # Extract wind direction
    wind_dir_match = re.search(r'Wind Dir\.\s*\|\s*(\d+)\s*deg,\s*([NSEW]{1,2})', content)
    wind_dir_deg = int(wind_dir_match.group(1)) if wind_dir_match else None
    wind_dir_cardinal = wind_dir_match.group(2) if wind_dir_match else None

    # Precipitation
    precip_mm = 0.0 if "Precipitation" in content and "No Report" in content else None

    # Wind Chill = Temperature (fallback)
    wind_chill_c = temperature_c

    # ✅ Final Output
    print("📊 Weather Report — July 23, 2025")
    print(f"Temperature: {temperature_c} °C")
    print(f"Wind Speed: {wind_kmh} km/h")
    print(f"Visibility: {visibility_km} km")
    print(f"Pressure: {pressure_mmhg} mmHg ({pressure_inhg} inHg)")
    print(f"Humidity: {humidity}%")
    print(f"Precipitation: {precip_mm} mm")
    print(f"Wind Chill: {wind_chill_c} °C")
    print(f"Wind Direction: {wind_dir_deg}° ({wind_dir_cardinal})")
    # Show results
    st.subheader("🔎 Matching Historical Accident Records")
    st.markdown(f"## 📊 Weather Report — {today.strftime('%B %d, %Y')}")

    st.markdown(f"""
    - **Temperature**: {temperature_c} °C  
    - **Wind Speed**: {wind_kmh} km/h  
    - **Visibility**: {visibility_km} km  
    - **Humidity**: {humidity}%  
    - **Precipitation**: {precip_mm} mm  
    - **Wind Chill**: {wind_chill_c} °C  
    - **Wind Direction**: {wind_dir_deg}° ({wind_dir_cardinal})  
    """)


    prompt = ChatPromptTemplate.from_template("""
    You are a traffic safety expert analyzing accident risk.

    Here is today's weather report:
    Temperature: {temperature} °C  
    Wind Speed: {wind_speed} km/h  
    Visibility: {visibility} km  
    Humidity: {humidity}%  
    Precipitation: {precip} mm  
    Wind Chill: {wind_chill} °C  
    Wind Direction: {wind_dir}° ({wind_cardinal})  

    Here is historical accident data under similar conditions:
    {accident_data}

    Based on this data, provide actionable traffic safety recommendations in a human-readable format with numbered bullet points and relevant emojis.
    """)

    chain = prompt | llm | StrOutputParser()


    st.header("🧠 LLM-Powered Traffic Safety Suggestions")

    suggestions = chain.invoke({
            "temperature": temperature_c,
            "wind_speed": wind_kmh,
            "visibility": visibility_km,
            "humidity": humidity,
            "precip": precip_mm,
            "wind_chill": wind_chill_c,
            "wind_dir": wind_dir_deg,
            "wind_cardinal": wind_dir_cardinal,
            "accident_data": filtered
        })

    st.markdown(suggestions)
