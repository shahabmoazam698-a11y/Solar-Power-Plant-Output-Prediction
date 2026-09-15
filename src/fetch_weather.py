import requests
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# TASK 3 - PUBLIC WEATHER DATA FROM OPEN-METEO
# ============================================================


# ------------------------------------------------------------
# 1. Plant 1 location and date range
# ------------------------------------------------------------

latitude = 14.82
longitude = 78.28

start_date = "2020-05-15"
end_date = "2020-06-17"


# ------------------------------------------------------------
# 2. Open-Meteo historical weather API
# ------------------------------------------------------------

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": start_date,
    "end_date": end_date,
    "hourly": "shortwave_radiation,temperature_2m,cloud_cover",
    "timezone": "Asia/Kolkata"
}


# ------------------------------------------------------------
# 3. Download weather data
# ------------------------------------------------------------

print("Downloading Open-Meteo weather data...")

response = requests.get(
    url,
    params=params
)

# Stop the program if the request was unsuccessful
response.raise_for_status()

# Convert API response from JSON
weather_json = response.json()

print("Weather data downloaded successfully.")


# ------------------------------------------------------------
# 4. Extract hourly data from JSON
# ------------------------------------------------------------

hourly = weather_json["hourly"]

weather = pd.DataFrame({
    "datetime": hourly["time"],
    "sw_radiation": hourly["shortwave_radiation"],
    "temp_2m": hourly["temperature_2m"],
    "cloud_cover": hourly["cloud_cover"]
})


# ------------------------------------------------------------
# 5. Convert datetime column
# ------------------------------------------------------------

weather["datetime"] = pd.to_datetime(
    weather["datetime"]
)


# ------------------------------------------------------------
# 6. Inspect downloaded weather data
# ------------------------------------------------------------

print("\nOpen-Meteo rows:", len(weather))

print("\nFirst five Open-Meteo rows:")
print(weather.head())

print("\nLast five Open-Meteo rows:")
print(weather.tail())

print("\nMissing values in Open-Meteo data:")
print(weather.isna().sum())


# ------------------------------------------------------------
# 7. Save Open-Meteo data
# ------------------------------------------------------------

weather.to_csv(
    "data/plant1_openmeteo.csv",
    index=False
)

print("\nSaved: data/plant1_openmeteo.csv")


# ------------------------------------------------------------
# 8. Load Task 1 hourly plant data
# ------------------------------------------------------------

plant = pd.read_csv(
    "data/plant1_hourly.csv"
)

plant["datetime"] = pd.to_datetime(
    plant["datetime"]
)

print("\nPlant hourly rows:", len(plant))


# ------------------------------------------------------------
# 9. Merge plant data and Open-Meteo data
# ------------------------------------------------------------

combined = plant.merge(
    weather,
    on="datetime",
    how="inner"
)

print("\nMerged rows:", len(combined))

print("\nFirst five merged rows:")
print(combined.head())


# ------------------------------------------------------------
# 10. Check missing values after merge
# ------------------------------------------------------------

print("\nMissing values after merge:")
print(combined.isna().sum())


# ------------------------------------------------------------
# 11. Convert plant irradiation units
# ------------------------------------------------------------

# Plant irradiation is in kW/m^2
# Open-Meteo shortwave radiation is in W/m^2
#
# 1 kW/m^2 = 1000 W/m^2

combined["sensor_radiation_wm2"] = (
    combined["irradiation"] * 1000
)


# ------------------------------------------------------------
# 12. Select three days for radiation comparison
# ------------------------------------------------------------

three_days = combined[
    (combined["datetime"] >= "2020-05-15") &
    (combined["datetime"] < "2020-05-18")
].copy()

print("\nRows used for 3-day check:", len(three_days))


# ------------------------------------------------------------
# 13. Plot plant sensor vs Open-Meteo radiation
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    three_days["datetime"],
    three_days["sensor_radiation_wm2"],
    label="Plant Sensor"
)

plt.plot(
    three_days["datetime"],
    three_days["sw_radiation"],
    label="Open-Meteo"
)

plt.xlabel("Date and Time")
plt.ylabel("Radiation (W/m²)")

plt.title(
    "Plant Sensor vs Open-Meteo Radiation - 3 Day Check"
)

plt.legend()

plt.grid(True)

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "results/radiation_3day_comparison.png"
)

plt.show()


# ------------------------------------------------------------
# 14. Find peak radiation time
# ------------------------------------------------------------

sensor_peak_index = (
    three_days["sensor_radiation_wm2"].idxmax()
)

weather_peak_index = (
    three_days["sw_radiation"].idxmax()
)


sensor_peak_time = three_days.loc[
    sensor_peak_index,
    "datetime"
]

weather_peak_time = three_days.loc[
    weather_peak_index,
    "datetime"
]


sensor_peak_value = three_days.loc[
    sensor_peak_index,
    "sensor_radiation_wm2"
]

weather_peak_value = three_days.loc[
    weather_peak_index,
    "sw_radiation"
]


# ------------------------------------------------------------
# 15. Print peak information
# ------------------------------------------------------------

print("\n3-day radiation check:")

print(
    "Plant sensor peak time:",
    sensor_peak_time
)

print(
    "Plant sensor peak value:",
    sensor_peak_value,
    "W/m^2"
)

print(
    "Open-Meteo peak time:",
    weather_peak_time
)

print(
    "Open-Meteo peak value:",
    weather_peak_value,
    "W/m^2"
)


# ------------------------------------------------------------
# 16. Calculate difference between peak times
# ------------------------------------------------------------

peak_difference = abs(
    (
        sensor_peak_time
        - weather_peak_time
    ).total_seconds()
) / 3600


print(
    "\nPeak time difference:",
    peak_difference,
    "hours"
)


# ------------------------------------------------------------
# 17. Check time alignment
# ------------------------------------------------------------

if peak_difference >= 1:

    print(
        "Peak difference is at least 1 hour."
    )

    print(
        "Timezone or time alignment should be investigated."
    )

else:

    print(
        "Peak times are reasonably aligned."
    )


# ------------------------------------------------------------
# 18. Finish
# ------------------------------------------------------------

print("\nTask 3 completed successfully.")
print("Weather file saved as:")
print("data/plant1_openmeteo.csv")

print("\nRadiation comparison graph saved as:")
print("results/radiation_3day_comparison.png")