import pandas as pd
from load_data import load_raw


# --------------------------------------------------
# 1. Load the raw CSV files
# --------------------------------------------------

raw = load_raw()

# We are working with Plant 1
gen = raw["gen1"].copy()
sensor = raw["sensor1"].copy()

print("Raw Plant 1 generation rows:", len(gen))
print("Raw Plant 1 sensor rows:", len(sensor))


# --------------------------------------------------
# 2. Inspect DATE_TIME before conversion
# --------------------------------------------------

print("\nGeneration DATE_TIME examples:")
print(gen["DATE_TIME"].head(10))

print("\nSensor DATE_TIME examples:")
print(sensor["DATE_TIME"].head(10))


# --------------------------------------------------
# 3. Convert DATE_TIME to datetime
# --------------------------------------------------
# These formats must match the raw timestamps
# printed above.

gen["datetime"] = pd.to_datetime(
    gen["DATE_TIME"],
    format="%d-%m-%Y %H:%M"
)

sensor["datetime"] = pd.to_datetime(
    sensor["DATE_TIME"],
    format="%Y-%m-%d %H:%M:%S"
)


# Verify the conversion
print("\nConverted generation timestamps:")
print(gen[["DATE_TIME", "datetime"]].head())

print("\nConverted sensor timestamps:")
print(sensor[["DATE_TIME", "datetime"]].head())


# First and last timestamps
print("\nGeneration first timestamp:", gen["datetime"].min())
print("Generation last timestamp :", gen["datetime"].max())

print("Sensor first timestamp    :", sensor["datetime"].min())
print("Sensor last timestamp     :", sensor["datetime"].max())


# --------------------------------------------------
# 4. Sum power from all inverters at each timestamp
# --------------------------------------------------

plant_power = (
    gen.groupby("datetime")[["AC_POWER", "DC_POWER"]]
    .sum()
    .reset_index()
)

print("\nPlant-level generation:")
print(plant_power.head())


# --------------------------------------------------
# 5. Keep the required sensor columns
# --------------------------------------------------

sensor_data = sensor[
    [
        "datetime",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION"
    ]
].copy()


# --------------------------------------------------
# 6. Check timestamps that exist in only one file
# --------------------------------------------------

timestamp_check = plant_power.merge(
    sensor_data,
    on="datetime",
    how="outer",
    indicator=True
)

generation_only = (
    timestamp_check["_merge"] == "left_only"
).sum()

sensor_only = (
    timestamp_check["_merge"] == "right_only"
).sum()

print("\nTimestamp check:")
print("Generation only:", generation_only)
print("Sensor only:", sensor_only)
print(
    "Total timestamps in only one file:",
    generation_only + sensor_only
)


# --------------------------------------------------
# 7. Merge generation and sensor data
# --------------------------------------------------

merged = plant_power.merge(
    sensor_data,
    on="datetime",
    how="inner"
)

print("\nMerged 15-minute rows:", len(merged))

print("\nMerged data sample:")
print(merged.head())


# --------------------------------------------------
# 8. Resample to hourly means
# --------------------------------------------------

merged = merged.set_index("datetime")

hourly = merged.resample("1h").mean()

hourly = hourly.reset_index()


# --------------------------------------------------
# 9. Rename the columns
# --------------------------------------------------

hourly = hourly.rename(
    columns={
        "AC_POWER": "ac_power",
        "DC_POWER": "dc_power",
        "AMBIENT_TEMPERATURE": "ambient_temp",
        "MODULE_TEMPERATURE": "module_temp",
        "IRRADIATION": "irradiation"
    }
)


# --------------------------------------------------
# 10. Check hourly data
# --------------------------------------------------

print("\nHourly data:")
print(hourly.head())

print("\nNumber of hourly rows:", len(hourly))


# --------------------------------------------------
# 11. Check missing values
# --------------------------------------------------

print("\nMissing values in each column:")
print(hourly.isna().sum())

missing_rows = hourly.isna().any(axis=1).sum()

print("\nRows containing missing values:", missing_rows)


# Display incomplete rows
if missing_rows > 0:
    print("\nRows with missing values:")
    print(
        hourly[
            hourly.isna().any(axis=1)
        ]
    )


# --------------------------------------------------
# 12. Handle missing values
# --------------------------------------------------
# If the missing-row count is small, incomplete rows
# are removed.

hourly_clean = hourly.dropna().copy()

removed_rows = len(hourly) - len(hourly_clean)

print("\nRows removed:", removed_rows)
print("Final usable hourly rows:", len(hourly_clean))


# --------------------------------------------------
# 13. Save the prepared hourly dataset
# --------------------------------------------------

hourly_clean.to_csv(
    "data/plant1_hourly.csv",
    index=False
)

print("\nSaved: data/plant1_hourly.csv")