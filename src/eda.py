import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------
# Load the prepared hourly data
# -----------------------------------------

df = pd.read_csv(
    "data/plant1_hourly.csv"
)

df["datetime"] = pd.to_datetime(
    df["datetime"]
)

print("Number of rows:", len(df))

print("\nFirst five rows:")
print(df.head())


# =========================================
# Plot 1: AC Power vs Irradiation
# =========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["irradiation"],
    df["ac_power"],
    alpha=0.5
)

plt.xlabel("Irradiation")
plt.ylabel("AC Power")
plt.title("AC Power vs Irradiation")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/ac_power_vs_irradiation.png"
)

plt.show()


# =========================================
# Plot 2: Module Temperature vs
# Ambient Temperature, colored by irradiation
# =========================================

plt.figure(figsize=(8, 5))

scatter = plt.scatter(
    df["ambient_temp"],
    df["module_temp"],
    c=df["irradiation"],
    alpha=0.6
)

plt.xlabel("Ambient Temperature")
plt.ylabel("Module Temperature")
plt.title(
    "Module Temperature vs Ambient Temperature"
)

plt.colorbar(
    scatter,
    label="Irradiation"
)

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/module_vs_ambient_temp.png"
)

plt.show()


# =========================================
# Plot 3: AC Power vs DC Power
# =========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["dc_power"],
    df["ac_power"],
    alpha=0.5
)

plt.xlabel("DC Power")
plt.ylabel("AC Power")
plt.title("AC Power vs DC Power")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/ac_vs_dc_power.png"
)

plt.show()


# Calculate AC/DC ratio for non-zero DC power

day_power = df[
    df["dc_power"] > 0
].copy()

day_power["ac_dc_ratio"] = (
    day_power["ac_power"]
    / day_power["dc_power"]
)

print("\nAC/DC ratio statistics:")
print(
    day_power["ac_dc_ratio"].describe()
)


# =========================================
# Plot 4: Average AC Power by Hour
# =========================================

df["hour"] = df["datetime"].dt.hour

hourly_profile = (
    df.groupby("hour")["ac_power"]
    .mean()
)

plt.figure(figsize=(8, 5))

plt.plot(
    hourly_profile.index,
    hourly_profile.values,
    marker="o"
)

plt.xlabel("Hour of Day")
plt.ylabel("Average AC Power")
plt.title("Average AC Power by Hour of Day")

plt.xticks(range(0, 24))

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "results/average_ac_power_by_hour.png"
)

plt.show()


print("\nTask 2 completed.")
print("Figures saved in the results folder.")