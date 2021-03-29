# from tqdm import tqdm
import pandas as pd
import numpy as np
import data.hts.hts as hts

"""
This stage cleans the regional HTS.
"""


def configure(context):
    context.stage("data.hts.emc2.raw")


INCOME_CLASS_BOUNDS = [800, 1200, 1600, 2000, 2400, 3000, 3500, 4500, 5500, 1e6]

PURPOSE_MAP = {
    "home": [1, 2],
    "work": [11, 12, 13] + [81],
    "leisure": [51, 52, 53, 54],
    "education": [21, 22, 23, 24, 25, 26, 27, 28, 29],
    "shop": [30, 31, 32, 33, 34, 35] + [44] + [82],
    "other": [41, 42, 43, 61, 62, 63, 64, 71, 72, 73, 74] + [91]
}

MODES_MAP = {
    "pt": [31, 33, 38, 39, 41, 42, 51, 91, 92],
    "car": [13, 15, 21, 71, 81],
    "car_passenger": [14, 16, 22, 61, 82],
    "bike": [10, 11, 12, 93, 94, 95],
    "walk": []
}

df_zone = pd.read_csv('ZONE_MAP.csv')
df_zone['ZF'] = df_zone['ZF'] // 100

def execute(context):
    df_households, df_persons, df_trips = context.stage("data.hts.emc2.raw")

    # Make copies
    df_households = pd.DataFrame(df_households, copy=True)
    df_persons = pd.DataFrame(df_persons, copy=True)
    df_trips = pd.DataFrame(df_trips, copy=True)

    df_households['ZFM'] = df_households['ZFM'].map(lambda x: int(x)//100)
    df_persons['ZFP'] = df_persons['ZFP'].map(lambda x: int(x)//100)
    df_trips['ZFD'] = df_trips['ZFD'].map(lambda x: int(x)//100)
    df_trips['D3'] = df_trips['D3'].map(lambda x: int(x//100))
    df_trips['D7'] = df_trips['D7'].map(lambda x: int(x//100))

    # Construct new IDs for households, persons and trips (which are unique globally)
    df_households["household_id"] = np.arange(len(df_households))

    df_persons = pd.merge(df_persons, df_households[["MID", "household_id"]], on="MID")
    df_persons["person_id"] = np.arange(len(df_persons))

    df_trips = pd.merge(df_trips, df_persons[["PER", "MID", "person_id", "household_id"]], on=["PER", "MID"])
    df_trips["trip_id"] = np.arange(len(df_trips))

    # Trip flags
    df_trips = hts.compute_first_last(df_trips)

    # Weight
    df_households["household_weight"] = df_households["COEM"]
    df_persons["person_weight"] = df_persons["COE2"]

    # Clean age
    df_persons["age"] = df_persons["P4"].astype(np.int)

    # Clean sex
    df_persons.loc[df_persons["P2"] == 1, "sex"] = "male"
    df_persons.loc[df_persons["P2"] == 2, "sex"] = "female"
    df_persons["sex"] = df_persons["sex"].astype("category")

    # Household size
    df_households["household_size"] = df_persons.groupby("household_id").size()

    # Clean commune
    df_persons = df_persons.merge(df_zone, left_on='ZFP', right_on='ZF', how='left').rename(columns = {'INSEE': 'commune_id'})
    df_households = df_households.merge(df_zone, left_on='ZFM', right_on='ZF', how='left').rename(columns = {'INSEE': 'commune_id'})

    df_persons['commune_id'].fillna(0, inplace=True)
    df_households['commune_id'].fillna(0, inplace=True)

    df_trips = df_trips.merge(df_zone, left_on='D3', right_on='ZF', how='left').rename(columns = {'INSEE': 'origin_commune_id'})
    df_trips = df_trips.merge(df_zone, left_on='D7', right_on='ZF', how='left').rename(columns = {'INSEE': 'destination_commune_id'})

    df_trips['origin_commune_id'].fillna(0, inplace=True)
    df_trips['destination_commune_id'].fillna(0, inplace=True)

    df_persons["commune_id"] = df_persons["commune_id"].astype("category")
    df_households["commune_id"] = df_households["commune_id"].astype("category")
    df_trips["origin_commune_id"] = df_trips["origin_commune_id"].astype("category")
    df_trips["destination_commune_id"] = df_trips["destination_commune_id"].astype("category")

    # Clean departement
    df_persons["departement_id"] = (df_persons["commune_id"].astype(int) / 1000).astype(int).astype(str).astype("category")
    df_households["departement_id"] = (df_households["commune_id"].astype(int) / 1000).astype(int).astype(str).astype("category")
    df_trips["origin_departement_id"] = (df_trips["origin_commune_id"].astype(int) / 1000).astype(int).astype(str).astype("category")
    df_trips["destination_departement_id"] = (df_trips["destination_commune_id"].astype(int) / 1000).astype(int).astype(str).astype("category")

    # Clean employment
    df_persons["employed"] = df_persons["P9"].astype('Float32').isin([1.0, 2.0])

    # Studies
    df_persons["studies"] = df_persons["P9"].astype('Float32').isin([3.0, 4.0, 5.0])

    # Number of vehicles
    df_households["number_of_vehicles"] = df_households["M6"] + df_households["M14"]
    df_households["number_of_vehicles"] = df_households["number_of_vehicles"].astype(np.int)
    df_households["number_of_bikes"] = df_households["M21"].astype(np.int)

    # License
    df_persons["has_license"] = (df_persons["P7"] == 1)

    # Has subscription
    df_persons["has_pt_subscription"] = (df_persons["P12"] != 4)

    # # Household income
    # df_households["income_class"] = df_households["REVENU"] - 1
    # df_households.loc[df_households["income_class"].isin([10.0, 11.0, np.nan]), "income_class"] = -1
    # df_households["income_class"] = df_households["income_class"].astype(np.int)

    df_households["income_class"] = 0  # np.random.randint(0, 10, size=(len(df_households), 1)).astype(int)

    # Trip purpose
    df_trips["following_purpose"] = "other"
    df_trips["preceding_purpose"] = "other"

    for purpose, category in PURPOSE_MAP.items():
        df_trips.loc[df_trips["D2A"].isin(category), "preceding_purpose"] = purpose
        df_trips.loc[df_trips["D5A"].isin(category), "following_purpose"] = purpose

    df_trips["preceding_purpose"] = df_trips["preceding_purpose"].astype("category")
    df_trips["following_purpose"] = df_trips["following_purpose"].astype("category")

    # Trip mode
    df_trips["mode"] = "walk"

    for mode, category in MODES_MAP.items():
        df_trips.loc[df_trips["MODP"].isin(category), "mode"] = mode

    df_trips["mode"] = df_trips["mode"].astype("category")

    # Further trip attributes
    df_trips["euclidean_distance"] = df_trips["DOIB"]

    # Trip times
    df_trips["departure_time"] = df_trips["D4A"] * 3600 + df_trips["D4B"] * 60
    df_trips["arrival_time"] = df_trips["D8A"] * 3600 + df_trips["D8B"] * 60
    df_trips = hts.fix_trip_times(df_trips)

    # Durations
    df_trips["trip_duration"] = df_trips["arrival_time"] - df_trips["departure_time"]
    hts.compute_activity_duration(df_trips)

    # Add weight to trips
    df_trips = pd.merge(df_trips, df_persons[["person_id", "person_weight"]], on="person_id", how="left").rename(columns={"person_weight": "trip_weight"})
    df_persons["trip_weight"] = df_persons["person_weight"]

    # Chain length
    df_persons["number_of_trips"] = df_trips.groupby("person_id").size()

    # Passenger attribute
    df_persons["is_passenger"] = df_persons["person_id"].isin(df_trips[df_trips["mode"] == "car_passenger"]["person_id"].unique())

    # Calculate consumption units
    hts.check_household_size(df_households, df_persons)
    df_households = pd.merge(df_households, hts.calculate_consumption_units(df_persons), on="household_id")

    # Socioprofessional class
    df_persons["socioprofessional_class"] = df_persons["P11"].fillna(8).astype(int)
    df_persons.loc[df_persons["socioprofessional_class"].isin([7, 8, 9]), "socioprofessional_class"] = 8
    df_persons.loc[df_persons["P9"] == 7, "socioprofessional_class"] = 7

    # Drop people that have NaN departure or arrival times in trips
    # Filter for people with NaN departure or arrival times in trips
    f = df_trips["departure_time"].isna()
    f |= df_trips["arrival_time"].isna()

    f = df_persons["person_id"].isin(df_trips[f]["person_id"])

    nan_count = np.count_nonzero(f)
    total_count = len(df_persons)

    print("Dropping %d/%d persons because of NaN values in departure and arrival times" % (nan_count, total_count))

    df_persons = df_persons[~f]
    df_trips = df_trips[df_trips["person_id"].isin(df_persons["person_id"].unique())]
    df_households = df_households[df_households["household_id"].isin(df_persons["household_id"])]

    # Fix activity types (because of inconsistent EGT data and removing in the timing fixing step)
    hts.fix_activity_types(df_trips)

    return df_households, df_persons, df_trips


def calculate_income_class(df):
    assert "household_income" in df
    assert "consumption_units" in df

    return np.digitize(df["household_income"] / df["consumption_units"], INCOME_CLASS_BOUNDS, right=True)
