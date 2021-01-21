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
    "walk": [],
    # "other": [?]
}

ZONES_MAP = {10100: 35238, 10200: 35238, 10300: 35238, 10400: 35238, 10500: 35238, 10600: 35238, 10700: 35238,
             10800: 35238, 20100: 35238, 20200: 35238, 20300: 35238, 20400: 35238, 20500: 35238, 20600: 35238,
             20700: 35238, 20800: 35238, 20900: 35238, 21000: 35238, 21100: 35238, 30100: 35238, 30200: 35238,
             30300: 35238, 30400: 35238, 30500: 35238, 40100: 35238, 40200: 35238, 40300: 35238, 40400: 35238,
             40500: 35238, 40600: 35238, 40700: 35238, 50100: 35238, 50200: 35238, 50300: 35238, 50400: 35238,
             50500: 35238, 50600: 35238, 50700: 35238, 50800: 35238, 60100: 35238, 60200: 35238, 60300: 35238,
             60400: 35238, 70100: 35238, 70200: 35238, 70300: 35238, 70400: 35238, 80100: 35238, 80200: 35238,
             80300: 35238, 80400: 35238, 80500: 35238, 80600: 35238, 80700: 35238, 80800: 35238, 80900: 35238,
             81000: 35238, 81100: 35238, 81200: 35238, 81300: 35238, 90100: 35238, 90200: 35238, 90300: 35238,
             90400: 35238, 90500: 35238, 90600: 35238, 90700: 35238, 90800: 35238, 100100: 35238, 100200: 35238,
             100300: 35238, 100400: 35238, 100500: 35238, 100600: 35238, 100700: 35238, 100800: 35238, 100900: 35238,
             101000: 35238, 110100: 35238, 110200: 35238, 110300: 35238, 110400: 35238, 110500: 35238, 110600: 35238,
             110700: 35238, 110800: 35238, 120100: 35238, 120200: 35238, 120300: 35238, 120400: 35238, 120500: 35238,
             120600: 35238, 210100: 35278, 210200: 35278, 210300: 35278, 220100: 35024, 220200: 35024, 220300: 35024,
             220400: 35024, 230100: 35001, 230200: 35001, 230300: 35039, 230400: 35334, 230500: 35334, 230600: 35334,
             240100: 35051, 240200: 35051, 240300: 35051, 240400: 35051, 240500: 35051, 240600: 35051, 240700: 35051,
             240800: 35051, 240900: 35051, 250100: 35055, 250200: 35055, 250300: 35055, 260100: 35352, 260200: 35352,
             260300: 35352, 270100: 35088, 270200: 35204, 270300: 35250, 280100: 35032, 280200: 35206, 280300: 35206,
             280400: 35266, 290100: 35139, 290200: 35208, 290300: 35363, 300100: 35066, 300200: 35066, 300300: 35066,
             300400: 35066, 310100: 35047, 310200: 35047, 310300: 35047, 310400: 35047, 310500: 35047, 310600: 35047,
             320100: 35281, 320200: 35281, 320300: 35281, 330100: 35076, 330200: 35080, 330300: 35351, 340100: 35240,
             340200: 35240, 340300: 35240, 350100: 35196, 350200: 35196, 350300: 35196, 360100: 35022, 360200: 35058,
             360300: 35065, 360400: 35081, 360500: 35120, 360600: 35131, 360700: 35144, 360800: 35180, 360900: 35216,
             361000: 35245, 361100: 35275, 361200: 35353, 370100: 35210, 370200: 35210, 370300: 35210, 370400: 35210,
             380100: 35059, 380200: 35079, 380300: 35189, 380400: 35315, 410100: 35173, 410200: 35173, 420100: 35128,
             420200: 35146, 420300: 35177, 420400: 35193, 420500: 35274, 420600: 35276, 420700: 35296, 420800: 35317,
             420900: 35356, 430100: 35003, 430200: 35007, 430300: 35110, 430400: 35118, 430500: 35195, 430600: 35197,
             430700: 35251, 430800: 35326, 430900: 35355, 440100: 35031, 440200: 35067, 440300: 35101, 440400: 35107,
             450100: 35152, 450200: 35152, 450300: 35152, 460100: 35015, 460200: 35038, 460300: 35061, 460400: 35072,
             460500: 35105, 460600: 35141, 460700: 35170, 460800: 35183, 460900: 35185, 461000: 35192, 461100: 35194,
             461200: 35217, 461300: 35232, 461400: 35260, 461500: 35300, 461600: 35330, 461700: 35347, 470100: 35360,
             470200: 35360, 470300: 35360, 470400: 35360, 470500: 35360, 470600: 35360, 470700: 35360, 480100: 35006,
             480200: 35008, 480300: 35014, 480400: 35042, 480500: 35097, 480600: 35102, 480700: 35119, 480800: 35125,
             480900: 35198, 481000: 35199, 481100: 35200, 481200: 35235, 481300: 35272, 481400: 35325, 481500: 35359,
             490100: 35052, 490200: 35068, 490300: 35087, 490400: 35096, 490500: 35109, 490600: 35161, 490700: 35166,
             490800: 35229, 490900: 35252, 491000: 35264, 491100: 35283, 491200: 35338, 491300: 35350, 500100: 35053,
             500200: 35099, 500300: 35207, 500400: 35209, 500500: 35220, 500600: 35254, 500700: 35327, 510100: 35069,
             510200: 35069, 520100: 35002, 520200: 35005, 520300: 35028, 520400: 35041, 520500: 35077, 520600: 35082,
             520700: 35103, 520800: 35108, 520900: 35114, 521000: 35136, 521100: 35136, 521200: 35165, 521300: 35167,
             521400: 35239, 521500: 35262, 521600: 35333, 521700: 35335, 530100: 35012, 530200: 35012, 530300: 35030,
             530400: 35054, 530500: 35089, 530600: 35090, 530700: 35098, 530800: 35106, 530900: 35124, 531000: 35140,
             531100: 35202, 531200: 35212, 531300: 35218, 531400: 35221, 531500: 35231, 531600: 35249, 531700: 35316,
             531800: 35321, 531900: 35322, 532000: 35332, 532100: 35343, 540100: 35236, 540200: 35236, 540300: 35236,
             540400: 35236, 550100: 35013, 550200: 35045, 550300: 35064, 550400: 35145, 550500: 35151, 550600: 35219,
             550700: 35237, 550800: 35268, 550900: 35285, 551000: 35294, 551100: 35328, 560100: 35035, 560200: 35046,
             560300: 35048, 560400: 35057, 560500: 35084, 560600: 35129, 560700: 35155, 560800: 35160, 560900: 35168,
             561000: 35175, 561100: 35176, 561200: 35289, 561300: 35311, 570100: 35016, 570200: 35033, 570300: 35123,
             570400: 35126, 570500: 35126, 570600: 35126, 570700: 35127, 570800: 35149, 570900: 35312, 580100: 35037,
             580200: 35169, 580300: 35187, 580400: 35211, 580500: 35223, 580600: 35305, 580700: 35319, 580800: 35340,
             590100: 35023, 590200: 35040, 590300: 35133, 590400: 35188, 590500: 35188, 590600: 35188, 590700: 35203,
             590800: 35227, 590900: 35277, 591000: 35331, 600100: 35026, 600200: 35027, 600300: 35060, 600400: 35091,
             600500: 35117, 600600: 35135, 600700: 35143, 600800: 35158, 600900: 35171, 601000: 35184, 601100: 35201,
             601200: 35234, 601300: 35290, 601400: 35295, 601500: 35297, 601600: 35301, 601700: 35302, 601800: 35307,
             601900: 35320, 610100: 35017, 610200: 35050, 610300: 35130, 610400: 35134, 610500: 35156, 610600: 35233,
             610700: 35258, 610800: 35265, 610900: 35318, 611000: 35337, 620100: 35029, 620200: 35056, 620300: 35085,
             620400: 35092, 620500: 35094, 620600: 35147, 620700: 35148, 620800: 35159, 620900: 35172, 621000: 35225,
             621100: 35226, 621200: 35286, 621300: 35308, 621400: 35342, 621500: 35344, 621600: 35345, 621700: 35346,
             630100: 35093, 630200: 35093, 630300: 35093, 630400: 35093, 630500: 35093, 630600: 35181, 630700: 35228,
             630800: 35241, 630900: 35256, 631000: 35287, 640100: 35288, 640200: 35288, 640300: 35288, 640400: 35288,
             640500: 35288, 640600: 35288, 640700: 35288, 640800: 35288, 640900: 35288, 650100: 35288, 650200: 35288,
             650300: 35288, 650400: 35288, 650500: 35288, 650600: 35288, 650700: 35288, 650800: 35288, 650900: 35288,
             651000: 35288, 651100: 35288, 651200: 35288, 660100: 35070, 660200: 35179, 660300: 35224, 660400: 35284,
             660500: 35306, 660600: 35314, 660700: 35358, 660800: 35362, 670100: 35049, 670200: 35049, 670300: 35116,
             670400: 35122, 670500: 35132, 670600: 35153, 670700: 35255, 670800: 35263, 670900: 35279, 671000: 35299,
             680100: 35009, 680200: 35010, 680300: 35034, 680400: 35044, 680500: 35078, 680600: 35095, 680700: 35104,
             680800: 35186, 680900: 35222, 681000: 35246, 681100: 35247, 681200: 35248, 681300: 35259, 681400: 35270,
             681500: 35291, 681600: 35329, 681700: 35339, 681800: 35354, 681900: 35361, 690100: 35004, 690200: 35019,
             690300: 35075, 690400: 35113, 690500: 35164, 690600: 35205, 690700: 35242, 690800: 35244, 690900: 35303,
             691000: 35309, 691100: 35341, 700100: 35011, 700200: 35071, 700300: 35083, 700400: 35191, 700500: 35257,
             700600: 35267, 700700: 35273, 700800: 35280, 700900: 35292, 701000: 35323, 701100: 35336, 710100: 35111,
             710200: 35162, 710300: 35174, 710400: 35190, 710500: 35230, 710600: 35271, 710700: 35357, 720100: 35018,
             720200: 35025, 720300: 35062, 720400: 35086, 720500: 35100, 720600: 35112, 720700: 35138, 720800: 35142,
             720900: 35157, 721000: 35163, 721100: 35214, 721200: 35215, 730100: 35021, 730200: 35115, 730300: 35115,
             730400: 35115, 730500: 35115, 730600: 35115, 730700: 35115, 730800: 35115, 730900: 35115, 731000: 35115,
             731100: 35115, 731200: 35137, 731300: 35150, 731400: 35324, 740100: 35063, 740200: 35243, 740300: 35261,
             740400: 35269, 740500: 35282, 740600: 35293, 740700: 35304, 740800: 35310, 740900: 35348, 750100: 35121,
             750200: 35154, 750300: 35178, 750400: 35253, 810100: 22049, 810200: 22094, 810300: 22103, 810400: 22105,
             810500: 22190, 810600: 22192, 810700: 22209, 810800: 22213, 810900: 22327, 811000: 22357, 811100: 22368,
             820100: 22026, 820200: 22035, 820300: 22050, 820400: 22050, 820500: 22050, 820600: 22050, 820700: 22050,
             820800: 22050, 820900: 22056, 821000: 22118, 821100: 22123, 821200: 22197, 821300: 22208, 821400: 22259,
             821500: 22263, 821600: 22274, 821700: 22280, 821800: 22299, 821900: 22306, 822000: 22339, 822100: 22352,
             822200: 22364, 822300: 22385, 830100: 56001, 830200: 56011, 830300: 56060, 830400: 56154, 830500: 56194,
             830600: 56216, 830700: 56221, 830800: 56223, 830900: 56232, 831000: 56239, 831100: 56250}


def execute(context):
    df_households, df_persons, df_trips = context.stage("data.hts.emc2.raw")  # , df_paths

    # Make copies
    df_households = pd.DataFrame(df_households, copy=True)
    df_persons = pd.DataFrame(df_persons, copy=True)
    df_trips = pd.DataFrame(df_trips, copy=True)
    # df_paths = pd.DataFrame(df_paths, copy = True)

    # Construct new IDs for households, persons and trips (which are unique globally)
    df_households["household_id"] = np.arange(len(df_households))

    df_persons = pd.merge(
        df_persons, df_households[["MID", "household_id"]],
        on="MID"
    )
    df_persons["person_id"] = np.arange(len(df_persons))

    df_trips = pd.merge(
        df_trips, df_persons[["PER", "MID", "person_id", "household_id"]],
        on=["PER", "MID"]
    )
    df_trips["trip_id"] = np.arange(len(df_trips))

    # df_paths = pd.merge(
    #     df_paths, df_trips[["DID", "PID", "MID", "trip_id", "person_id", "household_id"]],
    #     on = ["DID", "PID", "MID"]
    # )
    # df_paths["path_id"] = np.arange(len(df_paths))

    # Trip flags
    df_trips = hts.compute_first_last(df_trips)

    # Weight
    df_households["household_weight"] = df_households["COEM"]
    df_persons["person_weight"] = df_persons["COEP"]


    # Clean age
    df_persons["age"] = df_persons["P4"].astype(np.int)

    # Clean sex
    df_persons.loc[df_persons["P2"] == 1, "sex"] = "male"
    df_persons.loc[df_persons["P2"] == 2, "sex"] = "female"
    df_persons["sex"] = df_persons["sex"].astype("category")

    # Household size
    df_households["household_size"] = df_persons.groupby("household_id").size()

    # Clean commune
    df_persons["commune_id"] = df_persons.replace({"ZFP": ZONES_MAP})["ZFP"].astype("category")
    df_households["commune_id"] = df_households.replace({"ZFM": ZONES_MAP})["ZFM"].astype("category")
    df_trips["origin_commune_id"] = df_trips.replace({"D3": ZONES_MAP})["D3"].astype("category")
    df_trips["destination_commune_id"] = df_trips.replace({"D7": ZONES_MAP})["D7"].astype("category")

    # Clean departement
    df_persons["departement_id"] = (df_persons["commune_id"].astype(int) / 1000).astype(int).astype(str).astype(
        "category")
    df_households["departement_id"] = (df_households["commune_id"].astype(int) / 1000).astype(int).astype(str).astype(
        "category")
    df_trips["origin_departement_id"] = (df_trips["origin_commune_id"].astype(int) / 1000).astype(int).astype(
        str).astype("category")
    df_trips["destination_departement_id"] = (df_trips["destination_commune_id"].astype(int) / 1000).astype(int).astype(
        str).astype("category")

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

    # Household income
    # df_households["income_class"] = df_households["REVENU"] - 1
    # df_households.loc[df_households["income_class"].isin([10.0, 11.0, np.nan]), "income_class"] = -1
    # df_households["income_class"] = df_households["income_class"].astype(np.int)

    df_households["income_class"] = 0 # np.random.randint(0, 10, size=(len(df_households), 1)).astype(int)

    # Trip purpose
    df_trips["following_purpose"] = "other"
    df_trips["preceding_purpose"] = "other"

    for purpose, category in PURPOSE_MAP.items():
        df_trips.loc[df_trips["D2A"].isin(category), "following_purpose"] = purpose
        df_trips.loc[df_trips["D5A"].isin(category), "preceding_purpose"] = purpose

    df_trips["following_purpose"] = df_trips["following_purpose"].astype("category")
    df_trips["preceding_purpose"] = df_trips["preceding_purpose"].astype("category")

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
    df_trips = pd.merge(
        df_trips, df_persons[["person_id", "person_weight"]], on="person_id", how="left"
    ).rename(columns={"person_weight": "trip_weight"})
    df_persons["trip_weight"] = df_persons["person_weight"]

    # Chain length
    df_persons["number_of_trips"] = df_trips.groupby("person_id").size()

    # Passenger attribute
    df_persons["is_passenger"] = df_persons["person_id"].isin(
        df_trips[df_trips["mode"] == "car_passenger"]["person_id"].unique()
    )

    # Calculate consumption units
    hts.check_household_size(df_households, df_persons)
    df_households = pd.merge(df_households, hts.calculate_consumption_units(df_persons), on="household_id")

    # Socioprofessional class
    df_persons["socioprofessional_class"] = df_persons["P11"].fillna(8).astype(int)

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
    df_trips.to_csv('trips_emc².csv', index=False)
    df_households.to_csv('households_emc².csv', index=False)
    df_persons.to_csv('persons_emc².csv', index=False)

    return df_households, df_persons, df_trips


def calculate_income_class(df):
    assert "household_income" in df
    assert "consumption_units" in df

    return np.digitize(df["household_income"] / df["consumption_units"], INCOME_CLASS_BOUNDS, right=True)
