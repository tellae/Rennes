import data.hts.hts as hts

"""
This stage filters out EMC² observations
"""

def configure(context):
    context.stage("data.hts.emc2.cleaned")
    context.stage("data.spatial.codes")

def execute(context):
    df_codes = context.stage("data.spatial.codes")

    df_households, df_persons, df_trips = context.stage("data.hts.emc2.cleaned")

    len_ini = int(df_trips['trip_weight'].sum())

    # Filter for non-residents
    requested_departments = df_codes["departement_id"].unique()
    f = df_persons["departement_id"].astype(str).isin(requested_departments)  # pandas bug!
    df_persons = df_persons[f]
    df_trips = df_trips[df_trips["person_id"].isin(df_persons["person_id"].unique())]

    len_non_res = int(df_trips['trip_weight'].sum())

    # Filter for people going outside of the area (because they have NaN distances)
    remove_ids = set()

    remove_ids |= set(df_trips[~df_trips["origin_departement_id"].astype(str).isin(requested_departments) | ~df_trips["destination_departement_id"].astype(str).isin(requested_departments)]["person_id"].unique())

    # remove_ids |= set(df_persons[~df_persons["departement_id"].isin(requested_departments)])

    df = df_trips.groupby(['origin_departement_id', 'destination_departement_id'])['trip_weight'].agg('sum').reset_index(name='trip_weight')
    df['trip_weight'].fillna(0, inplace=True)
    df['trip_weight'] = df['trip_weight'].map(lambda x: int(x))
    df.to_csv('departement_matrix_od.csv')

    df_trips = df_trips[~df_trips["person_id"].isin(remove_ids)]

    len_out = int(df_trips['trip_weight'].sum())

    print('INITIAL NUMBER OF TRIPS', len_ini)
    print('FILTER FOR NON-RESIDENTS: MINUS', len_ini - len_non_res, round(100 * (len_ini - len_non_res) / len_ini, 2), '%')
    print('FILTER FOR OUTGOING TRIPS: MINUS', len_non_res - len_out, round(100 * (len_non_res - len_out) / len_ini, 2), '%')
    print('INPUT NUMBER OF TRIPS', len_out)

    df_persons = df_persons[~df_persons["person_id"].isin(remove_ids)]

    # Only keep trips and households that still have a person
    df_trips = df_trips[df_trips["person_id"].isin(df_persons["person_id"].unique())]
    df_households = df_households[df_households["household_id"].isin(df_persons["household_id"])]

    # Finish up
    df_households = df_households[hts.HOUSEHOLD_COLUMNS + ["MID"]]
    df_persons = df_persons[hts.PERSON_COLUMNS + ["MID", "PER"]]
    df_trips = df_trips[hts.TRIP_COLUMNS + ["euclidean_distance"] + ["MID", "PER", "NDEP"]]

    hts.check(df_households, df_persons, df_trips)
    print('NUMBER OF TRIPS', df_trips['trip_weight'].sum())

    df_trips.to_csv('trips_emc².csv', index=False)
    df_households.to_csv('households_emc².csv', index=False)
    df_persons.to_csv('persons_emc².csv', index=False)
    print('========= SAVED =========')
    return df_households, df_persons, df_trips
