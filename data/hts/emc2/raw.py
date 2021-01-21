from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import os

"""
This stage loads the raw data of the specified HTS (EMC² for Rennes Metropolis).
"""

MENAGES_COLUMNS = [
    "ZFM", "ECH", "M6", "M14", "M21", "COEM"
]

PERSONNES_COLUMNS = [
    "ZFP", "ECH", "PER", "JOUR", "P2", "P4", "P7", "P8", "P9", "P11", "P12", "COEP", "COE2"
]

DEPLACEMENTS_COLUMNS = [
    "ZFD", "ECH", "PER", "NDEP",
    "D2A", "D2B", "D3", "D4A", "D4B", "D5A", "D5B", "D6", "D7", "D8A", "D8B", "D9",
    "MODP", "MOIP", "DOIB", "DIST", "DISP"
]


def configure(context):
    context.config("data_path")


def execute(context):

    df_menages = pd.read_csv(
        "%s/emc2/03A_EM~1.csv" % context.config("data_path"),
        sep=",", usecols=MENAGES_COLUMNS
    )
    df_menages['MID'] = df_menages['ZFM'] + df_menages['ECH']

    df_personnes = pd.read_csv(
        "%s/emc2/03B_EM~1.csv" % context.config("data_path"),
        sep=",", usecols=PERSONNES_COLUMNS
    )

    df_personnes = pd.read_csv(
        "%s/emc2/03B_EM~1.csv" % context.config("data_path"),
        sep=",", usecols=PERSONNES_COLUMNS
    )
    df_personnes['MID'] = df_personnes['ZFP'] + df_personnes['ECH']

    df_deplacements = pd.read_csv(
        "%s/emc2/03C_EM~1.csv" % context.config("data_path"),
        sep=",", usecols=DEPLACEMENTS_COLUMNS
    )
    df_deplacements['MID'] = df_deplacements['ZFD'] + df_deplacements['ECH']

    return df_menages, df_personnes, df_deplacements


def validate(context):
    for name in ("03A_EM~1.csv", "03B_EM~1.csv", "03C_EM~1.csv"):
        if not os.path.exists("%s/emc2/%s" % (context.config("data_path"), name)):
            raise RuntimeError("File missing from EGT: %s" % name)

    return [
        os.path.getsize("%s/emc2/03A_EM~1.csv" % context.config("data_path")),
        os.path.getsize("%s/emc2/03B_EM~1.csv" % context.config("data_path")),
        os.path.getsize("%s/emc2/03C_EM~1.csv" % context.config("data_path"))
    ]
