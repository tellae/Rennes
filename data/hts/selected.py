import pandas as pd
import numpy as np

def configure(context):
    hts = context.config("hts")

    if hts == "egt":
        context.stage("data.hts.egt.filtered", alias = "hts")
    elif hts == "emc2":
        context.stage("data.hts.emc2.filtered", alias = "hts")
    elif hts == "entd":
        context.stage("data.hts.entd.reweighted", alias = "hts")
    else:
        raise RuntimeError("Unknown HTS: %s" % hts)

def execute(context):
    return context.stage("hts")
