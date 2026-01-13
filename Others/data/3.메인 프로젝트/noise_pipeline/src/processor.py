import pandas as pd
from config.settings import TARGET_REGION


COLUMNS = [
    "spotCode",
    "spotName",
    "spotAddr",
    "daytimeAve",
    "nightAve",
    "daytimeAveLv",
    "nightAveLv",
    "wgs84Lat",
    "wgs84Lon",
    "measYear",
    "quarterSect",
]


def filter_region(data: list):
    return [
        item for item in data
        if TARGET_REGION in item.get("spotAddr", "")
    ]


def to_dataframe(data: list):
    df = pd.DataFrame(data)
    return df[COLUMNS]
