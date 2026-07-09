import pandas as pd
from datetime import date, datetime
from dotenv import load_dotenv
import os

from pyprojroot import here

import pandas as pd
from sqlalchemy import create_engine, text

from mmm_utils.holidays import create_holiday_columns
from mmm_utils.meteo import create_national_temperature_columns

load_dotenv()


def compute_spend_distribution(df, date_column="date", significance_threshold=0.05):
    summary = (
        df.groupby("media", dropna=False)["budget"]
        .sum()
        .reset_index()
        .rename(columns={"media": "column", "budget": "depenses"})
    )

    total_depenses = summary["depenses"].sum()
    summary["proportion"] = (
        0 if total_depenses == 0 else summary["depenses"] / total_depenses
    )

    def _make_decision(x):
        if x >= significance_threshold:
            return "✅"
        elif x < significance_threshold / 2:
            return "❌"
        else:
            return "⚠️"

    summary["decision"] = summary["proportion"].apply(_make_decision)

    out = (
        summary.sort_values("depenses", ascending=False)
        .reset_index(drop=True)
        .round({"proportion": 4})
    )
    print(out)
    print()
    return out


tmp = 0


def delete_tmp_files():
    i = 0
    while True:
        tmp_file = here() / "data" / f"tmp_building_mm_{i}.csv"
        if not os.path.exists(tmp_file):
            break
        os.remove(tmp_file)
        i += 1


def df_to_tmp_csv(df):
    global tmp
    df.to_csv(here() / "data" / f"tmp_building_mm_{tmp}.csv", index=False)
    tmp += 1


delete_tmp_files()

# —————————————————————————————————————————————————————————————————————————————
# Loading the media mix and ca data from SQL Server
# —————————————————————————————————————————————————————————————————————————————
# Connexion SQL Server
MSSQL_HOST = os.getenv("MSSQL_HOST")
MSSQL_PORT = int(os.getenv("MSSQL_PORT"))
MSSQL_DB = os.getenv("MSSQL_DB")
MSSQL_USER = os.getenv("MSSQL_USER")
MSSQL_PWD = os.getenv("MSSQL_PWD")

url = f"mssql+pymssql://{MSSQL_USER}:{MSSQL_PWD}@{MSSQL_HOST}:{MSSQL_PORT}/{MSSQL_DB}"
engine = create_engine(url)

with engine.connect() as conn:
    mm = pd.read_sql(
        text("SELECT * FROM Datawarehouse.dbo.VUE_MMM_JARDILAND_AE_MIXMEDIA"), conn
    )
    ca = pd.read_sql(text("SELECT * FROM Datawarehouse.dbo.VUE_MMM_JARDILAND_CA"), conn)

mm = (
    mm.rename(columns={"week_begin": "date"})
    .assign(date=lambda x: pd.to_datetime(x["date"]))
    .sort_values("date")
    .drop(columns=["datasource"])
)

ca = (
    ca.rename(columns={"week_begin": "date", "ca": "CA"})
    .assign(date=lambda x: pd.to_datetime(x["date"]))
    .sort_values("date")
    .drop(columns=["datasource"])
)

df_to_tmp_csv(mm)
df_to_tmp_csv(ca)

mm = mm.merge(ca, on="date", how="left")
df_to_tmp_csv(mm)


def mutate(df, column, old_value, new_value):
    df.loc[df[column] == old_value, column] = new_value
    return df


def mutate_if(df, column, new_value, if_column, if_value):
    df.loc[df[if_column] == if_value, column] = new_value
    return df


mm = (
    # --- Renaming ---
    mm.pipe(mutate, "media", "tv", "TV")
    .pipe(mutate, "media", "radio", "Radio")
    .pipe(mutate, "media", "crm", "CRM")
    .pipe(mutate_if, "media", "Display", "digital_lever", "DISPLAY")
    .pipe(mutate, "media", "dooh - digital out of home", "DOOH")
    .pipe(mutate, "media", "cataloguebal", "Cataloguebal")
    .pipe(mutate, "media", "presse", "Presse")
    .pipe(mutate_if, "media", "E-CATALOG", "digital_lever", "E-CATALOG")
    .pipe(mutate_if, "media", "SMS", "digital_lever", "SMS")
    # --- Digital Definition ---
    .pipe(mutate_if, "media", "Video", "digital_lever", "VIDEO")
    .pipe(mutate_if, "media", "Audio", "digital_lever", "AUDIO")
    .pipe(mutate_if, "media", "Social", "digital_lever", "SOCIAL")
    .pipe(mutate_if, "media", "SEA", "digital_lever", "SEA")
)

df_to_tmp_csv(mm)
compute_spend_distribution(mm)
# —————————————————————————————————————————————————————————————————————————————
# Agregating or removing media
# —————————————————————————————————————————————————————————————————————————————

# --- Agregating Offline ---
mm = (
    mm.pipe(mutate_if, "media", "Offline", "media", "Cataloguebal")
    .pipe(mutate_if, "media", "Offline", "media", "Presse")
    .pipe(mutate_if, "media", "Offline", "media", "E-CATALOG")
)
# --- Agregating Video & Audio ---
mm = mm.pipe(mutate_if, "media", "Video & Audio", "media", "Video").pipe(
    mutate_if, "media", "Video & Audio", "media", "Audio"
)
# --- Removing Dooh, affichage, sms ---
mm = mm[~mm["media"].isin(["DOOH", "affichage", "SMS"])]


df_to_tmp_csv(mm)
compute_spend_distribution(mm)

# —————————————————————————————————————————————————————————————————————————————
mm_2024_2026 = mm.pivot_table(
    index=["date", "CA"],
    columns="media",
    values="budget",
    aggfunc="sum",
    fill_value=0,
).reset_index()


# print(
#     (
#         mm_2024_2026[
#             (mm_2024_2026["date"] >= pd.to_datetime("2026-04-01"))  # 1er avril
#             & (mm_2024_2026["date"] <= pd.to_datetime("2026-04-30"))  # 30 mai
#         ]
#         .drop(columns="date")
#         .sum()
#         .reset_index()
#         .rename(columns={"index": "column", 0: "somme"})
#     )
# )


# Add 20 empty weeks
last_date = mm_2024_2026["date"].max()
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=7), periods=20, freq="7D"
)
empty_weeks = pd.DataFrame({"date": future_dates})

for col in mm_2024_2026.columns:
    if col != "date":
        empty_weeks[col] = 0

mm_2024_2026 = pd.concat([mm_2024_2026, empty_weeks], ignore_index=True)

# Remove future dates
mm_2024_2026 = mm_2024_2026[mm_2024_2026["date"] <= pd.Timestamp.now()]


mm_2024_2026 = create_holiday_columns(mm_2024_2026)
mm_2024_2026 = create_national_temperature_columns(mm_2024_2026)

mm_2024_2026.to_csv(
    here() / "data" / "jardiland_MEDIA_MIX_2024_2026.csv",
    index=False,
    sep=";",
    decimal=".",
)


# Remove tmp files
delete = input("Do you want to remove temporary files? (any/no): ") != "no"
if delete:
    delete_tmp_files()
