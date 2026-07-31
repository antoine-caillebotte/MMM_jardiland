import pandas as pd
import numpy as np
import pandas as pd

from mmm_pipeline import DataCreator, compute_spend_distribution
from mmm_pipeline.holidays import create_holiday_columns

def create_dataset(
    filename: str, db_config, dirpath: str, safe_mode: bool = True
) -> pd.DataFrame:

    with DataCreator(filename, dirpath=dirpath, safe_mode=safe_mode) as mm_2025_2026:
        # —————————————————————————————————————————————————————————————————————————————
        # Loading the media mix and ca data from SQL Server
        # —————————————————————————————————————————————————————————————————————————————
        mm = mm_2025_2026.read_sql(
            db_config, "SELECT * FROM Datawarehouse.dbo.VUE_MMM_JARDILAND_AE_MIXMEDIA"
        )
        ca = mm_2025_2026.read_sql(
            db_config, "SELECT * FROM Datawarehouse.dbo.VUE_MMM_JARDILAND_CA"
        )

        mm = (
            mm.rename(columns={"week_begin": "date"})
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .sort_values("date")
            .drop(columns=["datasource"])
        )

        ca = (
            ca.rename(columns={"week_begin": "date"})
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .sort_values("date")
            .drop(columns=["datasource"])
        )

        mm_2025_2026.dump_to_tmp_csv(mm)
        mm_2025_2026.dump_to_tmp_csv(ca)

        mm_2025_2026.df = mm.merge(ca, on="date", how="left")
        mm_2025_2026.dump_to_tmp_csv()

        mm_2025_2026 = (
            # --- Renaming ---
            mm_2025_2026.rename("media", {"tv": "TV"})
            # .rename("media", {"radio": "Radio"})
            # .rename("media", {"crm": "CRM"})
            .rename_if("media", "Display", "digital_lever", "DISPLAY")
            .rename("media", {"dooh - digital out of home": "DOOH"})
            # .rename("media", {"cataloguebal": "Cataloguebal"})
            # .rename("media", {"presse": "Presse"})
            .rename_if("media", "E-CATALOG", "digital_lever", "E-CATALOG")
            .rename_if("media", "SMS", "digital_lever", "SMS")
            # --- Digital Definition ---
            .rename_if("media", "Video", "digital_lever", "VIDEO")
            .rename_if("media", "Audio", "digital_lever", "AUDIO")
            .rename_if("media", "Social", "digital_lever", "SOCIAL")
            .rename_if("media", "SEA", "digital_lever", "SEA")
        )

        mm_2025_2026.dump_to_tmp_csv()
        # —————————————————————————————————————————————————————————————————————————————
        # Agregating or removing media
        # —————————————————————————————————————————————————————————————————————————————

        # --- Agregating Offline ---
        mm_2025_2026 = (
            mm_2025_2026.rename_if("media", "Offline", "media", "Cataloguebal")
            .rename_if("media", "Offline", "media", "Presse")
            .rename_if("media", "Offline", "media", "E-CATALOG")
        )
        # --- Agregating Video & Audio ---
        mm_2025_2026 = mm_2025_2026.rename_if(
            "media", "Video & Audio", "media", "Video"
        ).rename_if("media", "Video & Audio", "media", "Audio")
        # --- Removing Dooh, affichage, sms ---
        mm_2025_2026.df = mm_2025_2026.df[
            ~mm_2025_2026.df["media"].isin(["DOOH", "affichage", "SMS"])
        ]

        mm_2025_2026.dump_to_tmp_csv()

        # —————————————————————————————————————————————————————————————————————————————
        mm_2025_2026.df = mm_2025_2026.df.pivot_table(
            index=["date", "ca"],
            columns="media",
            values="budget",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        compute_spend_distribution(mm_2025_2026.df.drop(columns=["date", "ca"]))

        # Add 20 empty weeks for future creation of lagged columns
        last_date = mm_2025_2026.df["date"].max()
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=7), periods=20, freq="7D"
        )
        empty_weeks = pd.DataFrame({"date": future_dates})

        for col in mm_2025_2026.df.columns:
            if col != "date":
                empty_weeks[col] = 0

        mm_2025_2026.df = pd.concat([mm_2025_2026.df, empty_weeks], ignore_index=True)

        # Remove future dates
        mm_2025_2026.df = mm_2025_2026.df[mm_2025_2026.df["date"] <= pd.Timestamp.now()]

        mm_2025_2026.df = create_holiday_columns(mm_2025_2026.df)
        # mm_2025_2026.df = create_national_temperature_columns(mm_2025_2026.df)

        mm_2025_2026.df["intercept"] = 1
        mm_2025_2026.df["trend"] = np.linspace(0, 1, len(mm_2025_2026.df))

        unkown_future = mm_2025_2026.df.loc[mm_2025_2026.df["ca"] == 0, "date"].min()
        mm_2025_2026.df["date"] = pd.to_datetime(mm_2025_2026.df["date"])
        mm_2025_2026.df = mm_2025_2026.df[mm_2025_2026.df["date"] < unkown_future]

    return mm_2025_2026.df
