from __future__ import print_function, division
import os
import os.path as op
import numpy as np
import pandas as pd
import time
from Data import dgp_config as dcf
from tqdm import tqdm

def build_all_period_ret_caches(country="CN"):

    if country=="USA":
        df = processed_US_data()                               # MultiIndex
    if country=="CN":
        df = processed_CN_data()
    if "log_ret" not in df.columns:
        df["log_ret"] = np.log1p(df["Ret"])

    if "cum_log_ret" not in df.columns:
        df["cum_log_ret"] = (
            df.groupby("StockID")["log_ret"].cumsum(skipna=True)
        )

    # Daily-horizon forward returns
    for d in (5, 20, 60):
        col = f"Ret_{d}d"
        if col not in df.columns:
            df[col] = (
                df.groupby("StockID", group_keys=False)["cum_log_ret"]
                  .apply(lambda x: np.exp(x.shift(-d) - x) - 1)
            )

    # Rolling windows 6-20 d and 6-60 d
    for start, end in ((6, 20), (6, 60)):
        col = f"Ret_{start}-{end}d"
        if col not in df.columns:
            df[col] = (
                df.groupby("StockID", group_keys=False)["cum_log_ret"]
                  .apply(lambda x: np.exp(x.shift(-start) - x.shift(-end)) - 1)
            )

    # ------------------------------------------------------------------
    # STEP 2 ─ handle week / month / quarter one by one
    # ------------------------------------------------------------------
    for freq in ("week", "month", "quarter"):

        freq_col       = f"{freq}_ret"
        next_col       = f"next_{freq}_ret"
        next_col_delay = f"{next_col}_0delay"

        # 2-a) Pick out the period-end rows (Fridays for weeks, etc.)
        period_ends = get_period_end_dates(freq, country=country)
        is_period_end = df.index.get_level_values("Date").isin(period_ends)
        period_df = df.loc[is_period_end].copy()

        # 2-b) Period return for *this* period (if not already present)
        if freq_col not in df.columns:
            period_df[freq_col] = (
                period_df.groupby("StockID", group_keys=False)["cum_log_ret"]
                         .apply(lambda x: np.exp(x.shift(-1) - x) - 1)
            )
            df.loc[period_df.index, freq_col] = period_df[freq_col]

        # 2-c) ***NEW*** – compute next-period return **inside the
        #        period-end slice only**, then write it back
        period_df[next_col] = (
            period_df.groupby("StockID")[freq_col].shift(-1)
        )
        df.loc[period_df.index, next_col] = period_df[next_col]
        df[next_col_delay] = df[next_col]          # simple duplicate

        # ------------------------------------------------------------------
        # STEP 3 ─ build the export frame
        # ------------------------------------------------------------------
        base_cols = [
            "MarketCap", "log_ret",
            "Ret_5d", "Ret_20d", "Ret_60d", "Ret_6-20d", "Ret_6-60d",
            freq_col, next_col, next_col_delay,
        ]

        mask   = df[freq_col].notna()              # keep only period-end rows
        out_df = (
            df.loc[mask, base_cols]                 # (1) filter while idx intact
              .reset_index()                        # (2) Date & StockID to cols
              .sort_values(["StockID", "Date"])     # (3) tidy order
              .copy()
        )

        # ------------------------------------------------------------------
        # STEP 4 ─ write parquet
        # ------------------------------------------------------------------
        out_path = op.join(dcf.CACHE_DIR, f"{country}_{freq}_ret_5.pq")
        out_df.to_parquet(out_path, index=False)
        print(f"✅ Saved {out_path}  — shape={out_df.shape}")

def get_processed_US_data_by_year(year, df):
    #df = processed_US_data()

    df = df[
        df.index.get_level_values("Date").year.isin([year, year - 1, year - 2])
    ].copy()
    return df


def get_benchmark_returns(freq, country):
    assert country in ["US", "CN"]
    assert freq in ["week", "month", "quarter"]

    if country == "US":
        df = pd.read_csv(
            os.path.join(dcf.CACHE_DIR, f"spy_{freq}_ret.csv"),
            parse_dates=["date"],
        )
        df.rename(columns={"date": "Date"}, inplace=True)
        df = df.set_index("Date")

        return df["ewretx"]
    if country == "CN":
        df = pd.read_csv(
            os.path.join(dcf.CACHE_DIR, f"SSE_{freq}.csv"),
            parse_dates=["date"],
        )
        df.rename(columns={"date": "Date"}, inplace=True)
        df = df.set_index("Date")
        return df["Ret"]


def get_spy_freq_rets(freq):
    assert freq in ["week", "month", "quarter"]
    spy = pd.read_csv(
        os.path.join(dcf.CACHE_DIR, f"spy_{freq}_ret.csv"),
        parse_dates=["date"],
    )
    spy.rename(columns={"date": "Date"}, inplace=True)
    spy = spy.set_index("Date")
    return spy


def get_period_end_dates(period, country="USA"):
    assert period in ["week", "month", "quarter"]

    if country == "USA":
        spy = get_spy_freq_rets(period)
        return spy.index

    if country == "CN":
        cache_path = op.join(dcf.CACHE_DIR, f"cn_period_end_dates_{period}.csv")

        if not op.exists(cache_path):
            raise FileNotFoundError(
                f"{cache_path} not found. Run processed_CN_data() once to build it."
            )

        dates = pd.read_csv(cache_path, parse_dates=["Date"])
        return pd.DatetimeIndex(dates["Date"])

    raise ValueError(f"Unsupported country code: {country}")

# def getProcessedData(country):
#     data_path = op.join(dcf.PROCESSED_DATA_DIR, "{country}_processed.feather")
#     if op.exists(processed_us_data_path):
#         print(f"Loading processed data from {processed_us_data_path}")
#         since = time.time()
#         df = pd.read_feather(processed_us_data_path)

#         df.set_index(["Date", "StockID"], inplace=True)
#         df.sort_index(inplace=True)
#         print(f"Finish loading processed data in {(time.time() - since) / 60:.2f} min")

#         return df.copy()

# def getProcessedData(country = "USA")
def processed_US_data():
    processed_us_data_path = op.join(dcf.PROCESSED_DATA_DIR, "us_ret.feather")
    
    if op.exists(processed_us_data_path):
        print(f"Loading processed data from {processed_us_data_path}")
        since = time.time()
        df = pd.read_feather(processed_us_data_path)

        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        print(f"Finish loading processed data in {(time.time() - since) / 60:.2f} min")

        return df.copy()

    raw_us_data_path = op.join(dcf.RAW_DATA_DIR, "us_920101-200731.csv")
    print("Reading raw data from {}".format(raw_us_data_path))

    since = time.time()
    df = pd.read_csv(
        raw_us_data_path,
        parse_dates=["date"],
        dtype={
            "PERMNO": str,
            "BIDLO": np.float64,
            "ASKHI": np.float64,
            "PRC": np.float64,
            "VOL": np.float64,
            "SHROUT": np.float64,
            "OPENPRC": np.float64,
            "RET": object,
            "EXCHCD": np.float64,
        },
        compression="gzip",
        header=0,
    )
    print(f"finish reading data in {(time.time() - since) / 60:.2f} s")
    df = process_raw_data_helper(df)

    df.reset_index().to_feather(processed_us_data_path)
    return df.copy()


def process_raw_data_helper(df):
    # ---------- basic cleaning / renaming ----------
    df = df.rename(
        columns={
            "date":  "Date",
            "PERMNO":"StockID",
            "BIDLO": "Low",
            "ASKHI": "High",
            "PRC":   "Close",
            "VOL":   "Vol",
            "SHROUT":"Shares",
            "OPENPRC":"Open",
            "RET":   "Ret",
        }
    )
    df["StockID"] = df["StockID"].astype(str)
    df["Ret"]     = df["Ret"].astype(str)

    df = df.replace({
        "Close": {0: np.nan}, "Open":  {0: np.nan},
        "High":  {0: np.nan}, "Low":   {0: np.nan},
        "Ret":   {"C": np.nan, "B": np.nan, "A": np.nan, ".": np.nan},
        "Vol":   {0: np.nan, -99: np.nan},
    })

    if "Shares" not in df.columns:
        df["Shares"] = 0

    df["Ret"] = df["Ret"].astype(np.float64)
    df = df.dropna(subset=["Ret"])

    df[["Close", "Open", "High", "Low", "Vol", "Shares"]] = df[
        ["Close", "Open", "High", "Low", "Vol", "Shares"]
    ].abs()

    # sort & index -----------------------------------------------------
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)

    # ---------- BACKWARD adjustment per stock ------------------------
    def apply_backward_adjust(stock_df):
        # stock_df index: (Date, StockID); ascending by Date
        stock_df = stock_df.reset_index()

        if stock_df.empty or stock_df.loc[len(stock_df)-1, "Close"] in (0, np.nan):
            return pd.DataFrame([], columns=stock_df.columns).set_index(["Date", "StockID"])

        adjusted = stock_df.copy()
        last_idx = len(adjusted) - 1

        # Set last day's AdjClose equal to its raw close (anchor)
        adjusted.at[last_idx, "AdjClose"] = adjusted.at[last_idx, "Close"]

        # Walk *backwards* from second-last row to first
        for i in range(last_idx - 1, -1, -1):
            next_adj_close = adjusted.at[i + 1, "AdjClose"]
            ret_next       = adjusted.at[i + 1, "Ret"]           # use next day's return
            adj_close      = next_adj_close / (1 + ret_next)     # back-adjust step

            raw_close = adjusted.at[i, "Close"]
            ratio     = adj_close / raw_close if raw_close else np.nan

            adjusted.at[i, "AdjClose"] = adj_close
            adjusted.at[i, "Open"]  *= ratio
            adjusted.at[i, "High"]  *= ratio
            adjusted.at[i, "Low"]   *= ratio
            adjusted.at[i, "Close"]  = adj_close                 # overwrite Close

        adjusted.set_index(["Date", "StockID"], inplace=True)
        return adjusted

    # print("Applying BACKWARD price adjustment...")
    # adjusted_parts = []
    # for sid, g in tqdm(df.groupby(level="StockID"), desc="Adjusting by StockID"):
    #     adjusted_parts.append(apply_backward_adjust(g))

    # df = pd.concat(adjusted_parts).sort_index()

    df = add_derived_features(df, country="USA")
    return df


def add_derived_features(df, country="USA"):
    df["MarketCap"] = np.abs(df["Close"] * df["Shares"])
    
    df["log_ret"] = np.log(1 + df.Ret)
    df["cum_log_ret"] = df.groupby("StockID")["log_ret"].cumsum(skipna=True)
    df["EWMA_vol"] = df.groupby("StockID")["Ret"].transform(
        lambda x: (x**2).ewm(alpha=0.05).mean().shift(periods=1)
    )
    
    #
    for freq in ["week", "month", "quarter"]:
        period_end_dates = get_period_end_dates(freq, country=country)
        freq_df = df[df.index.get_level_values("Date").isin(period_end_dates)].copy()
        freq_df["freq_ret"] = freq_df.groupby("StockID", group_keys=False)["cum_log_ret"].apply(
            lambda x: np.exp(x.shift(-1) - x) - 1
        )
        print(
            f"Freq: {freq}: {len(freq_df)}/{len(df)} preriod_end_dates from \
                        {period_end_dates[0]}, {period_end_dates[1]},  to {period_end_dates[-1]}"
        )
        df[f"Ret_{freq}"] = freq_df["freq_ret"]
        num_nan = np.sum(pd.isna(df[f"Ret_{freq}"]))
        print(f"df Ret_{freq} {len(df) - num_nan}/{len(df)} not nan")

    for i in [5, 20, 60, 65, 180, 250, 260]:
        print(f"Calculating {i}d return")
        df[f"Ret_{i}d"] = df.groupby("StockID", group_keys=False)["cum_log_ret"].apply(
            lambda x: np.exp(x.shift(-i) - x) - 1
        )
    return df


def get_period_ret(period, country="US"):
    assert country in ["US", "CN"]
    assert period in ["week", "month", "quarter"]
    period_ret_path = op.join(dcf.CACHE_DIR, f"{country}_{period}_ret.pq")
    period_ret = pd.read_parquet(period_ret_path)
    period_ret.set_index(["Date", "StockID"], inplace=True)
    period_ret.sort_index(inplace=True)
    return period_ret


# def getProcessedData(country = "CHN")
def processed_CN_data():
    """
    Load pre-processed China A-share data, or build it from raw.
    Also builds per-period end-date caches for week / month / quarter.
    """
    processed_cn_data_path = op.join(dcf.PROCESSED_DATA_DIR, "cn_ret.feather")

    # ---- fast path ---------------------------------------------------
    if op.exists(processed_cn_data_path):
        print(f"Loading processed data from {processed_cn_data_path}")
        since = time.time()
        df = pd.read_feather(processed_cn_data_path)
        df.set_index(["Date", "StockID"], inplace=True)
        df.sort_index(inplace=True)
        print(f"Finish loading in {(time.time() - since) / 60:.2f} min")
        return df.copy()

    # ---- slow path: read & preprocess raw ----------------------------
    raw_cn_data_path = op.join(dcf.RAW_DATA_DIR, "CSMAR/cn_93-25.csv")
    print(f"Reading raw data from {raw_cn_data_path}")

    since = time.time()
    df = pd.read_csv(
        raw_cn_data_path,
        parse_dates=["Trddt"],
        dtype={
            "Stkcd":      str,
            "Opnprc":     np.float64,
            "Hiprc":      np.float64,
            "Loprc":      np.float64,
            "Clsprc":     np.float64,
            "Dnshrtrd":   np.float64,
            "Dnvaltrd":   np.float64,
            "Dretwd":     object,
            "Dsmvosd":    np.float64,
            "Markettype": np.float64,
            "Adjprcwd":   np.float64,
        },
        header=0,
    )
    print(f"Finish reading in {(time.time() - since):.2f} s")

    # ➊ ---------- build CN period-end caches --------------------------
    df["Date"] = pd.to_datetime(df["Trddt"])       # already parsed but explicit
    for freq, pandas_code in [("week", "W"), ("month", "M"), ("quarter", "Q")]:
        period_ends = (
            df.groupby(df["Date"].dt.to_period(pandas_code))["Date"]
              .max()
              .sort_values()
              .unique()
        )
        cache_path = op.join(dcf.CACHE_DIR, f"cn_period_end_dates_{freq}.csv")
        pd.DataFrame({"Date": period_ends}).to_csv(cache_path, index=False)
        print(f"Saved {freq} period-end cache → {cache_path}")
    df = df.drop(columns="Date") 
    # -----------------------------------------------------------------

    # ---- clean & feature-engineer -----------------------------------
    df = process_cn_raw_data_helper(df)

    # ---- cache the fully processed feather --------------------------
    df.reset_index().to_feather(processed_cn_data_path)
    return df.copy()

def process_cn_raw_data_helper(df):
    # 1️⃣ rename --------------------------------------------------------
    df = df.rename(
        columns={
            "Trddt": "Date", "Stkcd": "StockID",
            "Loprc": "Low",  "Hiprc": "High",
            "Clsprc": "Close","Opnprc": "Open",
            "Dnshrtrd": "Vol",
            "Dretwd":  "Ret",
        }
    )

    # 2️⃣ keep only essentials (add Markettype & Adjprcwd for scaling) --
    keep_cols = [
        "Date", "StockID", "Low", "High", "Close", "Open",
        "Vol", "Ret", "Dsmvosd", "Markettype", "Adjprcwd"
    ]
    df = df.loc[:, df.columns.intersection(keep_cols)]

    # 3️⃣ basic dtypes --------------------------------------------------
    df["Date"]    = pd.to_datetime(df["Date"])
    df["StockID"] = df["StockID"].astype(str)
    df["Ret"]     = pd.to_numeric(df["Ret"], errors="coerce")

    # 🆕 3b.  create EXCHCD and drop Markettype -------------------------
    df["EXCHCD"]  = pd.to_numeric(df["Markettype"], errors="coerce")
    df = df.drop(columns="Markettype")

    # 🆕 3c.  replace raw OHLC with adjusted OHLC -----------------------
    if "Adjprcwd" in df.columns:
        ratio = df["Adjprcwd"] / df["Close"]
        df["Close"] = df["Adjprcwd"]                      # adjusted Close
        df["Open"]  = df["Open"]  * ratio
        df["High"]  = df["High"]  * ratio
        df["Low"]   = df["Low"]   * ratio
    df = df.drop(columns="Adjprcwd", errors="ignore")

    # 4️⃣ derive Shares -------------------------------------------------
    if "Dsmvosd" in df.columns:
        shares_raw = df["Dsmvosd"] / df["Close"] / ratio if 'ratio' in locals() else \
                     df["Dsmvosd"] / df["Close"]
        df["Shares"] = shares_raw
    else:
        df["Shares"] = 0.0
    df = df.drop(columns="Dsmvosd", errors="ignore")

    # 5️⃣ sentinels 6️⃣ dropna 7️⃣ abs() 8️⃣ index (unchanged) ----------
    df = df.replace({
        "Close": {0: np.nan}, "Open":  {0: np.nan},
        "High":  {0: np.nan}, "Low":   {0: np.nan},
        "Vol":   {0: np.nan, -99: np.nan},
    })
    df = df.dropna(subset=["Ret"])
    num_cols = ["Close", "Open", "High", "Low", "Vol", "Shares"]
    df[num_cols] = df[num_cols].abs()
    df.set_index(["Date", "StockID"], inplace=True)
    df.sort_index(inplace=True)

    df = add_derived_features(df, country="CN")
    return df