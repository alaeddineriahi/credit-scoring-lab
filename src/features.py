"""Load the raw UCI file, clean it, and build behavioural features with SQL (DuckDB).

Month convention: m = 1 is the most recent month (Sept 2005) and m = 6 the oldest (April 2005).
In the source data, the payment made in month m settles the bill of month m + 1.
"""
import duckdb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ID_COL, PROCESSED_DIR, PROTECTED, RAW_XLS, SEED, SPLIT, TARGET

PAY_SRC = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW_XLS, header=1, engine="calamine")
    rename = {"ID": ID_COL, "LIMIT_BAL": "limit_bal", "SEX": "sex", "EDUCATION": "education",
              "MARRIAGE": "marriage", "AGE": "age", "default payment next month": TARGET}
    for m, col in enumerate(PAY_SRC, start=1):
        rename[col] = f"pay_status_{m}"
        rename[f"BILL_AMT{m}"] = f"bill_{m}"
        rename[f"PAY_AMT{m}"] = f"paid_{m}"
    df = df.rename(columns=rename)
    # Undocumented codes are folded into the documented "other" category.
    df["education"] = df["education"].replace({0: 4, 5: 4, 6: 4})
    df["marriage"] = df["marriage"].replace({0: 3})
    return df


FEATURE_SQL = """
WITH long AS (
    SELECT l.*, b.limit_bal,
           GREATEST(l.pay_status, 0)                                   AS delay,
           l.bill / b.limit_bal                                        AS util,
           LEAD(l.bill) OVER (PARTITION BY l.id ORDER BY l.m)          AS prev_bill
    FROM long_raw l JOIN base b USING (id)
),
agg AS (
    SELECT id,
        MAX(delay)                                                     AS max_delay_6m,
        MAX(delay) FILTER (WHERE m <= 3)                               AS max_delay_3m,
        COUNT(*)   FILTER (WHERE delay >= 1)                           AS n_delay_months,
        COALESCE(MIN(m) FILTER (WHERE delay >= 1), 7)                  AS months_since_delay,
        COUNT(*)   FILTER (WHERE pay_status = -1)                      AS n_paid_in_full,
        COUNT(*)   FILTER (WHERE pay_status = -2)                      AS n_no_usage,
        MAX(delay) FILTER (WHERE m = 1)
          - AVG(delay) FILTER (WHERE m >= 4)                           AS delay_trend,
        MAX(util)  FILTER (WHERE m = 1)                                AS util_last,
        AVG(util)                                                      AS util_avg,
        MAX(util)                                                      AS util_max,
        REGR_SLOPE(util, 7 - m)                                        AS util_slope,
        AVG(LEAST(paid / NULLIF(prev_bill, 0), 2))
            FILTER (WHERE prev_bill > 0)                               AS pay_ratio_avg,
        MIN(LEAST(paid / NULLIF(prev_bill, 0), 2))
            FILTER (WHERE prev_bill > 0 AND m <= 3)                    AS pay_ratio_min_3m,
        COUNT(*)   FILTER (WHERE paid = 0 AND prev_bill > 0)           AS n_missed_payments,
        SUM(paid) / ANY_VALUE(limit_bal)                               AS paid_to_limit_6m,
        MAX(bill)  FILTER (WHERE m = 1)
          / NULLIF(AVG(bill) FILTER (WHERE m >= 2), 0)                 AS bill_growth
    FROM long
    GROUP BY id
)
SELECT b.*, a.* EXCLUDE (id)
FROM base b JOIN agg a USING (id)
ORDER BY b.id
"""


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    long_raw = pd.wide_to_long(
        df[[ID_COL] + [c for c in df.columns if c.startswith(("pay_status_", "bill_", "paid_"))]],
        stubnames=["pay_status", "bill", "paid"], i=ID_COL, j="m", sep="_",
    ).reset_index()
    con = duckdb.connect()
    con.register("base", df)
    con.register("long_raw", long_raw)
    out = con.execute(FEATURE_SQL).df()
    con.close()
    # Ratios on empty denominators (e.g. no bill history) become neutral values, not NaN.
    out["pay_ratio_avg"] = out["pay_ratio_avg"].fillna(1.0)
    out["pay_ratio_min_3m"] = out["pay_ratio_min_3m"].fillna(1.0)
    out["bill_growth"] = out["bill_growth"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(-5, 5)
    out["util_slope"] = out["util_slope"].fillna(0.0)
    # pay_status mixes payment codes (-2 no usage, -1 paid in full, 0 minimum paid) with months of
    # delay (1..8). Split it so that delay can carry a monotonic constraint while codes stay free:
    # in this data "paid in full" defaults more often than "minimum paid", so the raw column is not monotone.
    for m in range(1, 7):
        s = out[f"pay_status_{m}"]
        out[f"delay_{m}"] = s.clip(lower=0)
        out[f"code_{m}"] = s.clip(upper=1)   # -2, -1, 0, or 1 (= late)
    return out


def add_split(df: pd.DataFrame) -> pd.DataFrame:
    train_frac, valid_frac, test_frac = SPLIT
    idx_train, idx_rest = train_test_split(df.index, test_size=valid_frac + test_frac,
                                           stratify=df[TARGET], random_state=SEED)
    idx_valid, idx_test = train_test_split(idx_rest, test_size=test_frac / (valid_frac + test_frac),
                                           stratify=df.loc[idx_rest, TARGET], random_state=SEED)
    df = df.copy()
    df.loc[idx_train, "split"] = "train"
    df.loc[idx_valid, "split"] = "valid"
    df.loc[idx_test, "split"] = "test"
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {ID_COL, TARGET, "split", *PROTECTED}
    # Raw pay_status_m is replaced by its two components delay_m and code_m.
    return [c for c in df.columns if c not in excluded and not c.startswith("pay_status_")]


def main() -> None:
    df = add_split(build_features(load_raw()))
    path = PROCESSED_DIR / "features.parquet"
    df.to_parquet(path, index=False)
    feats = feature_columns(df)
    print(f"Saved {path.name}: {len(df):,} rows, {len(feats)} model features")
    print(df.groupby("split")[TARGET].agg(["size", "mean"]).round(4))


if __name__ == "__main__":
    main()
