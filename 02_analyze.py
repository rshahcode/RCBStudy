#!/usr/bin/env python3
"""Estimate expected annual cancer cases among refugees under UNHCR's mandate.

The analysis averages published end-2024 and end-2025 REF age-by-sex counts.
No missing demographic counts are imputed or rescaled. The same midpoint
population is evaluated under origin-country and host-country GLOBOCAN 2024
proxy rates. Results are modeled expectations, not observed refugee incidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, __version__ as PIL_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
QA = PROJECT_ROOT / "qa"
DOCUMENTATION = PROJECT_ROOT / "documentation"
for directory in (PROCESSED, TABLES, FIGURES, QA, DOCUMENTATION):
    directory.mkdir(parents=True, exist_ok=True)

YEARS = (2024, 2025)
REF_STOCK_COLUMN = "Refugees under UNHCR's mandate"
SEX_LABELS = {1: "Male", 2: "Female"}
AGE_ORDER = ["0-17", "18-59", "60+"]
AGE_SOURCE_LABELS = {
    "0-17": "UNHCR 0-17 mapped to GLOBOCAN 0-19",
    "18-59": "UNHCR 18-59 mapped to GLOBOCAN 20-59",
    "60+": "UNHCR 60+ mapped to GLOBOCAN 60+",
}
SCENARIOS = ["Origin-country rates", "Host-country rates"]
ALL_CANCER_CODE = 40
NMSC_CODE = 17
CUSTOM_COLORECTAL_CODE = 8009
SITE_CODES = [code for code in range(1, 39) if code != NMSC_CODE]

FEMALE_AGE_COLUMNS = [
    "Female 0 - 4",
    "Female 5 - 11",
    "Female 12 - 17",
    "Female 18 - 59",
    "Female 60",
]
MALE_AGE_COLUMNS = [
    "Male 0 - 4",
    "Male 5 - 11",
    "Male 12 - 17",
    "Male 18 - 59",
    "Male 60",
]
UNHCR_NUMERIC_COLUMNS = FEMALE_AGE_COLUMNS + MALE_AGE_COLUMNS + [
    "Female other",
    "Female total",
    "Male other",
    "Male total",
    "Total",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pct(numerator: float, denominator: float) -> float:
    return numerator / denominator * 100 if denominator else float("nan")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_unhcr() -> tuple[pd.DataFrame, dict[int, float], pd.DataFrame]:
    demographic_frames: list[pd.DataFrame] = []
    population_frames: list[pd.DataFrame] = []
    stocks: dict[int, float] = {}

    for year in YEARS:
        demographics_path = RAW / "unhcr" / str(year) / "demographics.csv"
        population_path = RAW / "unhcr" / str(year) / "population" / "population.csv"
        demographics = pd.read_csv(demographics_path)
        require(set(demographics["Year"]) == {year}, f"UNHCR demographics are not {year} only")
        for column in UNHCR_NUMERIC_COLUMNS:
            converted = pd.to_numeric(demographics[column], errors="coerce")
            require(converted.notna().all(), f"Nonnumeric/missing {year} UNHCR values: {column}")
            require((converted >= 0).all(), f"Negative {year} UNHCR values: {column}")
            demographics[column] = converted.astype(float)

        demographics = demographics.rename(
            columns={
                "Country of origin": "origin",
                "Country of origin (ISO)": "origin_iso",
                "Country of asylum": "host",
                "Country of asylum (ISO)": "host_iso",
                "Population Type": "pop_type",
                "Total": "row_total",
                "Year": "year",
            }
        )
        demographics["row_id"] = [f"{year}-{index}" for index in range(len(demographics))]
        demographics["origin_iso"] = demographics["origin_iso"].fillna("").astype(str).str.strip()
        demographics["host_iso"] = demographics["host_iso"].fillna("").astype(str).str.strip()
        demographics["observed_age_sex"] = demographics[
            FEMALE_AGE_COLUMNS + MALE_AGE_COLUMNS
        ].sum(axis=1)
        demographics["known_sex"] = demographics["Female total"] + demographics["Male total"]
        require(
            (demographics["observed_age_sex"] <= demographics["known_sex"] + 1e-9).all(),
            f"At least one {year} row has age-sex cells exceeding sex totals",
        )
        require(
            (demographics["known_sex"] <= demographics["row_total"] + 1e-9).all(),
            f"At least one {year} row has sex totals exceeding its published total",
        )
        demographics["row_complete"] = np.isclose(
            demographics["observed_age_sex"], demographics["row_total"]
        )

        population = pd.read_csv(population_path).rename(columns={"Year": "year"})
        require(set(population["year"]) == {year}, f"UNHCR population file is not {year} only")
        ref_values = pd.to_numeric(
            population[REF_STOCK_COLUMN].replace("-", np.nan), errors="coerce"
        ).fillna(0)
        require((ref_values >= 0).all(), f"Negative {year} REF stock")
        stocks[year] = float(ref_values.sum())
        population["ref_stock"] = ref_values.astype(float)

        ref_demo_total = demographics.loc[demographics["pop_type"] == "REF", "row_total"].sum()
        require(
            np.isclose(ref_demo_total, stocks[year]),
            f"UNHCR {year} REF total differs between demographic and population downloads",
        )
        ref_keys = demographics.loc[demographics["pop_type"] == "REF", ["origin_iso", "host_iso"]]
        require(
            not ref_keys.duplicated().any(),
            f"Duplicate {year} REF origin-host rows in demographics",
        )
        population_ref = population.loc[ref_values > 0].copy()
        population_ref["origin_iso"] = population_ref["Country of origin (ISO)"].fillna("").astype(str)
        population_ref["host_iso"] = population_ref["Country of asylum (ISO)"].fillna("").astype(str)
        require(
            not population_ref[["origin_iso", "host_iso"]].duplicated().any(),
            f"Duplicate {year} REF origin-host rows in population extract",
        )

        demographic_frames.append(demographics)
        population_frames.append(population)

    return (
        pd.concat(demographic_frames, ignore_index=True),
        stocks,
        pd.concat(population_frames, ignore_index=True),
    )


def load_gco_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    populations = pd.DataFrame(
        json.loads((RAW / "gco" / "2024" / "populations.json").read_text(encoding="utf-8"))
    )
    cancers = pd.DataFrame(
        json.loads((RAW / "gco" / "2024" / "cancers.json").read_text(encoding="utf-8"))
    )
    populations["country_code"] = pd.to_numeric(
        populations["country_code"], errors="raise"
    ).astype(int)
    countries = populations.loc[populations["country_code"] < 900].copy()
    countries["country_iso3"] = countries["country_iso3"].fillna("").astype(str)
    require(len(countries) == 186, f"Expected 186 GLOBOCAN countries; found {len(countries)}")
    require(countries["country_iso3"].nunique() == 186, "GLOBOCAN ISO3 codes not unique")
    cancer_codes = set(pd.to_numeric(cancers["cancer"], errors="raise").astype(int))
    require(set(range(1, 42)).issubset(cancer_codes), "Required GLOBOCAN cancer metadata missing")
    return countries, cancers


def read_gco_json(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(
        isinstance(payload, dict) and isinstance(payload.get("dataset"), list),
        f"Unexpected GLOBOCAN response: {path}",
    )
    return pd.DataFrame(payload["dataset"])


def load_direct_rates(countries: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("0_19", "0-17", "0-19"),
        ("20_59", "18-59", "20-59"),
        ("60_plus", "60+", "60+"),
    ]
    frames = []
    for file_label, analysis_age, gco_age in definitions:
        frame = read_gco_json(
            RAW / "gco" / "2024" / "incidence_broad" / f"age_{file_label}.json"
        )
        frame["analysis_age"] = analysis_age
        frame["gco_age"] = gco_age
        frames.append(frame)
    rates = pd.concat(frames, ignore_index=True)
    for column in ("country_code", "sex", "cancer_code"):
        rates[column] = pd.to_numeric(rates[column], errors="raise").astype(int)
    for column in ("total", "total_pop", "crude_rate", "asr"):
        if column in rates:
            rates[column] = pd.to_numeric(rates[column], errors="coerce")
    rates = rates.loc[
        (rates["country_code"] < 900) & rates["cancer_code"].between(1, 40)
    ].copy()
    rates = rates.merge(
        countries[
            ["country_code", "country_iso3", "label", "method_incidence", "grouping", "area_label"]
        ],
        on="country_code",
        how="inner",
        validate="many_to_one",
    ).rename(columns={"label": "country", "crude_rate": "api_crude_rate"})
    rates["rate_per_100k"] = np.where(
        rates["total_pop"] > 0,
        rates["total"] / rates["total_pop"] * 100_000,
        np.nan,
    )
    key = ["country_iso3", "sex", "cancer_code", "analysis_age"]
    require(not rates.duplicated(key).any(), "Duplicate GLOBOCAN broad-age rate keys")
    require(rates["rate_per_100k"].notna().all(), "Missing GLOBOCAN broad-age rates")
    return rates


def load_boundary_sensitivity_rates(countries: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for age_group in range(18):
        frame = read_gco_json(
            RAW / "gco" / "2024" / "incidence" / f"age_{age_group:02d}.json"
        )
        frame["age_group"] = age_group
        frames.append(frame)
    five_year = pd.concat(frames, ignore_index=True)
    for column in ("country_code", "sex", "cancer_code", "age_group"):
        five_year[column] = pd.to_numeric(five_year[column], errors="raise").astype(int)
    for column in ("total", "total_pop"):
        five_year[column] = pd.to_numeric(five_year[column], errors="coerce")
    five_year = five_year.loc[
        (five_year["country_code"] < 900) & five_year["cancer_code"].between(1, 40)
    ].copy()

    overlap_rows: list[dict[str, object]] = []
    for age_group in range(18):
        if age_group <= 2:
            overlap_rows.append({"age_group": age_group, "analysis_age": "0-17", "fraction": 1.0})
        elif age_group == 3:
            overlap_rows.extend(
                [
                    {"age_group": 3, "analysis_age": "0-17", "fraction": 3 / 5},
                    {"age_group": 3, "analysis_age": "18-59", "fraction": 2 / 5},
                ]
            )
        elif age_group <= 11:
            overlap_rows.append({"age_group": age_group, "analysis_age": "18-59", "fraction": 1.0})
        else:
            overlap_rows.append({"age_group": age_group, "analysis_age": "60+", "fraction": 1.0})
    work = five_year.merge(pd.DataFrame(overlap_rows), on="age_group", validate="many_to_many")
    work["weighted_cases"] = work["total"] * work["fraction"]
    work["weighted_population"] = work["total_pop"] * work["fraction"]
    grouped = work.groupby(
        ["country_code", "sex", "cancer_code", "analysis_age"], as_index=False
    )[["weighted_cases", "weighted_population"]].sum()
    grouped["rate_per_100k"] = (
        grouped["weighted_cases"] / grouped["weighted_population"] * 100_000
    )
    grouped["api_crude_rate"] = grouped["rate_per_100k"]
    grouped = grouped.merge(
        countries[["country_code", "country_iso3", "label"]],
        on="country_code",
        how="inner",
        validate="many_to_one",
    ).rename(columns={"label": "country"})
    require(grouped["rate_per_100k"].notna().all(), "Missing boundary-sensitivity rates")
    return grouped


def load_host_denominators(countries: pd.DataFrame) -> pd.DataFrame:
    frame = read_gco_json(
        RAW / "gco" / "2024" / "incidence_broad" / "age_all_ages.json"
    )
    for column in ("country_code", "sex", "cancer_code"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["total"] = pd.to_numeric(frame["total"], errors="coerce")
    national = frame.loc[
        (frame["country_code"] < 900) & (frame["cancer_code"] == ALL_CANCER_CODE)
    ].copy()
    burden = national.groupby("country_code", as_index=False)["total"].sum().rename(
        columns={"total": "gco_host_cancer_cases"}
    )
    burden = burden.merge(
        countries[["country_code", "country_iso3", "label"]],
        on="country_code",
        how="inner",
        validate="one_to_one",
    ).rename(columns={"label": "host"})
    require(len(burden) == 186, "Host cancer denominator does not cover 186 countries")
    burden.attrs["world_total"] = float(
        frame.loc[
            (frame["country_code"] == 900) & (frame["cancer_code"] == ALL_CANCER_CODE),
            "total",
        ].sum()
    )
    burden.attrs["country_sum"] = float(burden["gco_host_cancer_cases"].sum())
    return burden


def add_crosswalk_flags(rows: pd.DataFrame, countries: pd.DataFrame) -> pd.DataFrame:
    gco_isos = set(countries["country_iso3"])
    work = rows.copy()
    work["origin_defined"] = work["origin_iso"].str.len().eq(3)
    work["host_defined"] = work["host_iso"].str.len().eq(3)
    work["distinct_origin_host"] = work["origin_iso"].ne(work["host_iso"])
    work["origin_gco_match"] = work["origin_iso"].isin(gco_isos)
    work["host_gco_match"] = work["host_iso"].isin(gco_isos)
    work["analysis_geography"] = (
        work["origin_defined"]
        & work["host_defined"]
        & work["distinct_origin_host"]
        & work["origin_gco_match"]
        & work["host_gco_match"]
    )
    conditions = [
        ~work["origin_defined"],
        work["origin_defined"] & ~work["origin_gco_match"],
        ~work["host_defined"],
        work["host_defined"] & ~work["host_gco_match"],
        work["origin_defined"] & work["host_defined"] & ~work["distinct_origin_host"],
    ]
    labels = [
        "Origin not defined",
        "Origin absent from GLOBOCAN",
        "Host not defined",
        "Host absent from GLOBOCAN",
        "Origin and host identical",
    ]
    work["exclusion_reason"] = np.select(conditions, labels, default="Included")
    return work


def make_observed_cells(rows: pd.DataFrame, weight: float = 1.0) -> pd.DataFrame:
    work = rows.copy()
    work["male_0_17"] = work[MALE_AGE_COLUMNS[:3]].sum(axis=1)
    work["male_18_59"] = work[MALE_AGE_COLUMNS[3]]
    work["male_60_plus"] = work[MALE_AGE_COLUMNS[4]]
    work["female_0_17"] = work[FEMALE_AGE_COLUMNS[:3]].sum(axis=1)
    work["female_18_59"] = work[FEMALE_AGE_COLUMNS[3]]
    work["female_60_plus"] = work[FEMALE_AGE_COLUMNS[4]]
    specifications = [
        ("male_0_17", 1, "0-17"),
        ("male_18_59", 1, "18-59"),
        ("male_60_plus", 1, "60+"),
        ("female_0_17", 2, "0-17"),
        ("female_18_59", 2, "18-59"),
        ("female_60_plus", 2, "60+"),
    ]
    identifiers = [
        "year",
        "row_id",
        "origin",
        "origin_iso",
        "host",
        "host_iso",
        "pop_type",
        "row_complete",
    ]
    frames = []
    for column, sex, analysis_age in specifications:
        piece = work[identifiers + [column]].rename(columns={column: "population_end_year"})
        piece["sex"] = sex
        piece["analysis_age"] = analysis_age
        frames.append(piece)
    long = pd.concat(frames, ignore_index=True)
    long = long.loc[long["population_end_year"] > 0].copy()
    long["midpoint_weight"] = weight
    long["population"] = long["population_end_year"] * weight
    long["sex_label"] = long["sex"].map(SEX_LABELS)
    long["age_mapping"] = long["analysis_age"].map(AGE_SOURCE_LABELS)
    require(
        np.isclose(long["population_end_year"].sum(), rows["observed_age_sex"].sum()),
        "Observed cells do not reconcile to the UNHCR age-sex total",
    )
    return long


def midpoint_cell_table(cells: pd.DataFrame) -> pd.DataFrame:
    key = ["origin", "origin_iso", "host", "host_iso", "sex", "sex_label", "analysis_age"]
    pivot = cells.pivot_table(
        index=key,
        columns="year",
        values="population_end_year",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    for year in YEARS:
        if year not in pivot:
            pivot[year] = 0.0
    pivot = pivot.rename(columns={2024: "population_end_2024", 2025: "population_end_2025"})
    pivot["population_midpoint"] = (
        pivot["population_end_2024"] + pivot["population_end_2025"]
    ) / 2
    return pivot


def apply_all_cancer(
    cells: pd.DataFrame,
    rates: pd.DataFrame,
    target: str,
    scenario: str,
) -> pd.DataFrame:
    iso_column = f"{target}_iso"
    right = rates.loc[
        rates["cancer_code"] == ALL_CANCER_CODE,
        ["country_iso3", "sex", "analysis_age", "rate_per_100k", "api_crude_rate"],
    ].copy()
    output = cells.merge(
        right,
        left_on=[iso_column, "sex", "analysis_age"],
        right_on=["country_iso3", "sex", "analysis_age"],
        how="left",
        validate="many_to_one",
    )
    require(len(output) == len(cells), f"Row multiplication in {scenario}")
    require(output["rate_per_100k"].notna().all(), f"Missing all-cancer rates in {scenario}")
    output["expected_cases"] = output["population"] * output["rate_per_100k"] / 100_000
    output["scenario"] = scenario
    output["rate_country_basis"] = target
    return output


def apply_sites(
    cells: pd.DataFrame,
    rates: pd.DataFrame,
    target: str,
    scenario: str,
) -> pd.DataFrame:
    iso_column = f"{target}_iso"
    right = rates.loc[
        rates["cancer_code"].isin(SITE_CODES),
        ["country_iso3", "sex", "analysis_age", "cancer_code", "rate_per_100k"],
    ].copy()
    output = cells.merge(
        right,
        left_on=[iso_column, "sex", "analysis_age"],
        right_on=["country_iso3", "sex", "analysis_age"],
        how="inner",
        validate="many_to_many",
    )
    output["expected_cases"] = output["population"] * output["rate_per_100k"] / 100_000
    output["scenario"] = scenario
    output["rate_country_basis"] = target
    return output


def summarize_sites(
    site_detail: pd.DataFrame,
    cancers: pd.DataFrame,
    scenario_total: float,
) -> pd.DataFrame:
    labels = {
        int(row["cancer"]): row["label"]
        for _, row in cancers.iterrows()
    }
    summary = site_detail.groupby("cancer_code", as_index=False)["expected_cases"].sum()
    colorectal = summary.loc[summary["cancer_code"].isin([8, 9]), "expected_cases"].sum()
    summary = summary.loc[~summary["cancer_code"].isin([8, 9])].copy()
    summary = pd.concat(
        [
            summary,
            pd.DataFrame(
                [{"cancer_code": CUSTOM_COLORECTAL_CODE, "expected_cases": colorectal}]
            ),
        ],
        ignore_index=True,
    )
    summary["cancer_site"] = summary["cancer_code"].map(labels)
    summary.loc[
        summary["cancer_code"] == CUSTOM_COLORECTAL_CODE, "cancer_site"
    ] = "Colorectal (colon and rectum)"
    summary["site_category"] = np.where(
        summary["cancer_code"].isin([37, 38]),
        "Residual category",
        "Named cancer site",
    )
    summary["percent_of_all_cancers"] = summary["expected_cases"] / scenario_total * 100
    summary = summary.sort_values("expected_cases", ascending=False).reset_index(drop=True)
    summary["rank_all_categories"] = np.arange(1, len(summary) + 1)
    named = summary["site_category"].eq("Named cancer site")
    summary.loc[named, "rank_named_site"] = (
        summary.loc[named, "expected_cases"].rank(method="min", ascending=False).astype(int)
    )
    return summary


def analyze_cells(
    cells: pd.DataFrame,
    rates: pd.DataFrame,
    cancers: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    origin = apply_all_cancer(cells, rates, "origin", SCENARIOS[0])
    host = apply_all_cancer(cells, rates, "host", SCENARIOS[1])
    all_detail = pd.concat([origin, host], ignore_index=True)

    origin_sites = apply_sites(cells, rates, "origin", SCENARIOS[0])
    host_sites = apply_sites(cells, rates, "host", SCENARIOS[1])
    site_detail = pd.concat([origin_sites, host_sites], ignore_index=True)
    totals = all_detail.groupby("scenario")["expected_cases"].sum().to_dict()
    site_summaries = []
    for scenario in SCENARIOS:
        summary = summarize_sites(
            site_detail.loc[site_detail["scenario"] == scenario],
            cancers,
            totals[scenario],
        )
        summary["scenario"] = scenario
        site_summaries.append(summary)
    return all_detail, site_detail, pd.concat(site_summaries, ignore_index=True)


def aggregate_all_cancer(detail: pd.DataFrame) -> dict[str, pd.DataFrame]:
    global_summary = detail.groupby("scenario", as_index=False)[
        ["population", "expected_cases"]
    ].sum()
    by_age_sex = detail.groupby(
        ["scenario", "sex", "sex_label", "analysis_age"], as_index=False
    )[["population", "expected_cases"]].sum()
    by_origin = detail.groupby(
        ["scenario", "origin", "origin_iso"], as_index=False
    )[["population", "expected_cases"]].sum()
    by_host = detail.groupby(
        ["scenario", "host", "host_iso"], as_index=False
    )[["population", "expected_cases"]].sum()
    totals = global_summary.set_index("scenario")["expected_cases"].to_dict()
    for table in (by_age_sex, by_origin, by_host):
        table["percent_of_scenario_cases"] = table.apply(
            lambda row: row["expected_cases"] / totals[row["scenario"]] * 100,
            axis=1,
        )
    for table in (by_origin, by_host):
        table["rank"] = table.groupby("scenario")["expected_cases"].rank(
            method="min", ascending=False
        ).astype(int)
    return {
        "global": global_summary,
        "age_sex": by_age_sex,
        "origin": by_origin,
        "host": by_host,
    }


def build_table_1(
    all_detail: pd.DataFrame,
    site_summary: pd.DataFrame,
    host_burden: pd.DataFrame,
) -> pd.DataFrame:
    aggregates = aggregate_all_cancer(all_detail)
    global_values = aggregates["global"].set_index("scenario")
    analytic_population = float(global_values.loc[SCENARIOS[0], "population"])
    scenario_totals = global_values["expected_cases"].to_dict()
    rows: list[dict[str, object]] = [
        {
            "section": "Overall",
            "category": "Analyzed midpoint REF population",
            "analyzed_population": analytic_population,
            "analyzed_population_pct": 100.0,
            "origin_rate_expected": scenario_totals[SCENARIOS[0]],
            "origin_rate_percent": 100.0,
            "host_rate_expected": scenario_totals[SCENARIOS[1]],
            "host_rate_percent": 100.0,
            "origin_rank": np.nan,
            "host_rank": np.nan,
            "origin_service_need_share_pct": np.nan,
            "host_service_need_share_pct": np.nan,
        }
    ]

    named_sites = site_summary.loc[site_summary["site_category"] == "Named cancer site"]
    top_site_codes = set(
        named_sites.loc[named_sites["rank_named_site"] <= 5, "cancer_code"].astype(int)
    )
    site_order = (
        named_sites.loc[named_sites["cancer_code"].isin(top_site_codes)]
        .groupby(["cancer_code", "cancer_site"], as_index=False)["rank_named_site"]
        .min()
        .sort_values(["rank_named_site", "cancer_site"])
    )
    for _, key_row in site_order.iterrows():
        code = int(key_row["cancer_code"])
        subset = named_sites.loc[named_sites["cancer_code"] == code].set_index("scenario")
        rows.append(
            {
                "section": "Cancer sites in either scenario's top 5",
                "category": key_row["cancer_site"],
                "analyzed_population": np.nan,
                "analyzed_population_pct": np.nan,
                "origin_rate_expected": float(subset.loc[SCENARIOS[0], "expected_cases"]),
                "origin_rate_percent": float(subset.loc[SCENARIOS[0], "percent_of_all_cancers"]),
                "host_rate_expected": float(subset.loc[SCENARIOS[1], "expected_cases"]),
                "host_rate_percent": float(subset.loc[SCENARIOS[1], "percent_of_all_cancers"]),
                "origin_rank": int(subset.loc[SCENARIOS[0], "rank_named_site"]),
                "host_rank": int(subset.loc[SCENARIOS[1], "rank_named_site"]),
                "origin_service_need_share_pct": np.nan,
                "host_service_need_share_pct": np.nan,
            }
        )

    for dimension, label, name_col, iso_col in [
        ("origin", "Origin countries in either scenario's top 5", "origin", "origin_iso"),
        ("host", "Host countries in either scenario's top 5", "host", "host_iso"),
    ]:
        summary = aggregates[dimension]
        top_isos = set(summary.loc[summary["rank"] <= 5, iso_col])
        order = (
            summary.loc[summary[iso_col].isin(top_isos)]
            .groupby([iso_col, name_col], as_index=False)["rank"]
            .min()
            .sort_values(["rank", name_col])
        )
        for _, key_row in order.iterrows():
            iso = key_row[iso_col]
            subset = summary.loc[summary[iso_col] == iso].set_index("scenario")
            representative = subset.loc[SCENARIOS[0]]
            origin_share = np.nan
            host_share = np.nan
            if dimension == "host":
                national = float(
                    host_burden.loc[
                        host_burden["country_iso3"] == iso, "gco_host_cancer_cases"
                    ].iloc[0]
                )
                origin_share = float(subset.loc[SCENARIOS[0], "expected_cases"]) / national * 100
                host_share = float(subset.loc[SCENARIOS[1], "expected_cases"]) / national * 100
            rows.append(
                {
                    "section": label,
                    "category": key_row[name_col],
                    "analyzed_population": float(representative["population"]),
                    "analyzed_population_pct": float(representative["population"]) / analytic_population * 100,
                    "origin_rate_expected": float(subset.loc[SCENARIOS[0], "expected_cases"]),
                    "origin_rate_percent": float(subset.loc[SCENARIOS[0], "percent_of_scenario_cases"]),
                    "host_rate_expected": float(subset.loc[SCENARIOS[1], "expected_cases"]),
                    "host_rate_percent": float(subset.loc[SCENARIOS[1], "percent_of_scenario_cases"]),
                    "origin_rank": int(subset.loc[SCENARIOS[0], "rank"]),
                    "host_rank": int(subset.loc[SCENARIOS[1], "rank"]),
                    "origin_service_need_share_pct": origin_share,
                    "host_service_need_share_pct": host_share,
                }
            )
    return pd.DataFrame(rows)


def build_table_2(all_detail: pd.DataFrame) -> pd.DataFrame:
    global_values = all_detail.groupby("scenario")[["population", "expected_cases"]].sum()
    analytic_population = float(global_values.loc[SCENARIOS[0], "population"])
    scenario_totals = global_values["expected_cases"].to_dict()
    rows: list[dict[str, object]] = []
    for characteristic, column, order in [
        ("Sex", "sex_label", ["Male", "Female"]),
        ("Age group", "analysis_age", AGE_ORDER),
    ]:
        grouped = all_detail.groupby(["scenario", column], as_index=False)[
            ["population", "expected_cases"]
        ].sum()
        for category in order:
            subset = grouped.loc[grouped[column] == category].set_index("scenario")
            population = float(subset.loc[SCENARIOS[0], "population"])
            rows.append(
                {
                    "characteristic": characteristic,
                    "category": category,
                    "analyzed_population": population,
                    "analyzed_population_pct": population / analytic_population * 100,
                    "origin_rate_expected": float(subset.loc[SCENARIOS[0], "expected_cases"]),
                    "origin_rate_percent": float(subset.loc[SCENARIOS[0], "expected_cases"]) / scenario_totals[SCENARIOS[0]] * 100,
                    "host_rate_expected": float(subset.loc[SCENARIOS[1], "expected_cases"]),
                    "host_rate_percent": float(subset.loc[SCENARIOS[1], "expected_cases"]) / scenario_totals[SCENARIOS[1]] * 100,
                }
            )
    return pd.DataFrame(rows)


def equal_earth(lon: float, lat: float) -> tuple[float, float]:
    longitude = math.radians(lon)
    latitude = math.radians(max(-89.999, min(89.999, lat)))
    a1, a2, a3, a4 = 1.340264, -0.081106, 0.000893, 0.003796
    theta = math.asin(math.sqrt(3) / 2 * math.sin(latitude))
    theta2 = theta * theta
    theta6 = theta2 * theta2 * theta2
    denominator = 3 * (9 * a4 * theta6 * theta2 + 7 * a3 * theta6 + 3 * a2 * theta2 + a1)
    x = 2 * math.sqrt(3) * longitude * math.cos(theta) / denominator
    y = theta * (a4 * theta6 * theta2 + a3 * theta6 + a2 * theta2 + a1)
    return x, y


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def make_population_map(host_population: pd.DataFrame) -> None:
    geojson = json.loads(
        (RAW / "geospatial" / "ne_110m_admin_0_countries.geojson").read_text(encoding="utf-8")
    )
    values = dict(zip(host_population["host_iso"], host_population["population"]))
    thresholds = [1, 10_000, 50_000, 200_000, 500_000, 1_000_000]
    colors = ["#E9F3F6", "#C8E3EA", "#94CCD9", "#58AEC2", "#287E9A", "#0B4F6C"]
    labels = ["1-9,999", "10k-49,999", "50k-199,999", "200k-499,999", "500k-999,999", "1m+"]
    width, height = 3000, 1675
    map_left, map_top, map_right, map_bottom = 80, 45, 2920, 1325
    x_extreme, _ = equal_earth(180, 0)
    x_min, x_max = -abs(x_extreme), abs(x_extreme)
    _, y_north = equal_earth(0, 89.999)
    _, y_south = equal_earth(0, -89.999)

    def project(lon: float, lat: float) -> tuple[float, float]:
        x, y = equal_earth(lon, lat)
        return (
            map_left + (x - x_min) / (x_max - x_min) * (map_right - map_left),
            map_top + (y_north - y) / (y_north - y_south) * (map_bottom - map_top),
        )

    def feature_iso(properties: dict[str, object]) -> str:
        code = str(properties.get("ADM0_A3") or properties.get("ISO_A3") or "")
        return {"PSX": "PSE", "KOS": "XKX", "SDS": "SSD", "SAH": "ESH"}.get(code, code)

    def color_for(value: float | None) -> str:
        if value is None or value < 1:
            return "#EEEEEE"
        index = max(i for i, threshold in enumerate(thresholds) if value >= threshold)
        return colors[index]

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    svg_paths: list[str] = []
    for feature in geojson["features"]:
        if feature["properties"].get("CONTINENT") == "Antarctica":
            continue
        fill = color_for(values.get(feature_iso(feature["properties"])))
        geometry = feature["geometry"]
        polygon_groups = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygon_groups:
            if not polygon:
                continue
            points = [project(float(lon), float(lat)) for lon, lat in polygon[0]]
            if len(points) < 3:
                continue
            draw.polygon(points, fill=fill, outline="#FFFFFF", width=2)
            path_data = " ".join(
                ("M" if index == 0 else "L") + f"{x / 2:.2f},{y / 2:.2f}"
                for index, (x, y) in enumerate(points)
            ) + " Z"
            svg_paths.append(f'<path d="{path_data}" fill="{fill}" stroke="#FFFFFF" stroke-width="0.55"/>')

    draw.text((110, 1360), "Average analyzed refugee population, 2024-2025", fill="#1A1A1A", font=font(45, True))
    legend_entries = [("#EEEEEE", "<1 or none"), *zip(colors, labels)]
    x_start, y_start = 120, 1465
    for index, (color, label) in enumerate(legend_entries):
        x = x_start + index * 410
        draw.rounded_rectangle((x, y_start, x + 52, y_start + 52), radius=4, fill=color)
        draw.text((x + 66, y_start + 7), label, fill="#444444", font=font(26))
    image.save(FIGURES / "optional_world_map_midpoint_population.png", dpi=(300, 300), optimize=True)
    image.save(
        FIGURES / "optional_world_map_midpoint_population.tiff",
        dpi=(300, 300),
        compression="tiff_lzw",
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="825" viewBox="0 0 1500 825">'
        '<rect width="1500" height="825" fill="white"/>'
        + "".join(svg_paths)
        + "</svg>"
    )
    (FIGURES / "optional_world_map_midpoint_population.svg").write_text(svg, encoding="utf-8")
    host_population.to_csv(FIGURES / "optional_world_map_midpoint_population_data.csv", index=False)


def direct_vs_five_year_check(
    countries: pd.DataFrame,
    direct_rates: pd.DataFrame,
) -> pd.DataFrame:
    exact_rows = []
    for analysis_age, age_groups in {
        "0-17": list(range(0, 4)),
        "18-59": list(range(4, 12)),
        "60+": list(range(12, 18)),
    }.items():
        frames = []
        for age_group in age_groups:
            frame = read_gco_json(
                RAW / "gco" / "2024" / "incidence" / f"age_{age_group:02d}.json"
            )
            frames.append(frame)
        temp = pd.concat(frames, ignore_index=True)
        for column in ("country_code", "sex", "cancer_code"):
            temp[column] = pd.to_numeric(temp[column], errors="raise").astype(int)
        for column in ("total", "total_pop"):
            temp[column] = pd.to_numeric(temp[column], errors="coerce")
        temp = temp.loc[(temp["country_code"] < 900) & temp["cancer_code"].between(1, 40)]
        grouped = temp.groupby(["country_code", "sex", "cancer_code"], as_index=False)[
            ["total", "total_pop"]
        ].sum()
        grouped["analysis_age"] = analysis_age
        grouped["reconstructed_rate"] = grouped["total"] / grouped["total_pop"] * 100_000
        exact_rows.append(grouped)
    reconstructed = pd.concat(exact_rows, ignore_index=True).merge(
        countries[["country_code", "country_iso3"]],
        on="country_code",
        how="inner",
        validate="many_to_one",
    )
    comparison = direct_rates.merge(
        reconstructed[
            ["country_iso3", "sex", "cancer_code", "analysis_age", "reconstructed_rate"]
        ],
        on=["country_iso3", "sex", "cancer_code", "analysis_age"],
        how="left",
        validate="one_to_one",
    )
    comparison["absolute_rate_difference"] = (
        comparison["rate_per_100k"] - comparison["reconstructed_rate"]
    ).abs()
    return comparison


def totals_for_cells(cells: pd.DataFrame, rates: pd.DataFrame) -> dict[str, float]:
    origin = apply_all_cancer(cells, rates, "origin", SCENARIOS[0])
    host = apply_all_cancer(cells, rates, "host", SCENARIOS[1])
    return {
        "analyzed_population": float(cells["population"].sum()),
        "origin_rate_expected": float(origin["expected_cases"].sum()),
        "host_rate_expected": float(host["expected_cases"].sum()),
    }


def main() -> None:
    demographics, stocks, _ = load_unhcr()
    countries, cancers = load_gco_metadata()
    direct_rates = load_direct_rates(countries)
    boundary_rates = load_boundary_sensitivity_rates(countries)
    host_burden = load_host_denominators(countries)
    demographics = add_crosswalk_flags(demographics, countries)

    ref_rows = demographics.loc[demographics["pop_type"] == "REF"].copy()
    analytic_rows = ref_rows.loc[ref_rows["analysis_geography"]].copy()
    cells_by_year = make_observed_cells(analytic_rows, weight=1.0)
    midpoint_cells = make_observed_cells(analytic_rows, weight=0.5)
    midpoint_cells.to_csv(PROCESSED / "unhcr_ref_midpoint_weighted_observed_cells.csv", index=False)
    cells_by_year.to_csv(PROCESSED / "unhcr_ref_observed_cells_by_year.csv", index=False)
    midpoint_cell_table(cells_by_year).to_csv(
        PROCESSED / "unhcr_ref_midpoint_observed_cells.csv", index=False
    )
    direct_rates.to_csv(PROCESSED / "gco_2024_direct_broad_age_rates.csv", index=False)

    all_detail, site_detail, site_summary = analyze_cells(midpoint_cells, direct_rates, cancers)
    all_detail.to_csv(PROCESSED / "midpoint_all_cancer_cell_estimates.csv", index=False)
    site_detail.to_csv(PROCESSED / "midpoint_site_cell_estimates.csv", index=False)
    site_summary.to_csv(TABLES / "expected_by_cancer_site.csv", index=False)
    aggregates = aggregate_all_cancer(all_detail)
    aggregates["global"].to_csv(TABLES / "global_expected_cases.csv", index=False)
    aggregates["age_sex"].to_csv(TABLES / "expected_by_age_sex.csv", index=False)
    aggregates["origin"].to_csv(TABLES / "expected_by_origin.csv", index=False)
    aggregates["host"].to_csv(TABLES / "expected_by_host.csv", index=False)

    year_results = []
    year_detail_frames = []
    for year in YEARS:
        year_cells = cells_by_year.loc[cells_by_year["year"] == year].copy()
        year_all, _, _ = analyze_cells(year_cells, direct_rates, cancers)
        year_all["population_snapshot_year"] = year
        year_detail_frames.append(year_all)
        values = year_all.groupby("scenario")[["population", "expected_cases"]].sum()
        for scenario in SCENARIOS:
            year_results.append(
                {
                    "population_snapshot_year": year,
                    "scenario": scenario,
                    "analyzed_population": float(values.loc[scenario, "population"]),
                    "expected_cases": float(values.loc[scenario, "expected_cases"]),
                }
            )
    year_results_frame = pd.DataFrame(year_results)
    year_results_frame.to_csv(TABLES / "expected_cases_by_population_snapshot_year.csv", index=False)
    pd.concat(year_detail_frames, ignore_index=True).to_csv(
        PROCESSED / "all_cancer_cell_estimates_by_year.csv", index=False
    )

    coverage_rows = []
    for year in YEARS:
        rows = ref_rows.loc[ref_rows["year"] == year]
        geographic = rows.loc[
            rows["origin_defined"] & rows["host_defined"] & rows["distinct_origin_host"]
        ]
        included = rows.loc[rows["analysis_geography"]]
        coverage_rows.append(
            {
                "population_snapshot": f"End-{year}",
                "stock_population": stocks[year],
                "demographic_file_row_total": float(rows["row_total"].sum()),
                "reported_age_sex_population": float(rows["observed_age_sex"].sum()),
                "defined_distinct_origin_host_population": float(geographic["observed_age_sex"].sum()),
                "analytic_population": float(included["observed_age_sex"].sum()),
                "reported_age_sex_coverage_pct": pct(rows["observed_age_sex"].sum(), stocks[year]),
                "analytic_coverage_pct": pct(included["observed_age_sex"].sum(), stocks[year]),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    numeric_counts = [
        "stock_population",
        "demographic_file_row_total",
        "reported_age_sex_population",
        "defined_distinct_origin_host_population",
        "analytic_population",
    ]
    average_counts = coverage[numeric_counts].mean()
    average_row = {"population_snapshot": "Average 2024-2025", **average_counts.to_dict()}
    average_row["reported_age_sex_coverage_pct"] = pct(
        average_row["reported_age_sex_population"], average_row["stock_population"]
    )
    average_row["analytic_coverage_pct"] = pct(
        average_row["analytic_population"], average_row["stock_population"]
    )
    coverage = pd.concat([coverage, pd.DataFrame([average_row])], ignore_index=True)
    coverage.to_csv(TABLES / "population_coverage.csv", index=False)

    flow = pd.DataFrame(
        [
            {"stage": "Average end-2024/end-2025 REF stock", "population": average_row["stock_population"]},
            {"stage": "Average reported age-sex population", "population": average_row["reported_age_sex_population"]},
            {"stage": "Average reported cells with defined, distinct origin and host", "population": average_row["defined_distinct_origin_host_population"]},
            {"stage": "Average analytic population with both countries matched to GLOBOCAN", "population": average_row["analytic_population"]},
        ]
    )
    flow["percent_of_average_ref_stock"] = flow["population"] / average_row["stock_population"] * 100
    flow.to_csv(TABLES / "population_flow.csv", index=False)

    exclusions = ref_rows.loc[
        ref_rows["observed_age_sex"] > 0,
        [
            "year", "row_id", "origin", "origin_iso", "host", "host_iso", "row_total",
            "observed_age_sex", "row_complete", "origin_gco_match", "host_gco_match",
            "distinct_origin_host", "analysis_geography", "exclusion_reason",
        ],
    ].copy()
    exclusions.to_csv(TABLES / "country_crosswalk_and_exclusions.csv", index=False)

    host_service = aggregates["host"].merge(
        host_burden[["country_iso3", "gco_host_cancer_cases"]],
        left_on="host_iso",
        right_on="country_iso3",
        how="left",
        validate="many_to_one",
    )
    require(host_service["gco_host_cancer_cases"].notna().all(), "Missing host national totals")
    host_service["estimated_service_need_share_pct"] = (
        host_service["expected_cases"] / host_service["gco_host_cancer_cases"] * 100
    )
    host_service.to_csv(TABLES / "host_service_need_share.csv", index=False)

    table_1 = build_table_1(all_detail, site_summary, host_burden)
    table_2 = build_table_2(all_detail)
    table_1.to_csv(TABLES / "table_1_main.csv", index=False)
    table_2.to_csv(TABLES / "table_2_age_sex.csv", index=False)

    host_population = aggregates["host"].loc[
        aggregates["host"]["scenario"] == SCENARIOS[0], ["host", "host_iso", "population"]
    ].copy()
    make_population_map(host_population)

    strict_rows = analytic_rows.loc[analytic_rows["row_complete"]]
    strict_cells = make_observed_cells(strict_rows, weight=0.5)
    detailed_pairs = (
        analytic_rows.loc[analytic_rows["observed_age_sex"] > 0]
        .groupby(["origin_iso", "host_iso"])["year"]
        .nunique()
    )
    paired_pairs = set(detailed_pairs.loc[detailed_pairs == 2].index)
    paired_mask = analytic_rows.set_index(["origin_iso", "host_iso"]).index.isin(paired_pairs)
    paired_cells = make_observed_cells(analytic_rows.loc[paired_mask], weight=0.5)

    primary_values = totals_for_cells(midpoint_cells, direct_rates)
    sensitivity_specs = [
        ("Midpoint of end-2024 and end-2025 observed cells", midpoint_cells, direct_rates),
        ("End-2024 observed cells only", cells_by_year.loc[cells_by_year["year"] == 2024], direct_rates),
        ("End-2025 observed cells only", cells_by_year.loc[cells_by_year["year"] == 2025], direct_rates),
        ("Origin-host records with demographic detail in both years", paired_cells, direct_rates),
        ("Strictly complete demographic rows", strict_cells, direct_rates),
        ("Aligned 0-17/18-59 age-boundary rates", midpoint_cells, boundary_rates),
    ]
    sensitivity_rows = []
    for analysis, cells, rates in sensitivity_specs:
        values = totals_for_cells(cells, rates)
        for scenario, key in [
            (SCENARIOS[0], "origin_rate_expected"),
            (SCENARIOS[1], "host_rate_expected"),
        ]:
            sensitivity_rows.append(
                {
                    "analysis": analysis,
                    "scenario": scenario,
                    "analyzed_population": values["analyzed_population"],
                    "coverage_of_average_ref_stock_pct": pct(
                        values["analyzed_population"], average_row["stock_population"]
                    ),
                    "expected_cases": values[key],
                    "change_from_midpoint_analysis_pct": (
                        values[key] / primary_values[key] - 1
                    ) * 100,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(TABLES / "sensitivity_analyses.csv", index=False)

    comparison = direct_vs_five_year_check(countries, direct_rates)
    comparison.to_csv(QA / "gco_direct_vs_five_year_rate_check.csv", index=False)

    validations: list[dict[str, object]] = []

    def record(name: str, passed: bool, observed: object, expected: object, tolerance: str = "") -> None:
        validations.append(
            {
                "check": name,
                "passed": bool(passed),
                "observed": observed,
                "expected": expected,
                "tolerance": tolerance,
            }
        )

    record("UNHCR years", set(ref_rows["year"]) == set(YEARS), str(set(ref_rows["year"])), str(set(YEARS)))
    record("Population type", set(ref_rows["pop_type"]) == {"REF"}, str(set(ref_rows["pop_type"])), "{'REF'}")
    record("All counts nonnegative", (ref_rows[UNHCR_NUMERIC_COLUMNS[:-1] + ["row_total"]] >= 0).all().all(), True, True)
    record("Included origins differ from hosts", analytic_rows["distinct_origin_host"].all(), True, True)
    record("Both included countries matched", analytic_rows["origin_gco_match"].all() and analytic_rows["host_gco_match"].all(), True, True)
    for year in YEARS:
        year_rows = ref_rows.loc[ref_rows["year"] == year]
        record(
            f"{year} demographic total reconciles to stock",
            np.isclose(year_rows["row_total"].sum(), stocks[year]),
            year_rows["row_total"].sum(),
            stocks[year],
            "1e-6",
        )
        included_total = analytic_rows.loc[analytic_rows["year"] == year, "observed_age_sex"].sum()
        cell_total = cells_by_year.loc[cells_by_year["year"] == year, "population"].sum()
        record(f"{year} observed-cell reconciliation", np.isclose(included_total, cell_total), cell_total, included_total, "1e-6")

    record(
        "Midpoint population identity",
        np.isclose(midpoint_cells["population"].sum(), cells_by_year["population"].sum() / 2),
        midpoint_cells["population"].sum(),
        cells_by_year["population"].sum() / 2,
        "1e-8",
    )
    record(
        "Same midpoint population in both scenarios",
        all_detail.groupby("scenario")["population"].sum().nunique() == 1,
        all_detail.groupby("scenario")["population"].sum().to_dict(),
        "identical",
    )
    record("No missing all-cancer rates", all_detail["rate_per_100k"].notna().all(), int(all_detail["rate_per_100k"].isna().sum()), 0)
    record(
        "Direct broad-age endpoint reconciliation",
        comparison["absolute_rate_difference"].max() < 0.02,
        comparison["absolute_rate_difference"].max(),
        "<0.02 per 100,000",
        "absolute",
    )
    record(
        "GLOBOCAN national/world code-40 totals",
        abs(host_burden.attrs["country_sum"] / host_burden.attrs["world_total"] - 1) < 0.001,
        host_burden.attrs["country_sum"],
        host_burden.attrs["world_total"],
        "0.1% aggregate rounding",
    )

    midpoint_expected = aggregates["global"].set_index("scenario")["expected_cases"].to_dict()
    for scenario in SCENARIOS:
        year_average = year_results_frame.loc[
            year_results_frame["scenario"] == scenario, "expected_cases"
        ].mean()
        record(
            f"{scenario}: midpoint equals average of yearly estimates",
            np.isclose(midpoint_expected[scenario], year_average, atol=1e-8),
            midpoint_expected[scenario],
            year_average,
            "1e-8 cases",
        )
        global_total = midpoint_expected[scenario]
        for label, table in [
            ("age-sex", aggregates["age_sex"]),
            ("origin", aggregates["origin"]),
            ("host", aggregates["host"]),
        ]:
            subtotal = table.loc[table["scenario"] == scenario, "expected_cases"].sum()
            record(
                f"{scenario}: {label} aggregation",
                np.isclose(global_total, subtotal, atol=1e-8),
                subtotal,
                global_total,
                "1e-8 cases",
            )
        site_total = site_summary.loc[site_summary["scenario"] == scenario, "expected_cases"].sum()
        record(
            f"{scenario}: mutually exclusive site reconciliation",
            abs(site_total / global_total - 1) < 0.002,
            site_total,
            global_total,
            "0.2% independent GLOBOCAN rounding",
        )
        original_codes = site_detail.loc[
            (site_detail["scenario"] == scenario) & site_detail["cancer_code"].isin([8, 9, 10]),
            "expected_cases",
        ].sum()
        custom = site_summary.loc[
            (site_summary["scenario"] == scenario)
            & site_summary["cancer_code"].isin([CUSTOM_COLORECTAL_CODE, 10]),
            "expected_cases",
        ].sum()
        record(
            f"{scenario}: colorectal plus anus reconciliation",
            np.isclose(original_codes, custom, atol=1e-10),
            custom,
            original_codes,
            "1e-10 cases",
        )

    old_2025 = year_results_frame.set_index(["population_snapshot_year", "scenario"])["expected_cases"]
    record(
        "Prior verified 2025 origin-rate result reproduced",
        np.isclose(old_2025.loc[(2025, SCENARIOS[0])], 23552.1609, atol=0.001),
        old_2025.loc[(2025, SCENARIOS[0])],
        23552.1609,
        "0.001 case",
    )
    record(
        "Prior verified 2025 host-rate result reproduced",
        np.isclose(old_2025.loc[(2025, SCENARIOS[1])], 32363.4557, atol=0.001),
        old_2025.loc[(2025, SCENARIOS[1])],
        32363.4557,
        "0.001 case",
    )

    manual = all_detail.loc[all_detail["scenario"] == SCENARIOS[0]].nlargest(1, "population")[
        [
            "year", "origin", "origin_iso", "host", "host_iso", "sex_label", "analysis_age",
            "population", "rate_per_100k", "expected_cases",
        ]
    ].copy()
    manual["independent_recalculation"] = manual["population"] * manual["rate_per_100k"] / 100_000
    manual["difference"] = manual["expected_cases"] - manual["independent_recalculation"]
    manual.to_csv(QA / "manual_calculation_check.csv", index=False)
    record("Manual population x rate / 100,000 check", manual["difference"].abs().max() < 1e-12, manual["difference"].abs().max(), 0, "1e-12")

    validation_frame = pd.DataFrame(validations)
    validation_frame.to_csv(QA / "validation_checks.csv", index=False)
    require(validation_frame["passed"].all(), "At least one validation check failed")

    global_dict = aggregates["global"].set_index("scenario").to_dict(orient="index")
    headline = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "population_definition": "Mean of published end-2024 and end-2025 UNHCR REF age-sex counts; defined, distinct, GLOBOCAN-matched origin and host",
        "unhcr_ref_stock_end_2024": stocks[2024],
        "unhcr_ref_stock_end_2025": stocks[2025],
        "unhcr_ref_stock_average": average_row["stock_population"],
        "reported_age_sex_population_average": average_row["reported_age_sex_population"],
        "analytic_population_average": average_row["analytic_population"],
        "analytic_coverage_pct": average_row["analytic_coverage_pct"],
        "origin_rate_expected_cases": global_dict[SCENARIOS[0]]["expected_cases"],
        "host_rate_expected_cases": global_dict[SCENARIOS[1]]["expected_cases"],
        "host_vs_origin_percent_difference": (
            global_dict[SCENARIOS[1]]["expected_cases"] / global_dict[SCENARIOS[0]]["expected_cases"] - 1
        ) * 100,
        "gco_2024_world_cases_excluding_nmsc": host_burden.attrs["world_total"],
        "leading_sites_by_scenario": {
            scenario: site_summary.loc[
                (site_summary["scenario"] == scenario)
                & (site_summary["site_category"] == "Named cancer site")
            ].head(10)[
                ["cancer_site", "expected_cases", "percent_of_all_cancers", "rank_named_site"]
            ].to_dict(orient="records")
            for scenario in SCENARIOS
        },
        "top_origins_by_scenario": {
            scenario: aggregates["origin"].loc[
                aggregates["origin"]["scenario"] == scenario
            ].nsmallest(10, "rank").to_dict(orient="records")
            for scenario in SCENARIOS
        },
        "top_hosts_by_scenario": {
            scenario: aggregates["host"].loc[
                aggregates["host"]["scenario"] == scenario
            ].nsmallest(10, "rank").to_dict(orient="records")
            for scenario in SCENARIOS
        },
        "age_sex_results": aggregates["age_sex"].to_dict(orient="records"),
        "sensitivity_results": sensitivity.to_dict(orient="records"),
    }
    (TABLES / "headline_results.json").write_text(json.dumps(headline, indent=2), encoding="utf-8")

    source_files = [path for path in RAW.rglob("*") if path.is_file()]
    pd.DataFrame(
        [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(source_files)
        ]
    ).to_csv(DOCUMENTATION / "source_file_checksums.csv", index=False)
    (DOCUMENTATION / "session_info.txt").write_text(
        f"Generated UTC: {datetime.now(timezone.utc).isoformat()}\n"
        f"Python: {sys.version}\n"
        f"Platform: {platform.platform()}\n"
        f"pandas: {pd.__version__}\n"
        f"numpy: {np.__version__}\n"
        f"Pillow: {PIL_VERSION}\n",
        encoding="utf-8",
    )
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
