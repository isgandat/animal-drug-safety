from datetime import date

import pytest

from etl.parse import (Rejected, age_in_years, normalize_ingredient, parse_many, parse_record,
                       weight_in_kg)
from tests.conftest import raw_record


@pytest.mark.parametrize("age, expected", [
    ({"min": "6", "unit": "Month"}, 0.5),
    ({"min": "2.00", "unit": "Year"}, 2.0),
    ({"min": "52", "unit": "Week"}, 0.9966),
    ({"min": "200", "unit": "Year"}, None),       # implausible
    ({"min": "3", "unit": "Fortnight"}, None),    # unknown unit
    ({"min": "-1", "unit": "Year"}, None),
    ({"qualifier": "Unknown"}, None),
    (None, None),
])
def test_age_is_converted_to_years(age, expected):
    assert age_in_years(age) == expected


def test_weight_is_converted_to_kg():
    assert weight_in_kg({"min": "10", "unit": "Pound"}) == pytest.approx(4.536, abs=1e-3)
    assert weight_in_kg({"min": "0", "unit": "Kilogram"}) is None
    assert weight_in_kg({"min": "9000", "unit": "Kilogram"}) is None


@pytest.mark.parametrize("raw, expected", [
    ("Pyrantel Pamoate;Sarolaner", ["pyrantel", "sarolaner"]),
    ("Fluralaner 13.64% 12-Week Chew", ["fluralaner"]),
    ("Milbemycin Oxime/Lufenuron", ["milbemycin", "lufenuron"]),
    ("Insulin Injectable Vial", ["insulin"]),
    ("Meloxicam;Meloxicam", ["meloxicam"]),
    ("", []),
    (None, []),
])
def test_ingredient_names_are_normalised(raw, expected):
    assert normalize_ingredient(raw) == expected


def test_ingredient_normalisation_is_idempotent():
    for raw in ("Pyrantel As Pamoate Salt", "Oclacitinib Maleate", "Selamectin;Sarolaner"):
        once = normalize_ingredient(raw)
        assert [n for name in once for n in normalize_ingredient(name)] == once


def test_record_fields_are_typed_and_cleaned():
    rec = raw_record(serious_ae="true", treated_for_ae="false",
                     reaction=[{"veddra_term_name": "Vomiting"}, {"veddra_term_name": "Vomiting"},
                               {"veddra_term_name": "Lethargy"}])
    rec["drug"][0]["route"] = ["Oral", "Topical"]
    rec["drug"][0]["off_label_use"] = ["Overdosed", "Treatment regimen Off-Label"]
    parsed = parse_record(rec, "2023q1-0001")
    assert parsed.report["serious"] == 1 and parsed.report["treated"] == 0
    assert parsed.report["receive_date"] == "2023-01-15" and parsed.report["receive_year"] == 2023
    assert parsed.report["age_years"] == 5.0 and parsed.report["health_prior"] == "Good"
    assert parsed.drugs[0]["route"] == "Oral|Topical"
    assert parsed.drugs[0]["off_label"] == "Overdosed|Treatment regimen Off-Label"
    assert [i["name"] for i in parsed.ingredients] == ["meloxicam"]
    assert [r["term"] for r in parsed.reactions] == ["Vomiting", "Lethargy"]  # duplicate dropped


def test_merged_record_with_list_values_is_kept():
    rec = raw_record(original_receive_date=["20161102", "20161101"],
                     primary_reporter=["Veterinarian", "Veterinarian"])
    parsed = parse_record(rec)
    assert parsed.report["receive_date"] == "2016-11-01"
    assert parsed.report["reporter"] == "Veterinarian"


@pytest.mark.parametrize("overrides, reason", [
    ({"unique_aer_id_number": None}, "missing_id"),
    ({"original_receive_date": "2023-01-15"}, "bad_receive_date"),
    ({"original_receive_date": "20230230"}, "bad_receive_date"),
    ({"original_receive_date": "20990101"}, "receive_date_out_of_range"),
])
def test_bad_records_are_rejected_with_a_reason(overrides, reason):
    with pytest.raises(Rejected, match=reason):
        parse_record(raw_record(**overrides), today=date(2026, 1, 1))


def test_parse_many_counts_rejections_and_keeps_going():
    records = [raw_record("A-1"), raw_record(None), raw_record("A-3", animal="not a dict")]
    good, rejected = parse_many(records)
    assert [p.report["aer_id"] for p in good] == ["A-1"]
    assert rejected == {"missing_id": 1, "malformed": 1}
