import csv

from synthetic_swipes import FIELDNAMES, generate_rows, write_csv


def test_generator_is_deterministic_and_fully_synthetic():
    first = generate_rows(users=4, profiles=8, interactions_per_user=4, seed=7)
    second = generate_rows(users=4, profiles=8, interactions_per_user=4, seed=7)

    assert first == second
    assert len(first) == 16
    assert {row["decidermemberid"] for row in first} == {
        "u0001",
        "u0002",
        "u0003",
        "u0004",
    }
    assert all(str(row["othermemberid"]).startswith("p") for row in first)


def test_write_csv_emits_expected_schema(tmp_path):
    output = tmp_path / "synthetic.csv"
    rows = generate_rows(users=2, profiles=4, interactions_per_user=3)

    write_csv(output, rows)

    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == FIELDNAMES
        assert len(list(reader)) == 6
