import csv
from pathlib import Path
import subprocess
import sys

import pytest

from recommender import load_likes

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ['decidermemberid', 'othermemberid', 'timestamp', 'like']


def write_rows(tmp_path, rows):
    path = tmp_path / 'interactions.csv'
    with path.open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    return path


def test_numeric_looking_ids_and_na_are_preserved(tmp_path):
    path = write_rows(tmp_path, [['001', '007', '2026-01-01', 1], ['NA', '008', '2026-01-02', 1]])
    likes = load_likes(path)
    assert set(likes.decidermemberid) == {'001', 'NA'}
    assert set(likes.othermemberid) == {'007', '008'}


@pytest.mark.parametrize('row, field', [
    (['', 'p1', '2026-01-01', 1], 'decidermemberid'),
    (['u1', ' ', '2026-01-01', 1], 'othermemberid'),
    (['u1', 'p1', 'not-a-date', 1], 'timestamp'),
    (['u1', 'p1', '2026-01-01', 'maybe'], 'like'),
    (['u1', 'p1', '2026-01-01', 1.8], 'like'),
])
def test_bad_rows_are_rejected_with_row_and_field_not_silently_removed(tmp_path, row, field):
    path = write_rows(tmp_path, [row, ['good', 'p2', '2026-01-02', 1]])
    with pytest.raises(ValueError, match=rf'CSV row.*2.*{field}'):
        load_likes(path)


def test_summary_accounts_for_negatives_and_duplicate_likes(tmp_path):
    path = write_rows(tmp_path, [['u1', 'p1', '2026-01-01', 1],
                                 ['u1', 'p1', '2026-01-02', 1],
                                 ['u1', 'p2', '2026-01-03', 0]])
    likes = load_likes(path)
    assert likes.attrs['input_summary'] == {'input_rows': 3, 'negative_rows': 1, 'duplicate_positive_rows': 1, 'retained_positive_rows': 1}


def cli(*args, cwd=None):
    return subprocess.run([sys.executable, str(ROOT / 'recommender.py'), *args],
                          cwd=cwd, capture_output=True, text=True, timeout=60)


def test_default_sample_works_outside_repo_and_recommendations_exclude_all_seen_items(tmp_path):
    result = cli('recommend', '--user-id', 'u1', '--top-k', '10', cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'p4 ' in result.stdout
    assert all(f'{item}  score=' not in result.stdout for item in ('p1', 'p2', 'p3'))


def test_unknown_user_gets_disclosed_popularity_fallback():
    result = cli('recommend', '--user-id', 'new-user', '--top-k', '2')
    assert result.returncode == 0, result.stderr
    assert 'popularity' in result.stdout.lower()


def test_nonpositive_components_are_an_argument_error():
    result = cli('--components', '0', 'evaluate')
    assert result.returncode == 2
    assert 'positive' in result.stderr
