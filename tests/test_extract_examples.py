import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import pytest
from extract_examples import collect_lextype_percentile_examples, find_constituent

# mini-multidir contains wsj-a/ and wsj-b/, each a copy of mini-wsj.
# Known derivation contents from test_count_constructions.py:
#
# "Not this year."  preterminals and their parents:
#   not_prdp   -> parent: aj-hd_scp-xp_c
#   this_det   -> parent: sp-hd_n_c
#   year_n1    -> parent: n_sg_ilr
#   period_pct -> parent: hd-pct_c
#
# "Champagne and dessert followed."  preterminals and their parents:
#   champagne_n1 -> parent: n_ms_ilr
#   dessert_n1   -> parent: n_ms-cnt_ilr
#   follow_v1    -> parent: v_pst_olr
#   period_pct   -> parent: (some construction)
#   and_conj     -> parent: (some construction)

FIXTURE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'mini-multidir')


# ---------------------------------------------------------------------------
# collect_lextype_percentile_examples
# ---------------------------------------------------------------------------

def test_collect_lextype_percentile_finds_known_parent():
    """n_sg_ilr is the parent of year_n1 in 'Not this year.' — should be found."""
    examples = {'10': ['n_sg_ilr']}
    result = collect_lextype_percentile_examples(FIXTURE_DATA_DIR, examples)
    sentences = [s for d in result.values() for s in d.get('n_sg_ilr', [])]
    print(f"\n  Input:    examples={examples}, data_dir={FIXTURE_DATA_DIR}")
    print(f"  Expected: sentences containing 'year'")
    print(f"  Actual:   {sentences}")
    assert any('year' in s.lower() for s in sentences)


def test_collect_lextype_percentile_finds_in_both_datasets():
    """Both wsj-a and wsj-b should each yield a result for n_sg_ilr."""
    examples = {'10': ['n_sg_ilr']}
    result = collect_lextype_percentile_examples(FIXTURE_DATA_DIR, examples)
    datasets_with_hits = [d for d, types in result.items() if 'n_sg_ilr' in types]
    print(f"\n  Input:    examples={examples}")
    print(f"  Expected: 2 datasets with hits")
    print(f"  Actual:   {sorted(datasets_with_hits)}")
    assert len(datasets_with_hits) == 2


def test_collect_lextype_percentile_absent_type_not_in_result():
    """A type not present in any derivation should not appear in the output."""
    examples = {'10': ['nonexistent_type_xyz']}
    result = collect_lextype_percentile_examples(FIXTURE_DATA_DIR, examples)
    all_types = {t for d in result.values() for t in d}
    print(f"\n  Input:    examples={examples}")
    print(f"  Expected: 'nonexistent_type_xyz' not in result")
    print(f"  Actual:   all types found = {all_types}")
    assert 'nonexistent_type_xyz' not in all_types


def test_collect_lextype_percentile_multiple_types():
    """Two known parent types from different sentences should both be found."""
    examples = {'10': ['n_sg_ilr', 'v_pst_olr']}
    result = collect_lextype_percentile_examples(FIXTURE_DATA_DIR, examples)
    all_types = {t for d in result.values() for t in d}
    print(f"\n  Input:    examples={examples}")
    print(f"  Expected: both 'n_sg_ilr' and 'v_pst_olr' in result")
    print(f"  Actual:   {sorted(all_types)}")
    assert 'n_sg_ilr'  in all_types
    assert 'v_pst_olr' in all_types


def test_collect_lextype_percentile_returns_dict_keyed_by_dataset():
    """Result is a dict of {dataset_name: {type: [sentences]}}."""
    examples = {'10': ['n_sg_ilr']}
    result = collect_lextype_percentile_examples(FIXTURE_DATA_DIR, examples)
    print(f"\n  Input:    examples={examples}")
    print(f"  Actual:   keys={sorted(result.keys())}")
    assert isinstance(result, dict)
    for dataset, types in result.items():
        assert isinstance(dataset, str)
        assert isinstance(types, dict)


def test_collect_lextype_percentile_ignores_non_directories(tmp_path):
    """Stray files in data_dir should not cause a crash."""
    (tmp_path / 'stray.txt').write_text('not a tsdb suite')
    result = collect_lextype_percentile_examples(str(tmp_path), {'10': ['n_sg_ilr']})
    print(f"\n  Input:    tmp_path with only a stray .txt file")
    print(f"  Expected: empty result, no crash")
    print(f"  Actual:   {result}")
    assert result == {}


# ---------------------------------------------------------------------------
# find_constituent — pure logic, tested with a mock lattice
# ---------------------------------------------------------------------------

class _MockToken:
    """Minimal stand-in for a YY token."""
    def __init__(self, char_start, char_end):
        self.lnk = type('lnk', (), {'data': (char_start, char_end)})()


class _MockLattice:
    def __init__(self, tokens):
        self.tokens = tokens


def test_find_constituent_single_token():
    lattice = _MockLattice([_MockToken(0, 3), _MockToken(4, 8), _MockToken(9, 13)])
    result = find_constituent(lattice, 0, 1, 'Not this year.')
    print(f"\n  Input:    span [0,1), text='Not this year.'")
    print(f"  Expected: 'Not'")
    print(f"  Actual:   '{result}'")
    assert result == 'Not'


def test_find_constituent_multi_token():
    lattice = _MockLattice([_MockToken(0, 3), _MockToken(4, 8), _MockToken(9, 13)])
    result = find_constituent(lattice, 0, 3, 'Not this year.')
    print(f"\n  Input:    span [0,3), text='Not this year.'")
    print(f"  Expected: 'Not this year'")
    print(f"  Actual:   '{result}'")
    assert result == 'Not this year'


def test_find_constituent_middle_token():
    lattice = _MockLattice([_MockToken(0, 3), _MockToken(4, 8), _MockToken(9, 13)])
    result = find_constituent(lattice, 1, 2, 'Not this year.')
    print(f"\n  Input:    span [1,2), text='Not this year.'")
    print(f"  Expected: 'this'")
    print(f"  Actual:   '{result}'")
    assert result == 'this'
