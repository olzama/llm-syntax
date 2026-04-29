import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import pytest
from util import compute_ttr


def _make_frequencies():
    return {
        'constr': {
            'model_a': {'type1': 10, 'type2': 5, 'type3': 0},
            'model_b': {'type1': 20},
        },
        'lextype': {
            'model_a': {'lt1': 3, 'lt2': 7},
        },
    }


def test_compute_ttr_token_count():
    """Token count sums all values including zeros."""
    result = compute_ttr(_make_frequencies())
    actual = result['constr']['model_a']['tokens']
    print(f"\n  Input:    model_a constr counts = {{'type1': 10, 'type2': 5, 'type3': 0}}")
    print(f"  Expected: tokens = 15")
    print(f"  Actual:   tokens = {actual}")
    assert actual == 15


def test_compute_ttr_type_count_excludes_zeros():
    """Types with count 0 are not counted."""
    result = compute_ttr(_make_frequencies())
    actual = result['constr']['model_a']['types']
    print(f"\n  Input:    model_a constr counts = {{'type1': 10, 'type2': 5, 'type3': 0}}")
    print(f"  Expected: types = 2  (type3 excluded because count=0)")
    print(f"  Actual:   types = {actual}")
    assert actual == 2


def test_compute_ttr_value():
    result = compute_ttr(_make_frequencies())
    actual = result['constr']['model_a']['ttr']
    print(f"\n  Input:    model_a: 2 non-zero types, 15 tokens")
    print(f"  Expected: ttr = {2/15:.6f}")
    print(f"  Actual:   ttr = {actual:.6f}")
    assert actual == pytest.approx(2 / 15)


def test_compute_ttr_single_type():
    result = compute_ttr(_make_frequencies())
    actual = result['constr']['model_b']
    print(f"\n  Input:    model_b constr counts = {{'type1': 20}}")
    print(f"  Expected: types=1, tokens=20, ttr={1/20:.5f}")
    print(f"  Actual:   {actual}")
    assert actual['types']  == 1
    assert actual['tokens'] == 20
    assert actual['ttr']    == pytest.approx(1 / 20)


def test_compute_ttr_empty_model():
    """A model with no types produces ttr of 0."""
    data = {'constr': {'empty': {}}}
    result = compute_ttr(data)
    actual = result['constr']['empty']
    print(f"\n  Input:    model 'empty' with no types")
    print(f"  Expected: types=0, tokens=0, ttr=0.0")
    print(f"  Actual:   {actual}")
    assert actual['ttr']    == 0.0
    assert actual['types']  == 0
    assert actual['tokens'] == 0


def test_compute_ttr_phenomena_filter():
    """Only requested phenomena appear in the result."""
    result = compute_ttr(_make_frequencies(), phenomena=['constr'])
    print(f"\n  Input:    phenomena=['constr'], data has constr + lextype")
    print(f"  Expected: only 'constr' in result")
    print(f"  Actual:   keys = {set(result.keys())}")
    assert 'lextype' not in result
    assert 'constr'  in result


def test_compute_ttr_all_phenomena_by_default():
    result = compute_ttr(_make_frequencies())
    print(f"\n  Input:    no phenomena filter, data has constr + lextype")
    print(f"  Expected: keys = {{'constr', 'lextype'}}")
    print(f"  Actual:   keys = {set(result.keys())}")
    assert set(result.keys()) == {'constr', 'lextype'}


def test_compute_ttr_unknown_phenomenon_skipped():
    """Phenomena not present in frequencies are silently skipped."""
    result = compute_ttr(_make_frequencies(), phenomena=['constr', 'nonexistent'])
    print(f"\n  Input:    phenomena=['constr', 'nonexistent']")
    print(f"  Expected: 'nonexistent' absent, 'constr' present")
    print(f"  Actual:   keys = {set(result.keys())}")
    assert 'nonexistent' not in result
    assert 'constr'      in result
