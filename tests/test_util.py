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
    assert result['constr']['model_a']['tokens'] == 15


def test_compute_ttr_type_count_excludes_zeros():
    """Types with count 0 are not counted."""
    result = compute_ttr(_make_frequencies())
    assert result['constr']['model_a']['types'] == 2


def test_compute_ttr_value():
    result = compute_ttr(_make_frequencies())
    assert result['constr']['model_a']['ttr'] == pytest.approx(2 / 15)


def test_compute_ttr_single_type():
    result = compute_ttr(_make_frequencies())
    assert result['constr']['model_b']['types'] == 1
    assert result['constr']['model_b']['tokens'] == 20
    assert result['constr']['model_b']['ttr'] == pytest.approx(1 / 20)


def test_compute_ttr_empty_model():
    """A model with no types produces ttr of 0."""
    data = {'constr': {'empty': {}}}
    result = compute_ttr(data)
    assert result['constr']['empty']['ttr'] == 0.0
    assert result['constr']['empty']['types'] == 0
    assert result['constr']['empty']['tokens'] == 0


def test_compute_ttr_phenomena_filter():
    """Only requested phenomena appear in the result."""
    result = compute_ttr(_make_frequencies(), phenomena=['constr'])
    assert 'lextype' not in result
    assert 'constr' in result


def test_compute_ttr_all_phenomena_by_default():
    result = compute_ttr(_make_frequencies())
    assert set(result.keys()) == {'constr', 'lextype'}


def test_compute_ttr_unknown_phenomenon_skipped():
    """Phenomena not present in frequencies are silently skipped."""
    result = compute_ttr(_make_frequencies(), phenomena=['constr', 'nonexistent'])
    assert 'nonexistent' not in result
    assert 'constr' in result
