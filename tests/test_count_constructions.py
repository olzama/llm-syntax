import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import pytest
from unittest.mock import patch

# A minimal stand-in for derivation.UDFNode so tests don't depend on delphin internals.
class MockNode:
    def __init__(self, entity, daughters=None):
        self.entity = entity
        self.daughters = daughters or []

# Patch derivation.UDFNode in count_constructions so isinstance checks use our mock.
import count_constructions
count_constructions.derivation.UDFNode = MockNode

from count_constructions import traverse_derivation


def empty_types():
    return {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_regular_entity_classified_as_constr():
    root = MockNode('root', daughters=[MockNode('hcomp')])
    types = empty_types()
    traverse_derivation(root, types, preterminals=set(), lex={}, depth=1)
    assert types['constr'] == {'hcomp': 1}

def test_entity_ending_lr_classified_as_lexrule():
    root = MockNode('root', daughters=[MockNode('third_sg_fin_lr')])
    types = empty_types()
    traverse_derivation(root, types, preterminals=set(), lex={}, depth=1)
    assert types['lexrule'] == {'third_sg_fin_lr': 1}

def test_preterminal_without_supertypes_keeps_original_entity():
    root = MockNode('root', daughters=[MockNode('n_proper_le')])
    types = empty_types()
    with patch('count_constructions.get_n_supertypes', return_value=None):
        traverse_derivation(root, types, preterminals={'n_proper_le'}, lex={}, depth=1)
    assert types['lextype'] == {'n_proper_le': 1}

def test_preterminal_with_supertypes_uses_supertype():
    root = MockNode('root', daughters=[MockNode('n_proper_le')])
    types = empty_types()
    with patch('count_constructions.get_n_supertypes', return_value={0: {'n_-_pn_le'}}):
        traverse_derivation(root, types, preterminals={'n_proper_le'}, lex={}, depth=1)
    assert types['lextype'] == {'n_-_pn_le': 1}

# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def test_same_type_counted_multiple_times():
    root = MockNode('root', daughters=[MockNode('hcomp'), MockNode('hcomp')])
    types = empty_types()
    traverse_derivation(root, types, preterminals=set(), lex={}, depth=1)
    assert types['constr']['hcomp'] == 2

def test_different_types_counted_independently():
    root = MockNode('root', daughters=[
        MockNode('hcomp'), MockNode('hspec'), MockNode('third_sg_fin_lr')
    ])
    types = empty_types()
    traverse_derivation(root, types, preterminals=set(), lex={}, depth=1)
    assert types['constr'] == {'hcomp': 1, 'hspec': 1}
    assert types['lexrule'] == {'third_sg_fin_lr': 1}

# ---------------------------------------------------------------------------
# Recursion
# ---------------------------------------------------------------------------

def test_nested_daughters_are_traversed():
    grandchild = MockNode('hcomp')
    child = MockNode('hspec', daughters=[grandchild])
    root = MockNode('root', daughters=[child])
    types = empty_types()
    traverse_derivation(root, types, preterminals=set(), lex={}, depth=1)
    assert types['constr'] == {'hspec': 1, 'hcomp': 1}

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_daughters_no_error():
    root = MockNode('root', daughters=[])
    types = empty_types()
    traverse_derivation(root, types, preterminals=set(), lex={}, depth=1)
    assert types == empty_types()

def test_already_visited_root_skipped():
    root = MockNode('root', daughters=[MockNode('hcomp')])
    types = empty_types()
    visited = {id(root)}
    traverse_derivation(root, types, preterminals=set(), lex={}, depth=1, visited=visited)
    assert types == empty_types()
