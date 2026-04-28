import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import pytest
from erg import get_n_supertypes, lexical_types, dict_to_latex_table, create_friendly_name
from erg import populate_type_defs, read_lexicon, types2defs

FIXTURE_GRAMMAR_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'mini-eng')
FIXTURE_LEXICON     = os.path.join(FIXTURE_GRAMMAR_DIR, 'lexicon.tdl')


# ---------------------------------------------------------------------------
# Helpers / mocks
# ---------------------------------------------------------------------------

class MockTDLType:
    """Minimal stand-in for a PyDelphin TDL TypeDefinition object."""
    def __init__(self, supertypes):
        self.supertypes = supertypes


class MockTDLTypeWithDoc:
    """Stand-in for a TDL object that has a documentation() method."""
    def __init__(self, doc):
        self._doc = doc

    def documentation(self):
        return self._doc


# ---------------------------------------------------------------------------
# get_n_supertypes
#
# Hierarchy used in these tests:
#
#   A  ->  B, C     (A has two immediate supertypes)
#   B  ->  D
#   C  ->  D
#   D  ->  (nothing in lex)
# ---------------------------------------------------------------------------

def _make_lex():
    return {
        'A': MockTDLType(['B', 'C']),
        'B': MockTDLType(['D']),
        'C': MockTDLType(['D']),
    }


def test_get_n_supertypes_unknown_type_returns_none():
    """A type name not present in lex returns None."""
    result = get_n_supertypes(_make_lex(), 'UNKNOWN', 1)
    assert result is None


def test_get_n_supertypes_n_zero_returns_none():
    """n=0 means no levels are requested; result is None."""
    result = get_n_supertypes(_make_lex(), 'A', 0)
    assert result is None


def test_get_n_supertypes_n1_immediate_supertypes():
    """n=1 returns the immediate supertypes joined with '+' at level 0."""
    result = get_n_supertypes(_make_lex(), 'A', 1)
    assert result == {0: {'B+C'}}


def test_get_n_supertypes_n2_two_levels():
    """n=2 returns level-0 and level-1 supertypes."""
    result = get_n_supertypes(_make_lex(), 'A', 2)
    assert 0 in result
    assert 1 in result
    assert result[0] == {'B+C'}
    # Both B and C have supertype D, so level 1 contains 'D' once
    assert result[1] == {'D'}


def test_get_n_supertypes_stops_at_missing_parent():
    """Recursion stops gracefully when a supertype is not in lex."""
    # D has no entry in lex, so asking for 3 levels from A should stop at D
    result = get_n_supertypes(_make_lex(), 'A', 3)
    assert 2 not in result   # level 2 (supertypes of D) never populated


def test_get_n_supertypes_single_supertype():
    """A type with one supertype produces a single string (no '+')."""
    result = get_n_supertypes(_make_lex(), 'B', 1)
    assert result == {0: {'D'}}


# ---------------------------------------------------------------------------
# lexical_types
#
# Lexicon: 3 singletons, 4 types with 2 entries, 2 types with 5 entries,
#          1 type with 20 entries.
# np.percentile([1,1,1,2,2,2,2,5,5,20], 90) ≈ 6.5
# → high:      count > 6.5  → only the 20-entry type
# → low:       1 < count ≤ 6.5 → the 2- and 5-entry types
# → singletons: count == 1
# ---------------------------------------------------------------------------

def _make_lexicon():
    return {
        'type_a': ['e1'],
        'type_b': ['e2'],
        'type_c': ['e3'],
        'type_d': ['e4',  'e5'],
        'type_e': ['e6',  'e7'],
        'type_f': ['e8',  'e9'],
        'type_g': ['e10', 'e11'],
        'type_h': ['e12', 'e13', 'e14', 'e15', 'e16'],
        'type_i': ['e17', 'e18', 'e19', 'e20', 'e21'],
        'type_j': ['e{}'.format(i) for i in range(22, 42)],  # 20 entries
    }


def test_lexical_types_high_membership():
    """Only types above the 90th-percentile threshold are in high_membership."""
    high, low, singletons = lexical_types(_make_lexicon())
    assert set(high.keys()) == {'type_j'}


def test_lexical_types_low_membership():
    """Types with 2–5 entries (below threshold, above 1) are in low_membership."""
    high, low, singletons = lexical_types(_make_lexicon())
    assert set(low.keys()) == {'type_d', 'type_e', 'type_f', 'type_g', 'type_h', 'type_i'}


def test_lexical_types_singletons():
    """Types with exactly one entry are in singletons."""
    high, low, singletons = lexical_types(_make_lexicon())
    assert set(singletons.keys()) == {'type_a', 'type_b', 'type_c'}


def test_lexical_types_partitions_are_disjoint():
    """The three groups are mutually exclusive and cover every key."""
    lex = _make_lexicon()
    high, low, singletons = lexical_types(lex)
    all_keys = set(high) | set(low) | set(singletons)
    assert all_keys == set(lex)
    assert len(set(high) & set(low)) == 0
    assert len(set(high) & set(singletons)) == 0
    assert len(set(low) & set(singletons)) == 0


# ---------------------------------------------------------------------------
# create_friendly_name
# ---------------------------------------------------------------------------

def test_create_friendly_name_no_documentation():
    """Returns empty strings when the TDL object has no docstring."""
    obj = MockTDLTypeWithDoc(None)
    defn, ex = create_friendly_name(obj)
    assert defn == ''
    assert ex == ''


def test_create_friendly_name_definition_only():
    """A docstring without <ex> tag is used entirely as the definition."""
    obj = MockTDLTypeWithDoc('Subject-head phrase.')
    defn, ex = create_friendly_name(obj)
    assert defn == 'Subject-head phrase.'
    assert ex == ''


def test_create_friendly_name_with_example():
    """A docstring with <ex> tag splits into definition and example."""
    obj = MockTDLTypeWithDoc('Subject-head phrase.<ex>The dog barked.')
    defn, ex = create_friendly_name(obj)
    assert defn == 'Subject-head phrase.'
    assert ex == 'The dog barked.'


def test_create_friendly_name_strips_whitespace_from_definition():
    """Leading/trailing whitespace around the definition is stripped."""
    obj = MockTDLTypeWithDoc('  A bare NP.  <ex>Snow fell.')
    defn, ex = create_friendly_name(obj)
    assert defn == 'A bare NP.'


# ---------------------------------------------------------------------------
# dict_to_latex_table
# ---------------------------------------------------------------------------

def _make_type_data():
    return {
        'sb-hd_mc_c': {'def': 'Subject-head phrase', 'ex': 'The dog barked.'},
        'hd-comp_c':  {'def': 'Head-complement phrase', 'ex': 'gave the dog a bone.'},
        'hdn_bnp_c':  {'def': 'Bare NP', 'ex': 'Snow fell.'},
    }


def test_dict_to_latex_table_include_filter():
    """Only types listed in include appear in the table body (underscores are escaped)."""
    data = _make_type_data()
    result = dict_to_latex_table(data, include={'sb-hd_mc_c', 'hdn_bnp_c'})
    assert 'sb-hd\\_mc\\_c' in result
    assert 'hdn\\_bnp\\_c' in result
    assert 'hd-comp\\_c' not in result


def test_dict_to_latex_table_underscore_escape():
    """Underscores in type names are escaped as \\_ in the LaTeX output."""
    data = _make_type_data()
    result = dict_to_latex_table(data, include={'sb-hd_mc_c'})
    assert 'sb-hd\\_mc\\_c' in result


def test_dict_to_latex_table_structure():
    """Output contains required LaTeX table boilerplate."""
    data = _make_type_data()
    result = dict_to_latex_table(data, include=set(data.keys()))
    assert '\\begin{table}' in result
    assert '\\end{table}' in result
    assert '\\begin{tabular}' in result
    assert '\\hline' in result


def test_dict_to_latex_table_empty_include():
    """An empty include set produces a table with no data rows."""
    data = _make_type_data()
    result = dict_to_latex_table(data, include=set())
    # Structure still present, but none of the type names appear
    assert '\\begin{table}' in result
    for key in data:
        assert key not in result


# ---------------------------------------------------------------------------
# read_lexicon
#
# Tests our grouping-by-first-supertype and sorting logic.
#
# mini-eng/lexicon.tdl:
#   common-noun-lex:      cat, dog        (2 entries)
#   regular-adj-lex:      big, small      (2 entries)
#   1sg-pronoun-noun-lex: I               (1 entry)
#   3rd-sg-cop-lex:       is              (1 entry)
#   det1-determiner-lex:  the             (1 entry)
#   itr-verb-lex:         sleep           (1 entry)
#   pl-cop-lex:           are             (1 entry)
#   tr-verb-lex:          chase           (1 entry)
# ---------------------------------------------------------------------------

def test_read_lexicon_groups_entries_by_first_supertype():
    """cat and dog share supertype common-noun-lex and are grouped together."""
    result = read_lexicon([FIXTURE_LEXICON])
    assert 'common-noun-lex' in result
    assert sorted(result['common-noun-lex']) == ['cat', 'dog']


def test_read_lexicon_entries_within_group_are_sorted():
    """Entries within each supertype group are alphabetically sorted."""
    result = read_lexicon([FIXTURE_LEXICON])
    for entries in result.values():
        assert entries == sorted(entries)


def test_read_lexicon_groups_sorted_by_count_descending():
    """Groups with more entries appear before groups with fewer."""
    result = read_lexicon([FIXTURE_LEXICON])
    counts = [len(v) for v in result.values()]
    assert counts == sorted(counts, reverse=True)


def test_read_lexicon_singleton_type():
    """A type with one entry is correctly grouped as a single-element list."""
    result = read_lexicon([FIXTURE_LEXICON])
    assert 'tr-verb-lex' in result
    assert result['tr-verb-lex'] == ['chase']
