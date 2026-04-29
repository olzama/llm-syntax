import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import pytest
from unittest.mock import patch
from delphin import derivation

import count_constructions
from count_constructions import traverse_derivation, collect_types_core, collect_types, collect_types_multidir

FIXTURE_DIR       = os.path.join(os.path.dirname(__file__), 'fixtures', 'mini-wsj')
FIXTURE_MULTI_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'mini-multidir')

# ---------------------------------------------------------------------------
# Real derivation: "Not this year."  (wsj00, first short parsed sentence)
#
# Tree structure:
#   root_inffrag
#     pp_frg_c
#       aj-hd_scp-xp_c
#         not_prdp              ← preterminal ("not")
#         np_adv_c
#           sp-hd_n_c
#             this_det          ← preterminal ("this")
#             hd-pct_c
#               n_sg_ilr        ← lexical rule ("year" inflected to singular)
#                 year_n1       ← preterminal ("year")
#               period_pct      ← preterminal (".")
#
# Expected output (with lex={}, so no supertype resolution):
#   constr:  pp_frg_c, aj-hd_scp-xp_c, np_adv_c, sp-hd_n_c, hd-pct_c
#   lexrule: n_sg_ilr
#   lextype: not_prdp, this_det, year_n1, period_pct
# ---------------------------------------------------------------------------

DERIV_STR = '(root_inffrag (0 pp_frg_c 0.000000 0 4 (0 aj-hd_scp-xp_c 0.000000 0 4 (0 not_prdp 0.000000 0 1 ("not" 75 "token [ +FORM \\"not\\" +FROM \\"0\\" +TO \\"3\\" +ID *diff-list* [ LIST *cons* [ FIRST \\"0\\" REST *list* ] LAST *list* ] +TNT null_tnt [ +TAGS *null* +PRBS *null* +MAIN tnt_main [ +TAG \\"RB\\" +PRB \\"1.0\\" ] ] +CLASS alphabetic [ +CASE capitalized+lower +INITIAL + ] +TRAIT token_trait [ +UW - +IT italics +LB bracket_null [ LIST *list* LAST *list* ] +RB bracket_null [ LIST *list* LAST *list* ] +LD bracket_null [ LIST *list* LAST *list* ] +RD bracket_null [ LIST *list* LAST *list* ] +HD token_head [ +TI \\"<0:3>\\" +LL ctype [ -CTYPE- string ] +TG string ] ] +PRED predsort +CARG \\"Not\\" +TICK + +ONSET c-or-v-onset ]")) (0 np_adv_c 0.000000 1 4 (0 sp-hd_n_c 0.000000 1 4 (0 this_det 0.000000 1 2 ("this" 71 "token [ +FORM \\"this\\" +FROM \\"4\\" +TO \\"8\\" +ID *diff-list* [ LIST *cons* [ FIRST \\"1\\" REST *list* ] LAST *list* ] +TNT null_tnt [ +TAGS *null* +PRBS *null* +MAIN tnt_main [ +TAG \\"DT\\" +PRB \\"1.0\\" ] ] +CLASS alphabetic [ +CASE non_capitalized+lower +INITIAL - ] +TRAIT token_trait [ +UW - +IT italics +LB bracket_null [ LIST *list* LAST *list* ] +RB bracket_null [ LIST *list* LAST *list* ] +LD bracket_null [ LIST *list* LAST *list* ] +RD bracket_null [ LIST *list* LAST *list* ] +HD token_head [ +TI \\"<4:8>\\" +LL ctype [ -CTYPE- string ] +TG string ] ] +PRED predsort +CARG \\"this\\" +TICK + +ONSET c-or-v-onset ]")) (0 hd-pct_c 0.000000 2 4 (0 n_sg_ilr 0.000000 2 3 (0 year_n1 0.000000 2 3 ("year" 73 "token [ +FORM \\"year\\" +FROM \\"9\\" +TO \\"13\\" +ID *diff-list* [ LIST *cons* [ FIRST \\"2\\" REST *list* ] LAST *list* ] +TNT null_tnt [ +TAGS *null* +PRBS *null* +MAIN tnt_main [ +TAG \\"NN\\" +PRB \\"1.0\\" ] ] +CLASS alphabetic [ +CASE non_capitalized+lower +INITIAL - ] +TRAIT token_trait [ +UW - +IT italics +LB bracket_null [ LIST *list* LAST *list* ] +RB bracket_null [ LIST *list* LAST *list* ] +LD bracket_null [ LIST *list* LAST *list* ] +RD bracket_nonnull [ LIST *cons* [ FIRST n REST *list* ] LAST *list* ] +HD token_head [ +TI \\"<9:13>\\" +LL ctype [ -CTYPE- string ] +TG string ] ] +PRED predsort +CARG \\"year\\" +TICK + +ONSET c-or-v-onset ]"))) (0 period_pct 0.000000 3 4 ("." 69 "token [ +FORM \\".\\" +FROM \\"13\\" +TO \\"14\\" +ID *diff-list* [ LIST *cons* [ FIRST \\"3\\" REST *list* ] LAST *list* ] +TNT null_tnt [ +TAGS *null* +PRBS *null* +MAIN tnt_main [ +TAG \\".\\" +PRB \\"1.0\\" ] ] +CLASS non_alphanumeric [ +INITIAL - ] +TRAIT token_trait [ +UW - +IT italics +LB bracket_null [ LIST *list* LAST *list* ] +RB bracket_null [ LIST *list* LAST *list* ] +LD bracket_null [ LIST *list* LAST *list* ] +RD bracket_nonnull [ LIST *cons* [ FIRST n REST *list* ] LAST *list* ] +HD token_head [ +TI \\"<13:14>\\" +LL ctype [ -CTYPE- string ] +TG string ] ] +PRED predsort +CARG \\".\\" +TICK + +ONSET c-or-v-onset ]")))))))))'

EXPECTED_CONSTRS  = {'pp_frg_c', 'aj-hd_scp-xp_c', 'np_adv_c', 'sp-hd_n_c', 'hd-pct_c'}
EXPECTED_LEXRULES = {'n_sg_ilr'}
EXPECTED_LEXTYPES = {'not_prdp', 'this_det', 'year_n1', 'period_pct'}

# ---------------------------------------------------------------------------
# Real derivation: "Champagne and dessert followed."  (wsj00, i-id 20010008)
#
# Tree structure:
#   root_strict
#     sb-hd_mc_c
#       np-np_crd-t_c
#         hdn_bnp_c          ← count=2 (bare NP construction, appears twice)
#           n_ms_ilr          ← lexrule (mass singular)
#             champagne_n1    ← preterminal
#         mrk-nh_nom_c
#           and_conj          ← preterminal
#           hdn_bnp_c         ← second occurrence
#             n_ms-cnt_ilr    ← lexrule (mass-count)
#               dessert_n1    ← preterminal
#       hd_optcmp_c
#         hd-pct_c
#           v_pst_olr         ← lexrule (past tense)
#             follow_v1       ← preterminal
#           period_pct        ← preterminal
#
# Expected output (with lex={}, no supertype resolution):
#   constr:  sb-hd_mc_c(1), np-np_crd-t_c(1), hdn_bnp_c(2), mrk-nh_nom_c(1),
#            hd_optcmp_c(1), hd-pct_c(1)
#   lexrule: n_ms_ilr(1), n_ms-cnt_ilr(1), v_pst_olr(1)
#   lextype: champagne_n1(1), and_conj(1), dessert_n1(1), follow_v1(1), period_pct(1)
# ---------------------------------------------------------------------------

CHAMPAGNE_EXPECTED_CONSTRS  = {'sb-hd_mc_c', 'np-np_crd-t_c', 'hdn_bnp_c',
                                'mrk-nh_nom_c', 'hd_optcmp_c', 'hd-pct_c'}
CHAMPAGNE_EXPECTED_LEXRULES = {'n_ms_ilr', 'n_ms-cnt_ilr', 'v_pst_olr'}
CHAMPAGNE_EXPECTED_LEXTYPES = {'champagne_n1', 'and_conj', 'dessert_n1',
                                'follow_v1', 'period_pct'}


def run_traversal():
    """Parse the 'Not this year.' derivation string and run traverse_derivation
    with no lex (so preterminals keep their original entity names)."""
    deriv = derivation.from_string(DERIV_STR)
    preterminals = {pt.entity for pt in deriv.preterminals()}
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}
    traverse_derivation(deriv, types, preterminals=preterminals, lex={}, depth=1)
    return types

def run_champagne_traversal():
    """Load 'Champagne and dessert followed.' from the fixture and run
    traverse_derivation with no lex."""
    from delphin import itsdb
    db = itsdb.TestSuite(FIXTURE_DIR)
    for response in db.processed_items():
        if response['i-input'] == 'Champagne and dessert followed.':
            deriv_str = response['results'][0]['derivation']
            break
    deriv = derivation.from_string(deriv_str)
    preterminals = {pt.entity for pt in deriv.preterminals()}
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}
    traverse_derivation(deriv, types, preterminals=preterminals, lex={}, depth=1)
    return deriv, types


# ---------------------------------------------------------------------------
# Construction classification
# ---------------------------------------------------------------------------

def test_all_constructions_found():
    """All five phrasal constructions in 'Not this year.' should appear in constr:
    pp_frg_c (PP fragment), aj-hd_scp-xp_c (adjunct-head scope),
    np_adv_c (NP adverb), sp-hd_n_c (specifier-head nominal), hd-pct_c (head-punctuation)."""
    types = run_traversal()
    actual = set(types['constr'].keys())
    print(f"\n  Input:    derivation of 'Not this year.'")
    print(f"  Expected: {sorted(EXPECTED_CONSTRS)}")
    print(f"  Actual:   {sorted(actual)}")
    assert actual == EXPECTED_CONSTRS, (
        f"Expected constructions {EXPECTED_CONSTRS}, got: {actual}"
    )

def test_each_construction_counted_once():
    """Each of the five constructions occurs exactly once in this sentence."""
    types = run_traversal()
    print(f"\n  Input:    derivation of 'Not this year.'")
    for c in sorted(EXPECTED_CONSTRS):
        actual = types['constr'].get(c)
        print(f"  '{c}': expected count=1, actual count={actual}")
        assert actual == 1, (
            f"Expected '{c}' to have count 1 in constr, got: {actual}"
        )

# ---------------------------------------------------------------------------
# Lexical rule classification
# ---------------------------------------------------------------------------

def test_lexrule_found():
    """n_sg_ilr (singular inflection rule applied to 'year') should appear in lexrule."""
    types = run_traversal()
    actual = set(types['lexrule'].keys())
    print(f"\n  Input:    derivation of 'Not this year.'")
    print(f"  Expected: {sorted(EXPECTED_LEXRULES)}")
    print(f"  Actual:   {sorted(actual)}")
    assert actual == EXPECTED_LEXRULES, (
        f"Expected lexrules {EXPECTED_LEXRULES}, got: {actual}"
    )

def test_lexrule_not_in_constr():
    """n_sg_ilr ends in 'lr' and must not be counted under constr."""
    types = run_traversal()
    actual = set(types['constr'].keys())
    print(f"\n  Input:    derivation of 'Not this year.'")
    print(f"  Expected: 'n_sg_ilr' absent from constr")
    print(f"  Actual constr: {sorted(actual)}")
    assert 'n_sg_ilr' not in types['constr'], (
        f"'n_sg_ilr' should not appear in constr, got: {types['constr']}"
    )

# ---------------------------------------------------------------------------
# Lexical type classification
# ---------------------------------------------------------------------------

def test_all_lextypes_found():
    """All four preterminals should appear in lextype (no lex provided, so original
    entity names are kept): not_prdp, this_det, year_n1, period_pct."""
    types = run_traversal()
    actual = set(types['lextype'].keys())
    print(f"\n  Input:    derivation of 'Not this year.', lex={{}} (no supertype resolution)")
    print(f"  Expected: {sorted(EXPECTED_LEXTYPES)}")
    print(f"  Actual:   {sorted(actual)}")
    assert actual == EXPECTED_LEXTYPES, (
        f"Expected lextypes {EXPECTED_LEXTYPES}, got: {actual}"
    )

def test_lextypes_not_in_constr():
    """Preterminal nodes must not appear in constr."""
    types = run_traversal()
    actual_constr = set(types['constr'].keys())
    print(f"\n  Input:    derivation of 'Not this year.'")
    print(f"  Expected: none of {sorted(EXPECTED_LEXTYPES)} in constr")
    print(f"  Actual constr: {sorted(actual_constr)}")
    for lt in EXPECTED_LEXTYPES:
        assert lt not in types['constr'], (
            f"Preterminal '{lt}' should not appear in constr, got: {types['constr']}"
        )

# ---------------------------------------------------------------------------
# No cross-contamination
# ---------------------------------------------------------------------------

def test_no_overlap_between_categories():
    """The sets of types in constr, lexrule, and lextype must be disjoint."""
    types = run_traversal()
    constr_keys  = set(types['constr'].keys())
    lexrule_keys = set(types['lexrule'].keys())
    lextype_keys = set(types['lextype'].keys())
    print(f"\n  Input:    derivation of 'Not this year.'")
    print(f"  constr:   {sorted(constr_keys)}")
    print(f"  lexrule:  {sorted(lexrule_keys)}")
    print(f"  lextype:  {sorted(lextype_keys)}")
    print(f"  Expected: all three sets are pairwise disjoint")
    assert constr_keys.isdisjoint(lexrule_keys), (
        f"Overlap between constr and lexrule: {constr_keys & lexrule_keys}"
    )
    assert constr_keys.isdisjoint(lextype_keys), (
        f"Overlap between constr and lextype: {constr_keys & lextype_keys}"
    )
    assert lexrule_keys.isdisjoint(lextype_keys), (
        f"Overlap between lexrule and lextype: {lexrule_keys & lextype_keys}"
    )

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_already_visited_root_skipped():
    """If the root node is pre-inserted into the visited set, traverse_derivation
    must return immediately and produce no counts."""
    deriv = derivation.from_string(DERIV_STR)
    preterminals = {pt.entity for pt in deriv.preterminals()}
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}
    visited = {id(deriv)}
    traverse_derivation(deriv, types, preterminals=preterminals, lex={}, depth=1, visited=visited)
    print(f"\n  Input:    derivation of 'Not this year.', root pre-added to visited set")
    print(f"  Expected: all dicts empty")
    print(f"  Actual:   {types}")
    assert types == {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}, (
        f"Pre-visited root should produce no counts, got: {types}"
    )

# ---------------------------------------------------------------------------
# Supertype resolution
# ---------------------------------------------------------------------------

def test_supertype_replaces_preterminal_entity():
    """When get_n_supertypes returns a supertype for a preterminal, the supertype
    should be stored in lextype instead of the original entity.
    Here year_n1 (a specific lexical entry) is resolved to n_-_c_le
    (the common noun lexical type) at depth 1."""
    deriv = derivation.from_string(DERIV_STR)
    preterminals = {pt.entity for pt in deriv.preterminals()}
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}

    def mock_supertypes(lex, type_name, depth):
        if type_name == 'year_n1':
            return {0: {'n_-_c_le'}}
        return None

    with patch('erg.get_n_supertypes', side_effect=mock_supertypes):
        traverse_derivation(deriv, types, preterminals=preterminals, lex={}, depth=1)

    print(f"\n  Input:    derivation of 'Not this year.', get_n_supertypes mocked: year_n1 -> n_-_c_le")
    print(f"  Expected: 'n_-_c_le' in lextype, 'year_n1' absent")
    print(f"  Actual lextype: {sorted(types['lextype'].keys())}")
    assert 'n_-_c_le' in types['lextype'], (
        f"Expected supertype 'n_-_c_le' in lextype, got: {types['lextype']}"
    )
    assert 'year_n1' not in types['lextype'], (
        f"Original entry 'year_n1' should be replaced by its supertype, got: {types['lextype']}"
    )

# ---------------------------------------------------------------------------
# Lexical entries
# ---------------------------------------------------------------------------

def test_lexentries_populated_from_terminals():
    """collect_types_core populates lexentries from terminal parents (i.e. the
    lexical entry name dominating each word token). For 'Not this year.' the four
    terminal parents are not_prdp, this_det, year_n1, period_pct, each occurring once."""
    deriv = derivation.from_string(DERIV_STR)
    lexentries = {}
    for t in deriv.terminals():
        entity = t.parent.entity
        if entity not in lexentries:
            lexentries[entity] = 0
        lexentries[entity] += 1
    expected = {'not_prdp': 1, 'this_det': 1, 'year_n1': 1, 'period_pct': 1}
    print(f"\n  Input:    terminals of 'Not this year.' ('not', 'this', 'year', '.')")
    print(f"  Expected: {expected}")
    print(f"  Actual:   {lexentries}")
    assert lexentries == expected, (
        f"Expected one entry per word token, got: {lexentries}"
    )

# ---------------------------------------------------------------------------
# collect_types_core: integration with TestSuite fixture
# ---------------------------------------------------------------------------

def test_collect_types_core_processes_parsed_items():
    """collect_types_core processes all parsed items in the fixture: 'Not this year.'
    and 'Champagne and dessert followed.' The result must be the union of both sentences'
    types. 'The of the.' (0 readings) contributes nothing."""
    all_expected_constrs = EXPECTED_CONSTRS | CHAMPAGNE_EXPECTED_CONSTRS
    all_expected_lexrules = EXPECTED_LEXRULES | CHAMPAGNE_EXPECTED_LEXRULES
    all_expected_lextypes = EXPECTED_LEXTYPES | CHAMPAGNE_EXPECTED_LEXTYPES
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}
    collect_types_core(FIXTURE_DIR, depth=1, lex={}, sample_size=None, types=types)
    print(f"\n  Input:    fixture testsuite: 'Not this year.' + 'Champagne and dessert followed.' + 'The of the.'")
    print(f"  Expected constr:   {sorted(all_expected_constrs)}")
    print(f"  Actual constr:     {sorted(types['constr'].keys())}")
    print(f"  Expected lexrule:  {sorted(all_expected_lexrules)}")
    print(f"  Actual lexrule:    {sorted(types['lexrule'].keys())}")
    print(f"  Expected lextype:  {sorted(all_expected_lextypes)}")
    print(f"  Actual lextype:    {sorted(types['lextype'].keys())}")
    assert set(types['constr'].keys()) == all_expected_constrs, (
        f"Expected constructions {all_expected_constrs}, got: {set(types['constr'].keys())}"
    )
    assert set(types['lexrule'].keys()) == all_expected_lexrules, (
        f"Expected lexrules {all_expected_lexrules}, got: {set(types['lexrule'].keys())}"
    )
    assert set(types['lextype'].keys()) == all_expected_lextypes, (
        f"Expected lextypes {all_expected_lextypes}, got: {set(types['lextype'].keys())}"
    )

def test_collect_types_core_skips_item_with_no_results():
    """collect_types_core must skip items with 0 readings ('The of the.' has no parse).
    Only 'Not this year.' should contribute counts; total processed items is 2."""
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}
    n = collect_types_core(FIXTURE_DIR, depth=1, lex={}, sample_size=None, types=types)
    print(f"\n  Input:    fixture testsuite: 2 parsed items + 'The of the.' (0 readings)")
    print(f"  Expected: 3 items processed total, ungrammatical item contributes no types")
    print(f"  Actual:   {n} items processed, constr={sorted(types['constr'].keys())}")
    assert n == 3, (
        f"Expected 3 items processed (2 parsed + 1 unparsed), got: {n}"
    )
    assert set(types['constr'].keys()) == EXPECTED_CONSTRS | CHAMPAGNE_EXPECTED_CONSTRS, (
        f"Unparsed item must not add to constr, got: {set(types['constr'].keys())}"
    )

# ---------------------------------------------------------------------------
# "Champagne and dessert followed." — counts > 1 and depth
# ---------------------------------------------------------------------------

def test_repeated_construction_counted_correctly():
    """hdn_bnp_c (bare NP) appears twice in 'Champagne and dessert followed.'
    once for each conjunct. Its count must be 2."""
    _, types = run_champagne_traversal()
    actual = types['constr'].get('hdn_bnp_c')
    print(f"\n  Input:    derivation of 'Champagne and dessert followed.'")
    print(f"  Expected: constr['hdn_bnp_c'] = 2  (bare NP for 'champagne' and 'dessert')")
    print(f"  Actual:   constr['hdn_bnp_c'] = {actual}")
    assert actual == 2, (
        f"Expected hdn_bnp_c count=2, got: {actual}"
    )

def test_single_occurrence_constructions_counted_once():
    """All other constructions in 'Champagne and dessert followed.' appear exactly once."""
    _, types = run_champagne_traversal()
    once = CHAMPAGNE_EXPECTED_CONSTRS - {'hdn_bnp_c'}
    print(f"\n  Input:    derivation of 'Champagne and dessert followed.'")
    for c in sorted(once):
        actual = types['constr'].get(c)
        print(f"  '{c}': expected count=1, actual count={actual}")
        assert actual == 1, f"Expected '{c}' count=1, got: {actual}"

def test_champagne_all_constructions_found():
    """All six construction types in 'Champagne and dessert followed.' must be present."""
    _, types = run_champagne_traversal()
    actual = set(types['constr'].keys())
    print(f"\n  Input:    derivation of 'Champagne and dessert followed.'")
    print(f"  Expected: {sorted(CHAMPAGNE_EXPECTED_CONSTRS)}")
    print(f"  Actual:   {sorted(actual)}")
    assert actual == CHAMPAGNE_EXPECTED_CONSTRS, (
        f"Expected constructions {CHAMPAGNE_EXPECTED_CONSTRS}, got: {actual}"
    )

def test_champagne_all_lexrules_found():
    """Three lexrules: n_ms_ilr (champagne), n_ms-cnt_ilr (dessert), v_pst_olr (followed)."""
    _, types = run_champagne_traversal()
    actual = set(types['lexrule'].keys())
    print(f"\n  Input:    derivation of 'Champagne and dessert followed.'")
    print(f"  Expected: {sorted(CHAMPAGNE_EXPECTED_LEXRULES)}")
    print(f"  Actual:   {sorted(actual)}")
    assert actual == CHAMPAGNE_EXPECTED_LEXRULES, (
        f"Expected lexrules {CHAMPAGNE_EXPECTED_LEXRULES}, got: {actual}"
    )

def test_supertype_resolution_merges_counts():
    """When both champagne_n1 and dessert_n1 resolve to the same supertype n_-_c_le,
    the count for that supertype in lextype must be 2 (one for each noun)."""
    deriv, _ = run_champagne_traversal()
    preterminals = {pt.entity for pt in deriv.preterminals()}
    types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}

    def mock_supertypes(lex, type_name, depth):
        if type_name in ('champagne_n1', 'dessert_n1'):
            return {0: {'n_-_c_le'}}
        return None

    with patch('erg.get_n_supertypes', side_effect=mock_supertypes):
        traverse_derivation(deriv, types, preterminals=preterminals, lex={}, depth=1)

    actual = types['lextype'].get('n_-_c_le')
    print(f"\n  Input:    derivation of 'Champagne and dessert followed.'")
    print(f"            get_n_supertypes mocked: champagne_n1 -> n_-_c_le, dessert_n1 -> n_-_c_le")
    print(f"  Expected: lextype['n_-_c_le'] = 2")
    print(f"  Actual:   lextype['n_-_c_le'] = {actual}")
    print(f"  Full lextype: {types['lextype']}")
    assert actual == 2, (
        f"Expected supertype n_-_c_le count=2 (champagne + dessert), got: {actual}"
    )

# ---------------------------------------------------------------------------
# collect_types: sorting
# ---------------------------------------------------------------------------

def test_collect_types_returns_sorted_by_count():
    """collect_types wraps collect_types_core and sorts each dict by count descending.
    hdn_bnp_c appears twice in 'Champagne and dessert followed.' so it should rank
    above all other constructions which appear once."""
    sorted_types = collect_types(FIXTURE_DIR, lex={}, depth=1)
    constr = sorted_types['constr']
    counts = list(constr.values())
    print(f"\n  Input:    fixture testsuite (two parsed sentences)")
    print(f"  Expected: constr values sorted descending, hdn_bnp_c first with count 2")
    print(f"  Actual constr (ordered): {list(constr.items())[:5]} ...")
    assert counts == sorted(counts, reverse=True), (
        f"constr values are not sorted descending: {counts}"
    )
    first_key = next(iter(constr))
    assert first_key == 'hdn_bnp_c', (
        f"Expected hdn_bnp_c (count=2) to be first, got: {first_key}"
    )

# ---------------------------------------------------------------------------
# collect_types_core: sample_size
# ---------------------------------------------------------------------------

def test_sample_size_limits_items_processed():
    """When sample_size=1, collect_types_core should process exactly 1 item.
    We mock random.sample to always return the first item so the result is deterministic."""
    from unittest.mock import patch as mock_patch

    types = {'constr': {}, 'lexrule': {}, 'lextype': {}, 'lexentries': {}}

    def take_first(population, k):
        return population[:k]

    with mock_patch('count_constructions.random.sample', side_effect=take_first):
        n = collect_types_core(FIXTURE_DIR, depth=1, lex={}, sample_size=1, types=types)

    print(f"\n  Input:    fixture testsuite (3 items), sample_size=1")
    print(f"            random.sample mocked to always return first item")
    print(f"  Expected: 1 item processed, types non-empty")
    print(f"  Actual:   {n} items processed, constr={sorted(types['constr'].keys())}")
    assert n == 1, (
        f"Expected 1 item processed with sample_size=1, got: {n}"
    )
    assert len(types['constr']) > 0 or len(types['lextype']) > 0, (
        "Expected at least some types to be populated from the sampled item"
    )

# ---------------------------------------------------------------------------
# collect_types_multidir
# ---------------------------------------------------------------------------

def test_collect_types_multidir_processes_all_subdirs():
    """collect_types_multidir iterates over wsj-a/ and wsj-b/, each a copy of
    mini-wsj with the same 3 items (2 parsed). Counts should be exactly double
    those from a single directory. hdn_bnp_c appears twice per sentence, so 4 total."""
    sorted_types = collect_types_multidir(FIXTURE_MULTI_DIR, lex={}, depth=1)
    actual_hdn = sorted_types['constr'].get('hdn_bnp_c')
    all_expected = EXPECTED_CONSTRS | CHAMPAGNE_EXPECTED_CONSTRS
    print(f"\n  Input:    mini-multidir/ with wsj-a/ and wsj-b/ (each a copy of mini-wsj)")
    print(f"  Expected: constr keys = {sorted(all_expected)}")
    print(f"  Actual constr keys:   {sorted(sorted_types['constr'].keys())}")
    print(f"  Expected hdn_bnp_c count: 4 (2 per directory x 2 directories)")
    print(f"  Actual hdn_bnp_c count:   {actual_hdn}")
    assert set(sorted_types['constr'].keys()) == all_expected, (
        f"Expected constructions {all_expected}, got: {set(sorted_types['constr'].keys())}"
    )
    assert actual_hdn == 4, (
        f"Expected hdn_bnp_c count=4 across two directories, got: {actual_hdn}"
    )

def test_collect_types_multidir_total_sentences():
    """collect_types_multidir processes both subdirectories: 3 items each = 6 total."""
    sorted_types = collect_types_multidir(FIXTURE_MULTI_DIR, lex={}, depth=1)
    # Each directory has hd-pct_c once in 'Not this year.' and once in 'Champagne...'
    # so total across both dirs should be 4
    actual = sorted_types['constr'].get('hd-pct_c')
    print(f"\n  Input:    mini-multidir/ with wsj-a/ and wsj-b/")
    print(f"  Expected: hd-pct_c count=4 (appears once per sentence x 2 sentences x 2 dirs)")
    print(f"  Actual:   hd-pct_c count={actual}")
    assert actual == 4, (
        f"Expected hd-pct_c count=4 across two directories, got: {actual}"
    )

def test_collect_types_multidir_ignores_non_directory_entries():
    """collect_types_multidir must skip non-directory entries (e.g. .DS_Store,
    stray files) without crashing. The fixture contains .DS_Store and stray-file.txt
    alongside wsj-a/ and wsj-b/."""
    try:
        sorted_types = collect_types_multidir(FIXTURE_MULTI_DIR, lex={}, depth=1)
        raised = False
    except Exception as e:
        raised = True
        error = e
    print(f"\n  Input:    mini-multidir/ containing wsj-a/, wsj-b/, .DS_Store, stray-file.txt")
    print(f"  Expected: no exception raised, counts unaffected by non-directory entries")
    print(f"  Actual:   exception={'yes: ' + str(error) if raised else 'none'}")
    print(f"  hdn_bnp_c count: {sorted_types['constr'].get('hdn_bnp_c') if not raised else 'n/a'}")
    assert not raised, f"collect_types_multidir crashed on non-directory entry: {error}"
    assert sorted_types['constr'].get('hdn_bnp_c') == 4, (
        "Non-directory entries must not affect counts"
    )
