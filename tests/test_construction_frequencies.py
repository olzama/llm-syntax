import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import json
import pytest
from construction_frequencies import (
    exclusive_members,
    combine_types,
    combine_datasets,
    combine_lextype_datasets,
    separate_dataset,
    read_dataset,
    add_new_dataset,
    map_word2membership,
    add_membership_to_freq,
    compare_with_other_datasets,
    compare_human_vs_machine,
    find_absolute_diffs_lextype,
    compare_lexentries,
    build_llm_vs_human,
    is_human,
)

_REPO_ROOT = os.path.join(os.path.dirname(__file__), '..')
_FREQ_2023  = os.path.join(_REPO_ROOT, 'analysis', 'frequencies-json', 'frequencies-2023.json')
_REF_SYNTAX = os.path.join(_REPO_ROOT, 'analysis', 'cosine-pairs', 'models',
                            'norm-by-constr-count', 'syntax-only.json')

# Old name → new name mapping for the regression comparison
_NAME_MAP = {
    'Falcon7B-2023-llm':    'falcon_07',
    'Llama7B-2023-llm':     'llama_07',
    'Llama13B-2023-llm':    'llama_13',
    'Llama30B-2023-llm':    'llama_30',
    'Llama65B-2023-llm':    'llama_65',
    'Mistral7B-2023-llm':   'mistral_07',
    'NYT-2023-human':       'original',
    'WSJ-1987-human':       'wsj',
    'Wikipedia-2008-human': 'wikipedia',
}


# ---------------------------------------------------------------------------
# Shared minimal fixture data
#
# Two "models": human_a and llm_b.
# Phenomena: constr, lexrule, lextype.
# ---------------------------------------------------------------------------

def _minimal_freq():
    """Return a minimal freq dict with two models sharing some types."""
    return {
        'constr':  {'human_a': {'sb-hd_mc_c': 10, 'hd-pct_c': 5},
                    'llm_b':   {'sb-hd_mc_c':  8, 'np_frg_c': 3}},
        'lexrule': {'human_a': {'n_sg_ilr': 4},
                    'llm_b':   {'v_pst_olr': 2}},
        'lextype': {'human_a': {'n_-_c_le': 7, 'v_e_le': 2},
                    'llm_b':   {'n_-_c_le': 6}},
    }


# ---------------------------------------------------------------------------
# is_human
# ---------------------------------------------------------------------------

def test_is_human_new_convention():
    """Models with '-human' suffix are recognised as human."""
    assert is_human('NYT-2023-human')
    assert is_human('WSJ-1987-human')
    assert is_human('Wikipedia-2008-human')


def test_is_human_llm_returns_false():
    """Models with '-llm' suffix are not human."""
    assert not is_human('Falcon7B-2023-llm')
    assert not is_human('GPT4o-2025-llm')


def test_is_human_case_insensitive():
    """The check is case-insensitive."""
    assert is_human('NYT-2023-HUMAN')


# ---------------------------------------------------------------------------
# exclusive_members
# ---------------------------------------------------------------------------

def test_exclusive_members_only_in_mine():
    """Types present only in human_a (not in llm_b) appear in only_in_mine."""
    freq = _minimal_freq()
    only_mine, only_other = exclusive_members(freq, 'human_a', ['llm_b'])
    print(f"\n  Input:    human_a has {{hd-pct_c, sb-hd_mc_c}}, llm_b has {{sb-hd_mc_c, np_frg_c}}")
    print(f"  Expected: only_mine['constr'] = {{'hd-pct_c'}}")
    print(f"  Actual:   only_mine['constr'] = {only_mine['constr']}")
    assert only_mine['constr'] == {'hd-pct_c'}, (
        f"Expected only hd-pct_c exclusive to human_a, got: {only_mine['constr']}"
    )


def test_exclusive_members_only_in_other():
    """Types present only in llm_b (not in human_a) appear in only_in_other."""
    freq = _minimal_freq()
    only_mine, only_other = exclusive_members(freq, 'human_a', ['llm_b'])
    print(f"\n  Input:    human_a has {{hd-pct_c, sb-hd_mc_c}}, llm_b has {{sb-hd_mc_c, np_frg_c}}")
    print(f"  Expected: only_other['constr'] = {{'np_frg_c'}}")
    print(f"  Actual:   only_other['constr'] = {only_other['constr']}")
    assert only_other['constr'] == {'np_frg_c'}, (
        f"Expected only np_frg_c exclusive to llm_b, got: {only_other['constr']}"
    )


def test_exclusive_members_zero_count_excluded():
    """Types with count 0 should not appear in either exclusive set."""
    freq = {
        'constr':  {'human_a': {'sb-hd_mc_c': 5, 'rare_c': 0},
                    'llm_b':   {'sb-hd_mc_c': 3}},
        'lexrule': {'human_a': {}, 'llm_b': {}},
        'lextype': {'human_a': {}, 'llm_b': {}},
    }
    only_mine, only_other = exclusive_members(freq, 'human_a', ['llm_b'])
    print(f"\n  Input:    human_a has rare_c with count=0")
    print(f"  Expected: rare_c not in only_mine['constr']")
    print(f"  Actual:   only_mine['constr'] = {only_mine['constr']}")
    assert 'rare_c' not in only_mine['constr'], (
        f"Zero-count type 'rare_c' should be excluded, got: {only_mine['constr']}"
    )


def test_exclusive_members_shared_types_absent():
    """Types present in both human_a and llm_b appear in neither exclusive set."""
    freq = _minimal_freq()
    only_mine, only_other = exclusive_members(freq, 'human_a', ['llm_b'])
    print(f"\n  Input:    sb-hd_mc_c present in both models")
    print(f"  Expected: sb-hd_mc_c absent from both exclusive sets")
    print(f"  only_mine['constr']:  {only_mine['constr']}")
    print(f"  only_other['constr']: {only_other['constr']}")
    assert 'sb-hd_mc_c' not in only_mine['constr'], (
        f"Shared type sb-hd_mc_c should not be in only_mine"
    )
    assert 'sb-hd_mc_c' not in only_other['constr'], (
        f"Shared type sb-hd_mc_c should not be in only_other"
    )


# ---------------------------------------------------------------------------
# combine_types
# ---------------------------------------------------------------------------

def test_combine_types_sums_across_phenomena():
    """combine_types(['constr', 'lexrule']) merges both phenomena into one flat dict per model.
    For human_a: sb-hd_mc_c=10, hd-pct_c=5, n_sg_ilr=4."""
    freq = _minimal_freq()
    combined = combine_types(freq, ['constr', 'lexrule'])
    print(f"\n  Input:    freq with constr and lexrule phenomena for human_a")
    print(f"  Expected: human_a has sb-hd_mc_c=10, hd-pct_c=5, n_sg_ilr=4")
    print(f"  Actual:   {combined.get('human_a')}")
    assert combined['human_a']['sb-hd_mc_c'] == 10
    assert combined['human_a']['hd-pct_c'] == 5
    assert combined['human_a']['n_sg_ilr'] == 4


def test_combine_types_excludes_unrequested_phenomena():
    """combine_types(['constr']) must not include lexrule or lextype entries."""
    freq = _minimal_freq()
    combined = combine_types(freq, ['constr'])
    print(f"\n  Input:    relevant_keys=['constr'] only")
    print(f"  Expected: n_sg_ilr (lexrule) absent from combined human_a")
    print(f"  Actual:   {combined.get('human_a')}")
    assert 'n_sg_ilr' not in combined['human_a'], (
        f"Lexrule key should be excluded when not in relevant_keys"
    )
    assert 'n_-_c_le' not in combined['human_a'], (
        f"Lextype key should be excluded when not in relevant_keys"
    )


def test_combine_types_all_models_present():
    """All models present in the source data must appear in the combined output."""
    freq = _minimal_freq()
    combined = combine_types(freq, ['constr', 'lexrule', 'lextype'])
    print(f"\n  Input:    freq with models human_a, llm_b")
    print(f"  Expected: both models present in combined output")
    print(f"  Actual keys: {list(combined.keys())}")
    assert 'human_a' in combined
    assert 'llm_b' in combined


# ---------------------------------------------------------------------------
# combine_datasets
# ---------------------------------------------------------------------------

def test_combine_datasets_sums_counts():
    """combine_datasets(['human_a', 'llm_b'], 'all') sums counts across both datasets.
    sb-hd_mc_c: human_a=10, llm_b=8 → all=18."""
    freq = _minimal_freq()
    combined = combine_datasets(freq, ['human_a', 'llm_b'], 'all')
    print(f"\n  Input:    human_a sb-hd_mc_c=10, llm_b sb-hd_mc_c=8")
    print(f"  Expected: combined 'all' constr sb-hd_mc_c = 18")
    print(f"  Actual:   {combined['constr'].get('all', {}).get('sb-hd_mc_c')}")
    assert combined['constr']['all']['sb-hd_mc_c'] == 18, (
        f"Expected sb-hd_mc_c=18 in combined, got: {combined['constr']['all']}"
    )


def test_combine_datasets_includes_exclusive_types():
    """Types exclusive to one dataset still appear in the combined output.
    hd-pct_c is only in human_a (count=5); np_frg_c only in llm_b (count=3)."""
    freq = _minimal_freq()
    combined = combine_datasets(freq, ['human_a', 'llm_b'], 'all')
    print(f"\n  Input:    hd-pct_c only in human_a (5), np_frg_c only in llm_b (3)")
    print(f"  Expected: both appear in combined 'all'")
    print(f"  Actual:   {combined['constr']['all']}")
    assert combined['constr']['all']['hd-pct_c'] == 5
    assert combined['constr']['all']['np_frg_c'] == 3


def test_combine_datasets_creates_all_phenomena_keys():
    """The returned dict must have entries for all three phenomena."""
    freq = _minimal_freq()
    combined = combine_datasets(freq, ['human_a', 'llm_b'], 'all')
    print(f"\n  Input:    freq with constr, lexrule, lextype phenomena")
    print(f"  Expected: all three phenomena present in combined")
    print(f"  Actual keys: {list(combined.keys())}")
    assert 'constr' in combined
    assert 'lexrule' in combined
    assert 'lextype' in combined


# ---------------------------------------------------------------------------
# separate_dataset
# ---------------------------------------------------------------------------

def test_separate_dataset_extracts_correct_model():
    """separate_dataset on 'human_a' returns only human_a's types for each phenomenon."""
    freq = _minimal_freq()
    ds = separate_dataset(freq, 'human_a')
    print(f"\n  Input:    freq with two models, extract 'human_a'")
    print(f"  Expected: constr = {{sb-hd_mc_c: 10, hd-pct_c: 5}}")
    print(f"  Actual:   constr = {ds['constr']}")
    assert ds['constr'] == {'sb-hd_mc_c': 10, 'hd-pct_c': 5}, (
        f"Expected human_a constr only, got: {ds['constr']}"
    )


def test_separate_dataset_does_not_include_other_model():
    """separate_dataset must not include types from other models (llm_b's np_frg_c)."""
    freq = _minimal_freq()
    ds = separate_dataset(freq, 'human_a')
    print(f"\n  Input:    llm_b has np_frg_c which human_a does not")
    print(f"  Expected: np_frg_c absent from separated human_a")
    print(f"  Actual:   {ds['constr']}")
    assert 'np_frg_c' not in ds['constr'], (
        f"np_frg_c belongs to llm_b and must not appear in human_a's separated data"
    )


# ---------------------------------------------------------------------------
# read_dataset
# ---------------------------------------------------------------------------

def test_read_dataset_adds_to_frequencies():
    """read_dataset copies rules from a new_dataset dict into frequencies under dataset_name."""
    frequencies = {'constr': {}, 'lexrule': {}, 'lextype': {}}
    new_dataset = {'constr': {'sb-hd_mc_c': 3}, 'lexrule': {'n_sg_ilr': 1}, 'lextype': {'n_-_c_le': 5}}
    read_dataset(frequencies, new_dataset, 'model_x')
    print(f"\n  Input:    new_dataset with sb-hd_mc_c=3, n_sg_ilr=1, n_-_c_le=5")
    print(f"  Expected: frequencies['constr']['model_x']['sb-hd_mc_c'] = 3")
    print(f"  Actual:   {frequencies['constr'].get('model_x')}")
    assert frequencies['constr']['model_x']['sb-hd_mc_c'] == 3
    assert frequencies['lexrule']['model_x']['n_sg_ilr'] == 1
    assert frequencies['lextype']['model_x']['n_-_c_le'] == 5


# ---------------------------------------------------------------------------
# add_new_dataset
# ---------------------------------------------------------------------------

def test_add_new_dataset_transfers_counts():
    """add_new_dataset copies rules from new_dataset[model_name] into frequencies."""
    frequencies = {
        'constr':  {'NYT-2023-human': {'sb-hd_mc_c': 10, 'hd-pct_c': 5}},
        'lexrule': {'NYT-2023-human': {'n_sg_ilr': 4}},
        'lextype': {'NYT-2023-human': {'n_-_c_le': 7}},
    }
    new_data = {
        'constr':  {'NYT-2025-human': {'sb-hd_mc_c': 3, 'hd-pct_c': 1}},
        'lexrule': {'NYT-2025-human': {'n_sg_ilr': 2}},
        'lextype': {'NYT-2025-human': {'n_-_c_le': 5}},
    }
    add_new_dataset(frequencies, new_data, 'NYT-2025-human', model_name='NYT-2025-human')
    print(f"\n  Input:    new_data with sb-hd_mc_c=3 under 'NYT-2025-human'")
    print(f"  Expected: frequencies['constr']['NYT-2025-human']['sb-hd_mc_c'] = 3")
    print(f"  Actual:   {frequencies['constr'].get('NYT-2025-human')}")
    assert frequencies['constr']['NYT-2025-human']['sb-hd_mc_c'] == 3


def test_add_new_dataset_fills_zeros_for_missing_rules():
    """After add_new_dataset, rules present in existing models but absent in new_data
    should be filled with 0 in the new dataset entry."""
    frequencies = {
        'constr':  {'NYT-2023-human': {'sb-hd_mc_c': 10, 'hd-pct_c': 5}},
        'lexrule': {'NYT-2023-human': {'n_sg_ilr': 4}},
        'lextype': {'NYT-2023-human': {'n_-_c_le': 7}},
    }
    new_data = {
        'constr':  {'NYT-2025-human': {'sb-hd_mc_c': 3}},  # no hd-pct_c
        'lexrule': {'NYT-2025-human': {'n_sg_ilr': 2}},
        'lextype': {'NYT-2025-human': {'n_-_c_le': 5}},
    }
    add_new_dataset(frequencies, new_data, 'NYT-2025-human', model_name='NYT-2025-human')
    print(f"\n  Input:    new_data has no hd-pct_c; existing model has hd-pct_c=5")
    print(f"  Expected: frequencies['constr']['NYT-2025-human']['hd-pct_c'] = 0")
    print(f"  Actual:   {frequencies['constr']['NYT-2025-human'].get('hd-pct_c')}")
    assert frequencies['constr']['NYT-2025-human'].get('hd-pct_c') == 0, (
        f"Missing rule hd-pct_c should be filled with 0, got: {frequencies['constr']['NYT-2025-human']}"
    )


# ---------------------------------------------------------------------------
# map_word2membership
# ---------------------------------------------------------------------------

def test_map_word2membership_high():
    """Words in high_membership map to ('high', lextype)."""
    high = {'n_-_c_le': ['apple', 'tree']}
    low = {}
    singletons = {}
    word2membership = map_word2membership(high, low, singletons)
    print(f"\n  Input:    high_membership = {{n_-_c_le: [apple, tree]}}")
    print(f"  Expected: apple -> ('high', 'n_-_c_le')")
    print(f"  Actual:   {word2membership.get('apple')}")
    assert word2membership['apple'] == ('high', 'n_-_c_le')
    assert word2membership['tree'] == ('high', 'n_-_c_le')


def test_map_word2membership_low():
    """Words in low_membership map to ('low', lextype)."""
    high = {}
    low = {'v_e_le': ['run', 'jump']}
    singletons = {}
    word2membership = map_word2membership(high, low, singletons)
    print(f"\n  Input:    low_membership = {{v_e_le: [run, jump]}}")
    print(f"  Expected: run -> ('low', 'v_e_le')")
    print(f"  Actual:   {word2membership.get('run')}")
    assert word2membership['run'] == ('low', 'v_e_le')


def test_map_word2membership_singleton():
    """Words in singletons map to ('singleton', lextype)."""
    high = {}
    low = {}
    singletons = {'n_-_c_le': ['hapax']}
    word2membership = map_word2membership(high, low, singletons)
    print(f"\n  Input:    singletons = {{n_-_c_le: [hapax]}}")
    print(f"  Expected: hapax -> ('singleton', 'n_-_c_le')")
    print(f"  Actual:   {word2membership.get('hapax')}")
    assert word2membership['hapax'] == ('singleton', 'n_-_c_le')


# ---------------------------------------------------------------------------
# add_membership_to_freq
# ---------------------------------------------------------------------------

def test_add_membership_to_freq_high():
    """A word with high membership is added to lexentries[model]['high membership'][lt]."""
    lexentries = {'model_a': {'high membership': {}, 'low membership': {}, 'singletons': {}}}
    word2membership = {'apple': ('high', 'n_-_c_le')}
    add_membership_to_freq(lexentries, 'apple', 'model_a', word2membership)
    print(f"\n  Input:    word='apple', membership=('high', 'n_-_c_le')")
    print(f"  Expected: 'apple' in lexentries['model_a']['high membership']['n_-_c_le']")
    print(f"  Actual:   {lexentries['model_a']['high membership']}")
    assert 'apple' in lexentries['model_a']['high membership']['n_-_c_le']


def test_add_membership_to_freq_low():
    """A word with low membership is added to lexentries[model]['low membership'][lt]."""
    lexentries = {'model_a': {'high membership': {}, 'low membership': {}, 'singletons': {}}}
    word2membership = {'hapax': ('low', 'v_e_le')}
    add_membership_to_freq(lexentries, 'hapax', 'model_a', word2membership)
    print(f"\n  Input:    word='hapax', membership=('low', 'v_e_le')")
    print(f"  Expected: 'hapax' in lexentries['model_a']['low membership']['v_e_le']")
    print(f"  Actual:   {lexentries['model_a']['low membership']}")
    assert 'hapax' in lexentries['model_a']['low membership']['v_e_le']


def test_add_membership_to_freq_singleton():
    """A word with singleton membership is added to lexentries[model]['singletons'][lt]."""
    lexentries = {'model_a': {'high membership': {}, 'low membership': {}, 'singletons': {}}}
    word2membership = {'xeno': ('singleton', 'n_-_c_le')}
    add_membership_to_freq(lexentries, 'xeno', 'model_a', word2membership)
    print(f"\n  Input:    word='xeno', membership=('singleton', 'n_-_c_le')")
    print(f"  Expected: 'xeno' in lexentries['model_a']['singletons']['n_-_c_le']")
    print(f"  Actual:   {lexentries['model_a']['singletons']}")
    assert 'xeno' in lexentries['model_a']['singletons']['n_-_c_le']


# ---------------------------------------------------------------------------
# combine_lextype_datasets
# ---------------------------------------------------------------------------

def test_combine_lextype_datasets_merges_sets():
    """combine_lextype_datasets unions the word lists for the same (membership, lextype) key."""
    data = {
        'wsj-a': {'high membership': {'n_-_c_le': ['apple', 'tree']},
                  'low membership': {}, 'singletons': {}},
        'wsj-b': {'high membership': {'n_-_c_le': ['tree', 'stone']},
                  'low membership': {}, 'singletons': {}},
    }
    combined = combine_lextype_datasets(data, ['wsj-a', 'wsj-b'])
    result_set = set(combined['high membership']['n_-_c_le'])
    print(f"\n  Input:    wsj-a has [apple, tree], wsj-b has [tree, stone] under n_-_c_le/high")
    print(f"  Expected: combined = {{apple, tree, stone}}")
    print(f"  Actual:   {result_set}")
    assert result_set == {'apple', 'tree', 'stone'}, (
        f"Expected union {{apple, tree, stone}}, got: {result_set}"
    )


# ---------------------------------------------------------------------------
# compare_with_other_datasets
# ---------------------------------------------------------------------------

def test_compare_with_other_datasets_difference_larger():
    """When selected_dataset is more dissimilar from another than those two are from each other,
    the comparison result is True."""
    cosines = {
        ('human_a', 'llm_b'): 0.7,   # selected vs other
        ('human_a', 'human_c'): 0.95, # selected vs another
        ('llm_b', 'human_c'): 0.99,   # between non-selected
    }
    result = compare_with_other_datasets('human_a', cosines)
    print(f"\n  Input:    human_a vs llm_b cosine=0.7; llm_b vs human_c cosine=0.99")
    print(f"  Expected: result['llm_b'][0] = True (human_a more different from llm_b than others are)")
    print(f"  Actual:   {result.get('llm_b')}")
    assert result['llm_b'][0] is True, (
        f"Expected True (0.30 > 0.01), got: {result['llm_b']}"
    )


def test_compare_with_other_datasets_difference_not_larger():
    """When selected_dataset is not more dissimilar, the comparison result is False."""
    cosines = {
        ('human_a', 'llm_b'): 0.98,   # very similar
        ('human_a', 'human_c'): 0.95,
        ('llm_b', 'human_c'): 0.60,    # very dissimilar between the others
    }
    result = compare_with_other_datasets('human_a', cosines)
    print(f"\n  Input:    human_a vs llm_b cosine=0.98; llm_b vs human_c cosine=0.60")
    print(f"  Expected: result['llm_b'][0] = False (human_a NOT more different from llm_b)")
    print(f"  Actual:   {result.get('llm_b')}")
    assert result['llm_b'][0] is False, (
        f"Expected False (0.02 < 0.40), got: {result['llm_b']}"
    )


# ---------------------------------------------------------------------------
# compare_human_vs_machine
# ---------------------------------------------------------------------------

def test_compare_human_vs_machine_groups_correctly():
    """compare_human_vs_machine separates similarities into human vs machine buckets."""
    cosines = {
        ('human_a', 'human_b'): 0.95,
        ('human_a', 'llm_c'):   0.75,
        ('llm_c', 'llm_d'):     0.90,
    }
    avg_machine, avg_with_machine, similarities_with_human = compare_human_vs_machine(
        'human_a', cosines, ['human_b'], ['llm_c', 'llm_d']
    )
    print(f"\n  Input:    human_a, human datasets=[human_b], machine=[llm_c, llm_d]")
    print(f"  Expected: similarities_with_human = {{human_b: 0.95}}")
    print(f"            avg_with_machine = 0.75")
    print(f"  Actual:   human={similarities_with_human}, avg_with_machine={avg_with_machine}")
    assert similarities_with_human == {'human_b': 0.95}, (
        f"Expected human_b: 0.95 in with_human, got: {similarities_with_human}"
    )
    assert abs(avg_with_machine - 0.75) < 1e-9, (
        f"Expected avg_with_machine=0.75, got: {avg_with_machine}"
    )


def test_compare_human_vs_machine_avg_machine_similarity():
    """avg_machine_similarity is computed from pairwise machine-machine similarities."""
    cosines = {
        ('human_a', 'llm_c'):   0.80,
        ('llm_c', 'llm_d'):     0.90,
        ('llm_c', 'llm_e'):     0.70,
        ('llm_d', 'llm_e'):     0.85,
    }
    avg_machine, avg_with_machine, _ = compare_human_vs_machine(
        'human_a', cosines, [], ['llm_c', 'llm_d', 'llm_e']
    )
    expected_avg = (0.90 + 0.70 + 0.85) / 3
    print(f"\n  Input:    machine pairwise cosines: 0.90, 0.70, 0.85")
    print(f"  Expected: avg_machine = {expected_avg:.4f}")
    print(f"  Actual:   avg_machine = {avg_machine:.4f}")
    assert abs(avg_machine - expected_avg) < 1e-9, (
        f"Expected avg_machine={expected_avg:.4f}, got: {avg_machine}"
    )


# ---------------------------------------------------------------------------
# find_absolute_diffs_lextype
# ---------------------------------------------------------------------------

def test_find_absolute_diffs_lextype_same_words():
    """Words present in both models under the same membership/lextype go into both['same']."""
    model1 = {'high membership': {'n_-_c_le': ['apple', 'tree', 'stone']},
               'low membership': {}, 'singletons': {}}
    model2 = {'high membership': {'n_-_c_le': ['apple', 'stone', 'river']},
               'low membership': {}, 'singletons': {}}
    both, only_one = find_absolute_diffs_lextype(model1, model2, 'm1', 'm2')
    same = set(both['high membership']['n_-_c_le']['same'])
    print(f"\n  Input:    m1=[apple, tree, stone], m2=[apple, stone, river] for n_-_c_le/high")
    print(f"  Expected: same = {{apple, stone}}")
    print(f"  Actual:   same = {same}")
    assert same == {'apple', 'stone'}, (
        f"Expected {{apple, stone}} in same, got: {same}"
    )


def test_find_absolute_diffs_lextype_words_only_in_one():
    """Words in m1 but not m2 go into both['different']['m1']; vice versa for m2."""
    model1 = {'high membership': {'n_-_c_le': ['apple', 'tree']},
               'low membership': {}, 'singletons': {}}
    model2 = {'high membership': {'n_-_c_le': ['apple', 'river']},
               'low membership': {}, 'singletons': {}}
    both, only_one = find_absolute_diffs_lextype(model1, model2, 'm1', 'm2')
    diff_m1 = set(both['high membership']['n_-_c_le']['different']['m1'])
    diff_m2 = set(both['high membership']['n_-_c_le']['different']['m2'])
    print(f"\n  Input:    m1=[apple, tree], m2=[apple, river] for n_-_c_le/high")
    print(f"  Expected: diff_m1 = {{tree}}, diff_m2 = {{river}}")
    print(f"  Actual:   diff_m1 = {diff_m1}, diff_m2 = {diff_m2}")
    assert diff_m1 == {'tree'}, f"Expected {{tree}} only in m1, got: {diff_m1}"
    assert diff_m2 == {'river'}, f"Expected {{river}} only in m2, got: {diff_m2}"


def test_find_absolute_diffs_lextype_only_in_one_model():
    """A lextype present in m1 but absent from m2 goes into only_one['m1']."""
    model1 = {'high membership': {'n_-_c_le': ['apple'], 'v_e_le': ['run']},
               'low membership': {}, 'singletons': {}}
    model2 = {'high membership': {'n_-_c_le': ['apple']},
               'low membership': {}, 'singletons': {}}
    both, only_one = find_absolute_diffs_lextype(model1, model2, 'm1', 'm2')
    print(f"\n  Input:    m1 has v_e_le but m2 does not")
    print(f"  Expected: only_one['m1']['high membership']['v_e_le'] = ['run']")
    print(f"  Actual:   {only_one.get('m1', {}).get('high membership')}")
    assert 'v_e_le' in only_one['m1']['high membership'], (
        f"Expected v_e_le (only in m1) in only_one['m1'], got: {only_one['m1']}"
    )
    assert only_one['m1']['high membership']['v_e_le'] == ['run']


# ---------------------------------------------------------------------------
# compare_lexentries
# ---------------------------------------------------------------------------

def test_compare_lexentries_only_in_llm():
    """Lexentries present in an LLM but not in any human go into only_in_llm."""
    data = {
        'NYT-2023-human':  {'apple': 5, 'tree': 3},
        'Falcon7B-2023-llm': {'apple': 4, 'robot': 7},  # 'robot' not in human
    }
    only_in_llm, only_in_human = compare_lexentries(data)
    print(f"\n  Input:    human has {{apple, tree}}, LLM has {{apple, robot}}")
    print(f"  Expected: 'robot' in only_in_llm['Falcon7B-2023-llm']")
    print(f"  Actual:   {only_in_llm.get('Falcon7B-2023-llm')}")
    assert 'robot' in only_in_llm['Falcon7B-2023-llm'], (
        f"Expected 'robot' (exclusive to LLM) in only_in_llm, got: {only_in_llm['Falcon7B-2023-llm']}"
    )


def test_compare_lexentries_only_in_human():
    """Lexentries present in human but not in any LLM go into only_in_human."""
    data = {
        'NYT-2023-human':    {'apple': 5, 'nuance': 1},  # 'nuance' not in LLM
        'Falcon7B-2023-llm': {'apple': 4},
    }
    only_in_llm, only_in_human = compare_lexentries(data)
    print(f"\n  Input:    human has {{apple, nuance}}, LLM has {{apple}}")
    print(f"  Expected: 'nuance' in only_in_human['not in any llm']")
    print(f"  Actual:   {only_in_human.get('not in any llm')}")
    assert 'nuance' in only_in_human['not in any llm'], (
        f"Expected 'nuance' (exclusive to human) in only_in_human, got: {only_in_human['not in any llm']}"
    )


def test_compare_lexentries_shared_entry_absent_from_exclusive():
    """Lexentries present in both human and LLM should not appear in either exclusive set."""
    data = {
        'NYT-2023-human':    {'apple': 5},
        'Falcon7B-2023-llm': {'apple': 4},
    }
    only_in_llm, only_in_human = compare_lexentries(data)
    print(f"\n  Input:    'apple' present in both human and LLM")
    print(f"  Expected: 'apple' absent from both exclusive sets")
    print(f"  only_in_llm['Falcon7B-2023-llm']:  {only_in_llm.get('Falcon7B-2023-llm')}")
    print(f"  only_in_human['not in any llm']:    {only_in_human.get('not in any llm')}")
    assert 'apple' not in only_in_llm.get('Falcon7B-2023-llm', {}), (
        f"Shared entry 'apple' must not appear in only_in_llm"
    )
    assert 'apple' not in only_in_human.get('not in any llm', {}), (
        f"Shared entry 'apple' must not appear in only_in_human"
    )


# ---------------------------------------------------------------------------
# build_llm_vs_human
# ---------------------------------------------------------------------------

def _make_freq_for_combined():
    """Minimal frequencies dict with two LLMs and one human model."""
    return {
        'constr':  {'Falcon7B-2023-llm': {'sb-hd_mc_c': 10, 'hd-pct_c': 5},
                    'Llama7B-2023-llm':  {'sb-hd_mc_c':  8, 'hd-pct_c': 2},
                    'NYT-2023-human':    {'sb-hd_mc_c': 20, 'hd-pct_c': 9}},
        'lexrule': {'Falcon7B-2023-llm': {'n_sg_ilr': 4},
                    'Llama7B-2023-llm':  {'n_sg_ilr': 3},
                    'NYT-2023-human':    {'n_sg_ilr': 7}},
        'lextype': {'Falcon7B-2023-llm': {'n_-_c_le': 6},
                    'Llama7B-2023-llm':  {'n_-_c_le': 5},
                    'NYT-2023-human':    {'n_-_c_le': 11}},
    }


def test_build_llm_vs_human_contains_llm_key():
    """Result has an 'llm' key in each rule type."""
    result = build_llm_vs_human(_make_freq_for_combined())
    for rt in result:
        assert 'llm' in result[rt], f"'llm' key missing from rule type '{rt}'"


def test_build_llm_vs_human_sums_llm_counts():
    """Counts for the 'llm' entry are the sum across all LLM models."""
    freq = _make_freq_for_combined()
    result = build_llm_vs_human(freq)
    expected = (freq['constr']['Falcon7B-2023-llm']['sb-hd_mc_c']
                + freq['constr']['Llama7B-2023-llm']['sb-hd_mc_c'])
    actual = result['constr']['llm']['sb-hd_mc_c']
    print(f"\n  Input:    Falcon=10, Llama=8")
    print(f"  Expected: llm['sb-hd_mc_c'] = {expected}")
    print(f"  Actual:   {actual}")
    assert actual == expected


def test_build_llm_vs_human_contains_human_models():
    """Human baseline models are preserved in the result."""
    result = build_llm_vs_human(_make_freq_for_combined())
    assert 'NYT-2023-human' in result['constr'], "Human model 'NYT-2023-human' missing from result"


def test_build_llm_vs_human_excludes_individual_llms():
    """Individual LLM model keys are not present — only the aggregate 'llm'."""
    result = build_llm_vs_human(_make_freq_for_combined())
    for llm in ('Falcon7B-2023-llm', 'Llama7B-2023-llm'):
        assert llm not in result['constr'], f"Individual LLM '{llm}' should not appear in result"


def test_build_llm_vs_human_does_not_mutate_input():
    """Input frequencies dict is not modified."""
    freq = _make_freq_for_combined()
    original_count = freq['constr']['Falcon7B-2023-llm']['sb-hd_mc_c']
    build_llm_vs_human(freq)
    assert freq['constr']['Falcon7B-2023-llm']['sb-hd_mc_c'] == original_count, "Input was mutated"


# ---------------------------------------------------------------------------
# Regression: cosine output on frequencies-2023.json matches reference
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists(_FREQ_2023) or not os.path.exists(_REF_SYNTAX),
    reason="Reference data files not present"
)
def test_cosine_syntax_matches_reference():
    """Cosine similarities from frequencies-2023.json match the pre-refactor reference output.

    The reference file uses old model names; _NAME_MAP translates new names to old for comparison.
    """
    from util import normalize_by_constr_count, compute_cosine
    from construction_frequencies import combine_types

    with open(_FREQ_2023) as f:
        data = json.load(f)

    normalized = normalize_by_constr_count(data)
    only_syntactic = combine_types(normalized, ['constr'])
    cosines = compute_cosine(only_syntactic)

    with open(_REF_SYNTAX) as f:
        ref_raw = json.load(f)

    # Reference keys are stringified tuples, e.g. "('falcon_07', 'llama_07')"
    ref = {frozenset(eval(k)): float(v) for k, v in ref_raw.items()}

    mismatches = []
    for (m1, m2), val in cosines.items():
        old1, old2 = _NAME_MAP[m1], _NAME_MAP[m2]
        key = frozenset([old1, old2])
        if key not in ref:
            mismatches.append(f"Pair ({m1}, {m2}) not found in reference")
            continue
        expected = ref[key]
        if abs(val - expected) > 1e-6:
            mismatches.append(f"({m1},{m2}): got {val:.8f}, expected {expected:.8f}")

    assert not mismatches, "Cosine regression failures:\n" + "\n".join(mismatches)
