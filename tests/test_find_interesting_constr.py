import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import pytest
from find_interesting_constr import (
    is_human, is_nyt, merge_frequencies,
    find_hapax_constr, find_significant_constr,
)


# ---------------------------------------------------------------------------
# is_human / is_nyt
# ---------------------------------------------------------------------------

def test_is_human_nyt():
    result = is_human('NYT-2023-human')
    print(f"\n  Input:    'NYT-2023-human'")
    print(f"  Expected: True")
    print(f"  Actual:   {result}")
    assert result is True

def test_is_human_wsj():
    result = is_human('WSJ-1987-human')
    print(f"\n  Input:    'WSJ-1987-human'")
    print(f"  Expected: True")
    print(f"  Actual:   {result}")
    assert result is True

def test_is_human_wikipedia():
    result = is_human('Wikipedia-2008-human')
    print(f"\n  Input:    'Wikipedia-2008-human'")
    print(f"  Expected: True")
    print(f"  Actual:   {result}")
    assert result is True

def test_is_human_llm():
    result = is_human('Llama7B-2023-llm')
    print(f"\n  Input:    'Llama7B-2023-llm'")
    print(f"  Expected: False")
    print(f"  Actual:   {result}")
    assert result is False

def test_is_nyt_true():
    result = is_nyt('NYT-2023-human')
    print(f"\n  Input:    'NYT-2023-human'")
    print(f"  Expected: True")
    print(f"  Actual:   {result}")
    assert result is True

def test_is_nyt_wsj_is_false():
    result = is_nyt('WSJ-1987-human')
    print(f"\n  Input:    'WSJ-1987-human'  (human but not NYT)")
    print(f"  Expected: False")
    print(f"  Actual:   {result}")
    assert result is False

def test_is_nyt_llm_is_false():
    result = is_nyt('NYT-2023-llm')
    print(f"\n  Input:    'NYT-2023-llm'  (NYT but not human)")
    print(f"  Expected: False")
    print(f"  Actual:   {result}")
    assert result is False


# ---------------------------------------------------------------------------
# merge_frequencies
# ---------------------------------------------------------------------------

def test_merge_frequencies_combines_models(tmp_path):
    f1 = tmp_path / 'a.json'
    f2 = tmp_path / 'b.json'
    f1.write_text(json.dumps({'constr': {'NYT-2023-human': {'sb-hd_mc_c': 10}}}))
    f2.write_text(json.dumps({'constr': {'Llama7B-2023-llm': {'sb-hd_mc_c': 5}}}))
    result = merge_frequencies(str(f1), str(f2))
    print(f"\n  Input:    f1 has NYT-2023-human, f2 has Llama7B-2023-llm")
    print(f"  Expected: both models present in result['constr']")
    print(f"  Actual:   models = {sorted(result['constr'].keys())}")
    assert 'NYT-2023-human'   in result['constr']
    assert 'Llama7B-2023-llm' in result['constr']


def test_merge_frequencies_unions_phenomena(tmp_path):
    f1 = tmp_path / 'a.json'
    f2 = tmp_path / 'b.json'
    f1.write_text(json.dumps({'constr':   {'NYT-2023-human': {'type_a': 1}}}))
    f2.write_text(json.dumps({'lextype':  {'NYT-2023-human': {'type_b': 2}}}))
    result = merge_frequencies(str(f1), str(f2))
    print(f"\n  Input:    f1 has 'constr', f2 has 'lextype'")
    print(f"  Expected: both phenomena in result")
    print(f"  Actual:   phenomena = {sorted(result.keys())}")
    assert 'constr'  in result
    assert 'lextype' in result


def test_merge_frequencies_later_file_overwrites(tmp_path):
    f1 = tmp_path / 'a.json'
    f2 = tmp_path / 'b.json'
    f1.write_text(json.dumps({'constr': {'NYT-2023-human': {'type_a': 1}}}))
    f2.write_text(json.dumps({'constr': {'NYT-2023-human': {'type_a': 99}}}))
    result = merge_frequencies(str(f1), str(f2))
    actual = result['constr']['NYT-2023-human']['type_a']
    print(f"\n  Input:    f1 sets type_a=1, f2 sets type_a=99 (same model)")
    print(f"  Expected: type_a = 99  (f2 overwrites f1)")
    print(f"  Actual:   type_a = {actual}")
    assert actual == 99


# ---------------------------------------------------------------------------
# find_hapax_constr
#
# type_a: hapax in NYT-human (count=0), common in LLMs (count=20) → mismatch
# type_b: common in both → no mismatch
# type_c: hapax in both → no mismatch
# ---------------------------------------------------------------------------

def _hapax_fixture():
    return {
        'constr': {
            'NYT-2023-human':   {'type_a': 0,  'type_b': 10, 'type_c': 1},
            'Llama7B-2023-llm': {'type_a': 20, 'type_b': 8,  'type_c': 1},
        }
    }


def test_find_hapax_constr_detects_mismatch():
    result = find_hapax_constr(_hapax_fixture(), infrequent_threshold=2)
    print(f"\n  Input:    type_a: NYT=0 (hapax), LLM=20 (common)")
    print(f"  Expected: 'type_a' in result['constr']")
    print(f"  Actual:   constr keys = {sorted(result['constr'].keys())}")
    assert 'type_a' in result['constr']


def test_find_hapax_constr_ignores_common_in_both():
    result = find_hapax_constr(_hapax_fixture(), infrequent_threshold=2)
    print(f"\n  Input:    type_b: NYT=10 (common), LLM=8 (common)")
    print(f"  Expected: 'type_b' NOT in result['constr']")
    print(f"  Actual:   constr keys = {sorted(result['constr'].keys())}")
    assert 'type_b' not in result['constr']


def test_find_hapax_constr_ignores_hapax_in_both():
    result = find_hapax_constr(_hapax_fixture(), infrequent_threshold=2)
    print(f"\n  Input:    type_c: NYT=1 (hapax), LLM=1 (hapax)")
    print(f"  Expected: 'type_c' NOT in result['constr']")
    print(f"  Actual:   constr keys = {sorted(result['constr'].keys())}")
    assert 'type_c' not in result['constr']


def test_find_hapax_constr_records_counts():
    result = find_hapax_constr(_hapax_fixture(), infrequent_threshold=2)
    human_count = result['constr']['type_a']['human count']
    llm_count   = result['constr']['type_a']['llm count']
    print(f"\n  Input:    type_a: NYT-human=0, Llama7B-llm=20")
    print(f"  Expected: human count=0, llm count=20")
    print(f"  Actual:   human count={human_count}, llm count={llm_count}")
    assert human_count == 0
    assert llm_count   == 20


def test_find_hapax_constr_uses_nyt_models_only():
    """WSJ-human should not count as the human baseline for hapax."""
    data = {
        'constr': {
            'WSJ-1987-human':   {'type_a': 50},  # high count, non-NYT human
            'NYT-2023-human':   {'type_a': 0},   # hapax in NYT
            'Llama7B-2023-llm': {'type_a': 20},
        }
    }
    result = find_hapax_constr(data, infrequent_threshold=2)
    human_count = result['constr']['type_a']['human count']
    print(f"\n  Input:    WSJ-human=50 (not NYT), NYT-human=0, LLM=20")
    print(f"  Expected: 'type_a' detected as mismatch; human count=0 (WSJ excluded)")
    print(f"  Actual:   type_a in result={('type_a' in result['constr'])}, human count={human_count}")
    # NYT count is 0, LLM count is 20 → mismatch detected (WSJ not included)
    assert 'type_a' in result['constr']
    assert human_count == 0


# ---------------------------------------------------------------------------
# find_significant_constr — structure and basic behaviour
# ---------------------------------------------------------------------------

def _significant_fixture():
    """Three human and three LLM models; type_a is very different between groups."""
    humans = ['NYT-2023-human', 'WSJ-1987-human', 'Wikipedia-2008-human']
    llms   = ['Llama7B-2023-llm', 'Llama13B-2023-llm', 'Falcon7B-2023-llm']
    freq = {'constr': {}}
    for m in humans:
        freq['constr'][m] = {'type_a': 0.001, 'type_b': 0.5}
    for m in llms:
        freq['constr'][m] = {'type_a': 0.999, 'type_b': 0.5}
    return freq


def test_find_significant_constr_output_structure():
    result = find_significant_constr(_significant_fixture())
    print(f"\n  Input:    fixture with 3 human + 3 LLM models")
    print(f"  Expected: each phenomenon has 'frequent' and 'infrequent' keys")
    print(f"  Actual:   top-level keys = {sorted(result.keys())}, "
          f"constr keys = {sorted(result.get('constr', {}).keys())}")
    for phenomenon in result:
        assert 'frequent'   in result[phenomenon]
        assert 'infrequent' in result[phenomenon]


def test_find_significant_constr_detects_difference():
    """type_a has maximally different human vs LLM values and should be significant."""
    result = find_significant_constr(_significant_fixture())
    found = any(
        'type_a' in result[p][tier]
        for p in result
        for tier in ('frequent', 'infrequent')
    )
    print(f"\n  Input:    type_a: humans=0.001, LLMs=0.999 (maximally different)")
    print(f"  Expected: 'type_a' flagged as significant")
    print(f"  Actual:   found={found}")
    assert found


def test_find_significant_constr_identical_values_not_significant():
    """type_b has identical values across all models and should not be significant."""
    result = find_significant_constr(_significant_fixture())
    found = any(
        'type_b' in result[p][tier]
        for p in result
        for tier in ('frequent', 'infrequent')
    )
    print(f"\n  Input:    type_b: all models=0.5 (identical)")
    print(f"  Expected: 'type_b' NOT flagged as significant")
    print(f"  Actual:   found={found}")
    assert not found


# ---------------------------------------------------------------------------
# find_significant_constr — non-uniform distributions within groups
#
# type_x: humans ~[0.10, 0.12, 0.15], LLMs ~[0.80, 0.85, 0.90]
#         Groups are completely separated → should be significant.
# type_y: humans ~[0.30, 0.38, 0.45], LLMs ~[0.35, 0.40, 0.42]
#         Ranges overlap → should NOT be significant.
#         (With n=3 per group, Mann-Whitney p cannot reach 0.05 when ranks interleave.)
# ---------------------------------------------------------------------------

def _uneven_fixture():
    """Models with varied (non-uniform) values within each group.

    Uses 4 models per group: with n=3 the two-sided Mann-Whitney minimum p-value
    is ~0.08, which never reaches 0.05 even for perfectly separated groups.
    With n=4 the minimum p-value drops to ~0.03, enabling real significance detection.

    type_x: humans ~[0.10–0.13], LLMs ~[0.80–0.90] — fully separated → significant.
    type_y: humans ~[0.30–0.45], LLMs ~[0.35–0.43] — interleaved ranks → not significant.
    """
    return {'constr': {
        'NYT-2023-human':        {'type_x': 0.10, 'type_y': 0.30},
        'WSJ-1987-human':        {'type_x': 0.13, 'type_y': 0.45},
        'Wikipedia-2008-human':  {'type_x': 0.12, 'type_y': 0.38},
        'Brown-2000-human':      {'type_x': 0.11, 'type_y': 0.42},
        'Llama7B-2023-llm':      {'type_x': 0.80, 'type_y': 0.35},
        'Llama13B-2023-llm':     {'type_x': 0.85, 'type_y': 0.41},
        'Falcon7B-2023-llm':     {'type_x': 0.88, 'type_y': 0.43},
        'GPT4-2023-llm':         {'type_x': 0.90, 'type_y': 0.37},
    }}


def test_find_significant_constr_uneven_detects_separated_groups():
    """type_x has non-uniform but fully separated human/LLM ranges → significant."""
    result = find_significant_constr(_uneven_fixture())
    found = any(
        'type_x' in result[p][tier]
        for p in result
        for tier in ('frequent', 'infrequent')
    )
    print(f"\n  Input:    type_x: humans~[0.10,0.12,0.15], LLMs~[0.80,0.85,0.90] (no overlap)")
    print(f"  Expected: 'type_x' flagged as significant")
    print(f"  Actual:   found={found}")
    assert found


def test_find_significant_constr_uneven_overlapping_not_significant():
    """type_y has overlapping human/LLM ranges → not significant."""
    result = find_significant_constr(_uneven_fixture())
    found = any(
        'type_y' in result[p][tier]
        for p in result
        for tier in ('frequent', 'infrequent')
    )
    print(f"\n  Input:    type_y: humans~[0.30,0.38,0.45], LLMs~[0.35,0.40,0.42] (overlapping)")
    print(f"  Expected: 'type_y' NOT flagged as significant")
    print(f"  Actual:   found={found}")
    assert not found
