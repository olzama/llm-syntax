"""
erg.py — utilities for reading and querying the English Resource Grammar (ERG).

Reads:
  - ERG TDL files (e.g. constructions.tdl, lextypes.tdl, lexicon.tdl)

Imported by:
  - count_constructions.py  (populate_type_defs, get_n_supertypes, classify_node)
  - extract_examples.py     (populate_type_defs, classify_node)
  - extract_ex_JSD.py       (populate_type_defs, get_n_supertypes, read_lexicon)
  - construction_frequencies.py (get_n_supertypes, populate_type_defs, read_lexicon,
                                  types2defs, lexical_types)
"""

import os
from delphin import tdl
import numpy as np

_TYPED_TDL_FILES = {
    'constructions.tdl', 'letypes.tdl', 'lexrinst.tdl', 'inflr.tdl',
    'lexrinst-tok.tdl', 'lextypes.tdl', 'auxverbs.tdl', 'ple.tdl',
    'gle.tdl', 'gle-gen.tdl',
}


def get_n_supertypes(lex, type_name, n):
    """Return supertypes of type_name up to n levels up the hierarchy.

    Result is a dict {level: set_of_supertype_strings}, where level 0
    contains the immediate supertypes, level 1 their parents, etc.
    Multiple supertypes at the same level are joined with '+'.
    Returns None if the type is not found in lex.
    """
    def _recurse(type_name, remaining, level, depth_dict):
        if remaining == 0 or type_name not in lex:
            return depth_dict
        if depth_dict is None:
            depth_dict = {}
        supertypes = [str(st) for st in lex[type_name].supertypes]
        depth_dict.setdefault(level, set()).add('+'.join(supertypes))
        if remaining > 1:
            for st in supertypes:
                _recurse(st, remaining - 1, level + 1, depth_dict)
        return depth_dict

    return _recurse(type_name, n, 0, None)


def classify_node(node, preterminals, lex, depth):
    """Classify a derivation node and resolve its type name.

    node:         a UDFNode from a derivation tree.
    preterminals: set of entity names that are preterminals in this derivation.
    lex:          type hierarchy dict from populate_type_defs.
    depth:        supertype depth for lextype resolution.

    Returns (category, resolved_type) where:
      - category is 'lextype', 'lexrule', or 'constr'
      - resolved_type is the depth-N supertype name for lextypes, or node.entity otherwise
    """
    resolved_type = node.entity
    if node.entity in preterminals:
        category = 'lextype'
        supertypes = get_n_supertypes(lex, node.entity, depth)
        if supertypes:
            resolved_type = list(supertypes[depth - 1])[0]
    elif node.entity.endswith('lr'):
        category = 'lexrule'
    else:
        category = 'constr'
    return category, resolved_type


def populate_type_defs(directory):
    """Parse all TDL files in directory and return the full type hierarchy.

    Returns a tuple (lex, constr_types) where:
      - lex maps every type/rule identifier to its TDL object
      - constr_types groups identifiers from the known grammar files into
        'syntax' (_c suffix), 'lexrule' (lr suffix), and 'lextype' (_le suffix)
    """
    lex = {}
    constr_types = {'syntax': [], 'lexrule': [], 'lextype': []}
    for filename in os.listdir(directory):
        if not filename.endswith('.tdl'):
            continue
        file_path = os.path.join(directory, filename)
        for event, obj, lineno in tdl.iterparse(file_path):
            if event in ('TypeDefinition', 'LexicalRuleDefinition'):
                lex[obj.identifier] = obj
                if filename in _TYPED_TDL_FILES:
                    if obj.identifier.endswith('_c'):
                        constr_types['syntax'].append(obj.identifier)
                    if obj.identifier.endswith('lr'):
                        constr_types['lexrule'].append(obj.identifier)
                    if obj.identifier.endswith('_le'):
                        constr_types['lextype'].append(obj.identifier)
    return dict(sorted(lex.items())), constr_types


def read_lexicon(lexicon_files):
    """Parse lexicon TDL files and return a map from lexical type to its entries.

    Returns a dict {supertype_string: [entry_id, ...]}, sorted descending by
    number of entries (ties broken alphabetically by type name).
    """
    lex = {}
    for lexicon_file in lexicon_files:
        for event, obj, lineno in tdl.iterparse(lexicon_file):
            if event == 'TypeDefinition':
                lex.setdefault(str(obj.supertypes[0]), []).append(obj.identifier)
    return {
        key: sorted(value)
        for key, value in sorted(
            lex.items(), key=lambda item: (len(item[1]), item[0]), reverse=True
        )
    }


def dict_to_latex_table(data, include):
    """Render a subset of type definitions as a LaTeX three-column table.

    data: dict {type_name: {'def': str, 'ex': str}} as returned by types2defs.
    include: collection of type names to include (others are skipped).
    Returns a complete LaTeX table string (tabular inside table environment).
    """
    rows = []
    for type_name, details in data.items():
        if type_name in include:
            escaped = type_name.replace("_", "\\_")
            rows.append(f"\\textbf{{{escaped}}} & {details['def']} & {details['ex']} \\\\ ")
    return (
        "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{|l|l|l|}\n\\hline\n"
        "\\textbf{Type Name} & \\textbf{Definition} & \\textbf{Example} \\\\\n\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n\\end{tabular}\n\\caption{Construction Types and Examples}\n\\end{table}\n"
    )


def create_friendly_name(tdl_obj):
    """Extract a human-readable definition and example from a TDL object's docstring.

    Splits the documentation on '<ex>' to separate the prose definition from the
    example sentence. Returns (definition, example) strings; both are '' if the
    object has no documentation.
    """
    definition = ''
    example = ''
    documentation = tdl_obj.documentation()
    if documentation:
        docstring = documentation.strip()
        parts = docstring.split('<ex>') if '<ex>' in docstring else [docstring, '']
        definition = parts[0].strip()
        example = parts[1]
    return definition, example


def types2defs(grammar_dir):
    """Build a mapping from every TDL type in grammar_dir to its friendly name.

    Reads all TDL files in grammar_dir and returns a dict
    {type_id: {'def': str, 'ex': str}} using create_friendly_name to populate
    the definition and example fields. Only TypeDefinition events are included.
    """
    mapping = {}
    for tdl_file in os.listdir(grammar_dir):
        filepath = os.path.join(grammar_dir, tdl_file)
        for event, obj, lineno in tdl.iterparse(filepath):
            if event == 'TypeDefinition':
                friendly_def, example = create_friendly_name(obj)
                if obj.identifier not in mapping:
                    mapping[obj.identifier] = {'def': friendly_def, 'ex': example}
    return mapping


def lexical_types(lexicon):
    """Partition a lexicon dict into high-membership, low-membership, and singleton types.

    lexicon: dict {type_name: [entry_id, ...]} as returned by read_lexicon.
    Returns (high_membership, low_membership, singletons) where:
      - high_membership: types with entry count above the 90th percentile
      - low_membership:  types with 2–90th-percentile entries (inclusive)
      - singletons:      types with exactly 1 entry
    """
    lengths = [len(value) for value in lexicon.values()]
    high_threshold = np.percentile(lengths, 90)
    high_membership = {k: v for k, v in lexicon.items() if len(v) > high_threshold}
    low_membership  = {k: v for k, v in lexicon.items() if 1 < len(v) <= high_threshold}
    singletons      = {k: v for k, v in lexicon.items() if len(v) == 1}
    return high_membership, low_membership, singletons
