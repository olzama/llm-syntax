import sys, os
import re
from delphin import tdl
from copy import copy, deepcopy

def get_n_supertypes(lex, type_name, n):
    # Helper function to recursively retrieve supertypes
    def get_supertypes_recursive(type_name, remaining_depth, level=0, depth_dict=None):
        if remaining_depth == 0 or type_name not in lex:
            return depth_dict  # Return depth_dict if no more recursion is needed
        # Initialize the depth_dict if it's the first call
        if depth_dict is None:
            depth_dict = {}
        # Retrieve the supertypes for the current type and convert them to strings
        supertypes = [str(supertype) for supertype in lex[type_name].supertypes]
        # Add combined supertypes for the current level (level starts from 0)
        combined_same_level_supertypes = '+'.join(supertypes)
        # If level doesn't exist in the dict, create a new list for this level
        if level not in depth_dict:
            depth_dict[level] = set()  # Using a set to avoid duplicates
        # Add the combined supertypes for this level (as a string)
        depth_dict[level].add(combined_same_level_supertypes)
        # If there's more depth, recurse for each supertype and increase the level
        if remaining_depth > 1:
            for supertype in supertypes:
                get_supertypes_recursive(str(supertype), remaining_depth - 1, level + 1, depth_dict)
        return depth_dict

    # Start the recursive process
    depth_dict = get_supertypes_recursive(type_name, n)
    # Return the depth_dict with combined supertypes by levels
    return depth_dict

def populate_type_defs(directory):
    global lex
    # Initialize an empty dictionary to store the lexicon
    lex = {}
    # Iterate through all files in the directory with the .tdl extension
    for filename in os.listdir(directory):
        if filename.endswith('.tdl'):
            file_path = os.path.join(directory, filename)
            # Parse the TDL file
            for event, obj, lineno in tdl.iterparse(file_path):
                if event == 'TypeDefinition':
                    # Add the object to the lexicon
                    lex[obj.identifier] = obj
    # sort the lexicon by type name
    lex = dict(sorted(lex.items()))
    return lex


def dict_to_latex_table(data, include):
    latex_table = """
\\begin{table}[ht]
\\centering
\\begin{tabular}{|l|l|l|}
\\hline
\\textbf{Type Name} & \\textbf{Definition} & \\textbf{Example} \\\\
\\hline
"""

    for type_name, details in data.items():
        if type_name in include:
            escaped_type_name = type_name.replace("_", "\\_")
            latex_table += f"\\textbf{{{escaped_type_name}}} & {details['def']} & {details['ex']} \\\\ \n"

    latex_table += """
\\hline
\\end{tabular}
\\caption{Construction Types and Examples}
\\end{table}
"""
    return latex_table


def create_friendly_name(tdl_obj):
    documentation = tdl_obj.documentation()
    definition = ''
    example = ''
    if documentation:
        docstring = documentation.strip()
        docstring_parts = docstring.split('<ex>') if '<ex>' in docstring else [docstring, '']
        example = docstring_parts[1]
        definition = docstring_parts[0].strip()
    return (definition, example)

def types2defs(grammar_dir):
    mapping = {}
    for tdl_file in os.listdir(grammar_dir):
        filepath = os.path.join(grammar_dir, tdl_file)
        for event, obj, lineno in tdl.iterparse(filepath):
            if event == 'TypeDefinition':
                (friendly_def, example) = create_friendly_name(obj)
                if obj.identifier not in mapping:
                    mapping[obj.identifier] = {'def': friendly_def, 'ex': example}
    return mapping

def lexical_types(erg_dir):
    high_membership = {}
    low_membership = {}
    return high_membership, low_membership

if __name__ == '__main__':
    erg_dir = sys.argv[1]
    mapping = types2defs(erg_dir)
    with open('/mnt/kesha/llm-syntax/analysis/constructions/top_constr_list.txt', 'r') as f:
        include = [ln.strip() for ln in f.readlines()]
    latex_table = dict_to_latex_table(mapping, include)
    with open('/mnt/kesha/llm-syntax/analysis/latex/appendix-erg.txt', 'w') as f:
        f.write(latex_table + '\n')
    # directory = '/home/olga/delphin/erg/trunk'
    # type_name = 'pp_-_i-dir-novmd_le'  # Replace with the type you're interested in
    # lex = populate_type_defs(directory)
    # supertypes = get_n_supertypes(lex, type_name, 3)
    # print(supertypes)
