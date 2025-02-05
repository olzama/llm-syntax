import sys, os
import re

from delphin import tdl


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

if __name__ == '__main__':
    erg_dir = sys.argv[1]
    mapping = types2defs(erg_dir)
    with open('/mnt/kesha/llm-syntax/analysis/constructions/top_constr_list.txt', 'r') as f:
        include = [ln.strip() for ln in f.readlines()]
    latex_table = dict_to_latex_table(mapping, include)
    with open('/mnt/kesha/llm-syntax/analysis/latex/appendix-erg.txt', 'w') as f:
        f.write(latex_table + '\n')
