# Given a lexical type, search in the ERG grammar files for its supertypes, for N levels of the hierarchy

import os
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


if __name__ == "__main__":
    directory = '/home/olga/delphin/erg/trunk'
    type_name = 'pp_-_i-dir-novmd_le'  # Replace with the type you're interested in
    lex = populate_type_defs(directory)
    supertypes = get_n_supertypes(lex, type_name, 3)
    print(supertypes)