"""
Manage interaction blocks that have and haven't been commented in an input itp file
"""

from collections import defaultdict
from fast_forward.virtual_site_mass import MARTINI_MASS

def finalise_interaction_types(block_interactions, fitted_interactions):
    """
    Combine two keysets of interactions, merges them, and returns the combined
    list with a priority ordering to make sure bonds are written first into file

    Parameters
    ----------
    block_interactions: dict
        keys of interactions from a block
    fitted_interactions: dict
        keys of interactions which have been fitted

    Returns
    -------
    sorted_list: list
        interaction names sorted with priority to key interaction types
    """
    combined_set = set(block_interactions) | set(fitted_interactions)
    priority_order = ['bonds', 'constraints', 'angles', 'dihedrals']
    order_map = defaultdict(lambda: len(priority_order))  # unknown keys get max priority
    order_map.update({key: i for i, key in enumerate(priority_order)})

    sorted_list = sorted(combined_set, key=lambda x: order_map[x])
    return sorted_list

def interaction_finalising(block, fitted_interactions):
    """
    Function to write fitted interactions into a block. Ensures that interactions
    that haven't otherwise been analysed (e.g. a fragment already parameterised elsewhere)
    are not overwritten.

    Parameters
    ----------
    block: :class:`~vermouth.molecule.Block`
        Input block analysed. Modified in place.
    fitted_interactions: dict 
        see fast_forward.interaction_fit.InteractionFitter.interactions_dict
        Dictionary of fitted interactions to be assigned into the block

    """
    all_interactions = finalise_interaction_types(block.interactions.keys(), fitted_interactions.keys())

    for inter_type in all_interactions:
        retain = []
        # get interactions which don't have comments
        for interaction in block.interactions[inter_type]:
            if not interaction.meta.get('comment'):
                retain.append(interaction)

        block.interactions[inter_type] = retain + fitted_interactions[inter_type]
