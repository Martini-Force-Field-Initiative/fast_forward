
from numpy import isclose

MARTINI_MASS = {"S": 54, "T": 36}

def mass_redistribution(block, pair_idx):
    """
    Redistribute the mass of a virtual site across
    Parameters
    ----------
    block
    pair_idx

    Returns
    -------

    """
    for pairs in pair_idx:
        vs = pairs[0]
        block.nodes[vs]['mass'] = 0.0
        block.nodes[vs]['charge'] = block.nodes[vs].get('charge', '0.0')
        vs_type_mass = MARTINI_MASS.get(block.nodes[vs]["atype"][0], 72)
        redistribution = vs_type_mass / len(pairs[1:])

        # if only 1 constructor (i.e. building vs directly on top of atom), no redistribution necessary.
        reassign = []
        if len(pairs[1:]) > 1:
            constructors = pairs[1:]
            for node in constructors:
                current_mass = block.nodes[node].get("mass", MARTINI_MASS.get(block.nodes[node]["atype"][0], 72))
                expected_new_mass = MARTINI_MASS.get(block.nodes[node]["atype"][0], 72) + redistribution

                if not isclose(current_mass, expected_new_mass):
                    block.nodes[node]["mass"] = expected_new_mass
                    block.nodes[node]["charge"] = block.nodes[node].get("charge", "0.0")
                else:
                    reassign.append(True)

        if any(reassign):
            print(f"Masses for constructor beads of virtual site at index {vs+1} "
                  f"did not correspond to standard Martini masses. Will not attempt "
                  f"automatic mass reassignment.")
