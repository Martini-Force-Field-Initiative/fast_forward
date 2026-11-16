"""
ITP comparison helpers, adapted from vermouth.tests.integration_tests.test_integration
(Copyright 2018 University of Groningen, Apache License 2.0).

They are copied here because vermouth's test suite is not included in its installed package.
"""
from collections import OrderedDict

import numpy as np
import pytest
import vermouth
from vermouth.forcefield import ForceField


def assert_equal_blocks(block1, block2, blocknames_equal=True):
    """
    Asserts that two blocks are equal to gain the pytest rich comparisons,
    which is lost when doing `assert block1 == block2`
    """
    if blocknames_equal:
        assert block1.name == block2.name
    assert block1.nrexcl == block2.nrexcl
    assert block1.force_field == block2.force_field  # Set to be equal
    nodes2 = OrderedDict(block2.nodes(data=True))
    for n_idx, attrs in nodes2.items():
        for k, v in attrs.items():
            if isinstance(v, np.ndarray):
                nodes2[n_idx][k] = pytest.approx(v, abs=1e-3)
    assert OrderedDict(block1.nodes(data=True)) == nodes2
    edges1 = {frozenset(e[:2]): e[2] for e in block1.edges(data=True)}
    edges2 = {frozenset(e[:2]): e[2] for e in block2.edges(data=True)}
    for e, attrs in edges2.items():
        for k, v in attrs.items():
            if isinstance(v, float):
                attrs[k] = pytest.approx(v, abs=1e-3)
    assert edges1 == edges2
    for inter_type, interactions in block1.interactions.items():
        block2_interactions = block2.interactions.get(inter_type, [])
        assert sorted(interactions, key=lambda i: i.atoms) == sorted(block2_interactions, key=lambda i: i.atoms)


def compare_itp(filename1, filename2):
    """
    Asserts that two itps are functionally identical
    """
    dummy_ff = ForceField(name='dummy')
    with open(filename1) as fn1:
        vermouth.gmx.read_itp(fn1, dummy_ff)
    dummy_ff2 = ForceField(name='dummy')
    with open(filename2) as fn2:
        vermouth.gmx.read_itp(fn2, dummy_ff2)
    for block in dummy_ff2.blocks.values():
        block._force_field = dummy_ff
    assert set(dummy_ff.blocks.keys()) == set(dummy_ff2.blocks.keys())
    for name in dummy_ff.blocks:
        assert_equal_blocks(dummy_ff.blocks[name], dummy_ff2.blocks[name])
