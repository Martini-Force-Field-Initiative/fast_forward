# Copyright 2020 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from vermouth.gmx.itp_read import ITPDirector
from vermouth.parser_utils import split_comments
from vermouth.molecule import Interaction
import numpy as np

class FastForwardITPParser(ITPDirector):
    '''
    Same as the ITP parser but allows stashing of comments with interactions.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_comment = None

    def parse(self, file_handle):
        """
        Reads lines from `file_handle`, and calls :meth:`dispatch` to find
        which method to call to do the actual parsing. Yields the result of
        that call, if it's not `None`.
        At the end, calls :meth:`finalize`, and yields its results, iff
        it's not None.
        Parameters
        ----------
        file_handle: collections.abc.Iterable[str]
            The data to parse. Should produce lines of data.
        Yields
        ------
        object
            The results of dispatching to parsing methods, and of
            :meth:`finalize`.
        """
        lineno = 0
        for lineno, line in enumerate(file_handle, 1):
            line, comment = split_comments(line, self.COMMENT_CHAR)
            if self.COMMENT_CHAR in comment:
                atom_comment, _ = split_comments(comment, self.COMMENT_CHAR)
                self.current_comment = atom_comment
            else:
                self.current_comment = comment
            if not line:
                continue
            result = self.dispatch(line)(line, lineno)

            if result is not None:
                yield result

        result = self.finalize(lineno)
        if result is not None:
            yield result

    def _base_parser(self, tokens, context, section, atom_idxs):
        """
        Converts an interaction line into a vermouth interaction
        tuple. It updates the block interactions in place.
        Parameters
        ----------
        tokens: collections.deque[str]
            Deque of token to inspect. The deque **can be modified** in place.
        context: :class:`vermouth.molecule.Block`
            The current block we parse
        section: str
            The current section header
        atom_idxs: list of ints or strings that are valid python slices
        """
        # split atoms and parameters

        atoms, parameters = self._split_atoms_and_parameters(tokens, atom_idxs)

        # perform check on the atom ids
        treated_atoms = self._treat_block_interaction_atoms(atoms, context, section)

        if self.current_meta:
            meta = {self.current_meta['condition']: self.current_meta['tag']}
        else:
            meta = {}

        if self.current_comment:
            meta['comment'] = self.current_comment

        interaction = Interaction(
            atoms=treated_atoms,
            parameters=parameters,
            meta=meta,)

        context.interactions[section] = context.interactions.get(section, []) + [interaction]

def read_itp(lines, force_field):
    director = FastForwardITPParser(force_field)
    return list(director.parse(iter(lines)))

def _matching_angles(angles):
    '''

    for a list of tuples that may contain integers in reversed order, return only one entry.

    e.g. for [(0,1,2),(1,2,3),(2,1,0)] we get [(0,1,2),(1,2,3)]

    '''

    seen = set()
    unique_list = []

    for tup in angles:
        sorted_tup = tuple(sorted(tup))  # Sort to ensure order-independent uniqueness
        if sorted_tup not in seen:
            seen.add(sorted_tup)
            unique_list.append(tup)  # Append the original order

    return unique_list

def guess_interactions(block):
    """
    From the bonds described in a block's interactions, generate all possible angles and dihedrals.
    -----
    block: :class:`vermouth.molecule.Block`
    """

    # clear any existing angles and dihedrals to prevent duplicates
    if block.interactions.get('angles'):
        del block.interactions['angles']
    if block.interactions.get('dihedrals'):
        del block.interactions['dihedrals']

    block.make_edges_from_interactions()

    angles = _matching_angles(block.guess_angles())
    dihedrals = _matching_angles(block.guess_dihedrals())

    # add dummy interactions to block if they're not already there.
    for i in angles:
        if i not in [[int(j) for j in k.atoms] for k in block.interactions['angles']]:
            comment = '_'.join([f"{block.nodes[atom]['resid']}_{block.nodes[atom]['atomname']}" for atom in i])
            block.add_interaction('angles', atoms=i,
                                  parameters=['2', '10', '10'], meta={'version': 0, 'comment': comment})
    for i in dihedrals:
        if i not in [[int(j) for j in k.atoms] for k in block.interactions['dihedrals']]:
            comment = '_'.join([f"{block.nodes[atom]['resid']}_{block.nodes[atom]['atomname']}" for atom in i])
            block.add_interaction('dihedrals', atoms=i,
                                  parameters=['1', '10', '1', '1'], meta={'version': 0, 'comment': comment})
