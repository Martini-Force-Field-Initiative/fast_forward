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
import numpy as np
from collections import OrderedDict, defaultdict
from vermouth.parser_utils import SectionLineParser

class Mapping():

    def __init__(self, from_resname, to_resname):
        self.from_resname = from_resname
        self.to_resname = to_resname
        self.atom_weights = defaultdict(list) 
        self.bead_to_idx = defaultdict(list) 
        self.bead_to_atom = defaultdict(list)

    def add_atom(self, bead, idx, atom=None, weight=None):
        self.bead_to_atom[bead].append(atom)
        self.bead_to_idx[bead].append(idx)
        self.atom_weights[bead].append(weight)

    @property
    def beads(self):
        return list(self.bead_to_idx.keys())

    @property
    def n_beads(self):
        return len(self.bead_to_idx)

class MapDirector(SectionLineParser):

    COMMENT_CHAR = ';'

    def __init__(self):
        super().__init__()
        self.current_mapping = None
        self.current_from = None
        self.mappings = {}
        self.header_actions = {
            ('molecule',): self._new_mapping
        }

    def parse_header(self, line, lineno=0):
        """
        Parses a section header with line number `lineno`. Sets
        :attr:`vermouth.parser_utils.SectionLineParser.section`
        when applicable. Does not check whether `line` is a valid section
        header.

        Parameters
        ----------
        line: str
        lineno: str

        Returns
        -------
        object
            The result of calling :meth:`~vermouth.parser_utils.SectionLineParser.finalize_section`,
            which is called if a section ends.

        Raises
        ------
        KeyError
            If the section header is unknown.
        """
        result = super().parse_header(line, lineno)
        action = self.header_actions.get(tuple(self.section))
        if action:
            action()

        return result

    def _new_mapping(self):
        if self.current_mapping:
            self.mappings[self.current_from] = self.current_mapping
            self.current_from = None

        self.current_mapping = None

    @SectionLineParser.section_parser('molecule')
    def _molecules(self, line, lineno=0):
        """
        Parses the lines in the '[molecule]'
        directive and stores it. This should actually
        be residue but to comply with the bachwards
        format we keep molecule.
        """
        tokens = line.split()
        if len(tokens) == 2:
            from_resname, to_resname = tokens
        else:
            from_resname = to_resname = tokens[0]
        self.current_mapping = Mapping(from_resname=from_resname,
                                       to_resname=to_resname)
        self.current_from = from_resname

    @SectionLineParser.section_parser('martini')
    def _bead_order(self, line, lineno=0):
        """
        defines the order of beads in the final mapped trajectory
        """
        tokens = line.split()
        for bead in tokens:
            self.current_mapping.bead_to_idx[bead] = []
            self.current_mapping.bead_to_atom[bead] = []
            self.current_mapping.atom_weights[bead] = []

    @SectionLineParser.section_parser('to')
    @SectionLineParser.section_parser('from')
    @SectionLineParser.section_parser('mapping')
    @SectionLineParser.section_parser('chiral')
    @SectionLineParser.section_parser('trans')
    @SectionLineParser.section_parser('out')
    def _skip_line(self, line, lineno=0):
        pass

    @SectionLineParser.section_parser('atoms')
    def _atoms(self, line, lineno=0):
        """
        Parse and store atomtypes section
        """
        tokens = line.split()
        idx = tokens[0]
        atom = tokens[1]
        beads = tokens[2:]
        for bead in range(len(beads)):
            if beads[bead][0] != "!" and beads[bead][0].isalpha():
                if not beads[bead-1][0].isalpha():
                    self.current_mapping.add_atom(idx=idx,
                                                  atom=atom,
                                                  bead=beads[bead],
                                                  weight=np.float32(beads[bead-1]))
                else:
                    self.current_mapping.add_atom(idx=idx,
                                                  atom=atom,
                                                  bead=beads[bead],
                                                  weight=np.float32(1.0))
    def finalize(self, lineno=0):
        """
        Called at the end of the file
        """
        self._new_mapping()
        return self.mappings

def read_mapping(lines):
    """
    Parses `lines` of itp format and adds the
    molecule as a block to `force_field`.

    Parameters
    ----------
    lines: list
        list of lines of an itp file
    force_field: :class:`vermouth.forcefield.ForceField`
    """
    director = MapDirector()
    return list(director.parse(iter(lines)))
