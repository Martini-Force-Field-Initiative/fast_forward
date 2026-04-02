
from . import DATA_PATH
from vermouth.gmx.itp import write_molecule_itp
from vermouth.file_writer import deferred_open
from vermouth.data import COMMON_CITATIONS
from vermouth.citation_parser import citation_formatter, read_bib
from collections import ChainMap

def itp_writer(molname, block, command_used):
    '''
    Write an itp for a block with new interactions. All existing interactions will be removed and new ones
    written in their place

    Parameters
    ----------
    molname: str
        molname for block
    block: :class:`~vermouth.molecule.Block`
        The block for which to write the itp
    command_used: str
        Command used to run the program

    '''

    header = ['This file was generated using the following command:',
              command_used, '\n',
              'Please cite the following papers:'
              ]

    with open(DATA_PATH/'citations.bib') as citation_file:
        ff_citations = read_bib(citation_file)

    # make the block a molecule for writing
    mol_out = block.to_molecule()
    mol_out.citations = {'vermouth', 'fast_forward'}
    citation_map = ChainMap(ff_citations, COMMON_CITATIONS)

    for citation in mol_out.citations:
        cite_string = citation_formatter(
            citation_map[citation]
        )
        header.append(cite_string)

    mol_out.meta['molname'] = molname

    with deferred_open(f'{molname}.itp', 'w') as fout:
        write_molecule_itp(mol_out, fout, moltype=molname, header=header)
