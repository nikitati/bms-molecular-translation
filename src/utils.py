from typing import List

from rdkit import Chem
from rdkit.Chem import inchi
import selfies
import deepsmiles


def inchi2smiles(inchi_str: str):
    mol = inchi.MolFromInchi(inchi_str)
    return Chem.MolToSmiles(mol)


def smiles2inchi(smiles_str: str):
    mol = Chem.MolFromSmiles(smiles_str)
    if mol is not None:
        inchi_str = inchi.MolToInchi(mol)
    else:
        inchi_str = "InChI=1S/"
    return inchi_str


def smiles2deepsmiles(smiles: List[str], rings=True, branches=True):
    converter = deepsmiles.Converter(rings=rings, branches=branches)
    return [converter.encode(smile) for smile in smiles]


def deepsmiles2smiles(dsm: List[str], rings=True, branches=True):
    converter = deepsmiles.Converter(rings=rings, branches=branches)
    return [converter.decode(ds) for ds in dsm]


def smiles2selfies(smiles: List[str]):
    return [selfies.encoder(smile) for smile in smiles]


def selfies2smiles(selfies_list: List[str]):
    return [selfies.decoder(sf) for sf in selfies_list]


def selfies2inchi(selfies_str: str) -> str:
    smiles = selfies.decoder(selfies_str)
    inchi_str = smiles2inchi(smiles)
    return inchi_str
