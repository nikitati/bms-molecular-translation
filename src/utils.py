from rdkit import Chem
from rdkit.Chem import inchi
from rdkit.Chem import AllChem
import selfies
import deepsmiles


def inchi2smiles(inchi_str: str):
    mol = inchi.MolFromInchi(inchi_str)
    return Chem.MolToSmiles(mol)


def smiles2inchi(smiles_str: str):
    mol = Chem.MolFromSmiles(smiles_str)
    return Chem.MolToInchi(mol)

def smiles2deepsmiles(smiles: list[str], rings=True, branches=True):
    converter = deepsmiles.Converter(rings=rings, branches=branches)
    return [converter.encode(smile) for smile in smiles]


def deepsmiles2smiles(dsm: list[str], rings=True, branches=True):
    converter = deepsmiles.Converter(rings=rings, branches=branches)
    return [converter.decode(ds) for ds in dsm]


def smiles2selfies(smiles: list[str]):
    return [selfies.encoder(smile) for smile in smiles]


def selfies2smiles(selfies: list[str]):
    return [selfies.decoder(sf) for sf in selfies]
