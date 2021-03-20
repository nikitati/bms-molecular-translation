import click
import pandas as pd
from tqdm import tqdm
import selfies

from src.utils import *


@click.command()
@click.argument('targets-path', type=click.Path())
@click.argument('result-path', type=click.Path())
def main(targets_path, result_path):
    df = pd.read_csv(targets_path)
    print("Converting target to additional formats")
    tqdm.pandas(desc="Progress")
    df['smiles'] = df['InChI'].progress_map(inchi2smiles)
    df['deepsmiles'] = pd.Series(smiles2deepsmiles(tqdm(df['smiles'])))
    df['selfies'] = pd.Series(smiles2selfies(tqdm(df['smiles'])))
    print("Computing resulting lengths")
    df['InChI_len'] = df['InChI'].map(len)
    df['smiles_len'] = df['smiles'].map(len)
    df['deepsmiles_len'] = df['deepsmiles'].map(len)
    df['selfies_str_len'] = df['selfies'].map(len)
    df['selfies_token_len'] = df['selfies'].map(selfies.len_selfies)
    print("Writing new targets file")
    df.to_csv(result_path)


if __name__ == "__main__":
    main()