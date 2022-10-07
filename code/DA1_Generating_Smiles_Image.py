"""Code for saving molecule images"""

from rdkit import Chem
from rdkit.Chem import Draw
import os
import os.path as osp
import pandas as pd


# File path
IN_DIR = '../data/input/raw/'
OUT_DIR = '../data/output/DA1/'

# Prepare data
df = pd.read_csv(osp.join(IN_DIR + 'kinase_data.csv'))

# Group data
grouped_type_and_kinase = df.groupby(['KINASE'])
smiles = df['SMILES'].unique()

print("The number of smiles", len(smiles))

# Save as molecule images
for i, g in enumerate(grouped_type_and_kinase):
    dir = g[0]
    os.makedirs(osp.join(OUT_DIR + dir), exist_ok=True)

    for index, row in g[1].iterrows():
        smiles = row['SMILES']
        value = row['PKI']
        molecule = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(molecule, osp.join(OUT_DIR + dir + f'/m_{value}-{index}.png'))
