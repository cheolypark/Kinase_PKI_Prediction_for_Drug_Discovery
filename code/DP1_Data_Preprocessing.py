"""Code for preprocessing data"""

import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Dataset, Data
import numpy as np
import os


#===================================================================================================================
# Code reference:
#  I referred to the code in the link below.
#  And I edited this code to convert the original Kinase data (.csv) to the data structure of torch_geometric (.gp)
#  https://github.com/deepfindr/gnn-project/blob/main/dataset.py
#===================================================================================================================

class KinaseDataset(Dataset):
    def __init__(self, root='../data/input', raw_file_name='kinase_data.csv', transform=None, pre_transform=None, pre_filter=None):
        self.raw_file_name = raw_file_name
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return self.raw_file_name

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0], index_col=0)
        self.data = self.data.sample(frac=1)
        self.data = self.data.reset_index()
        return [f'graph_{i}.gp' for i in list(self.data.index)]

    def get_measurement_type_id(self, measurement_type):
        return 1 if measurement_type == 'pIC50' else 0

    def get_kinase_type_id(self, kinase):
        if kinase == 'JAK1':
            return 0
        elif kinase == 'JAK2':
            return 1
        elif kinase == 'JAK3':
            return 2
        elif kinase == 'TYK2':
            return 3

    def process(self):
        for index, row in self.data.iterrows():
            smiles = row['SMILES']
            measurement_value = row['PKI']
            kinase = row['KINASE']

            mol = Chem.MolFromSmiles(smiles)

            node_feats = self.get_node_features(mol, kinase)
            edge_feats = self.get_edge_features(mol)
            edge_index = self.get_adjacency_info(mol)

            y = self.get_y(measurement_value)

            data = Data(x=node_feats, edge_index=edge_index, edge_attr=edge_feats, y=y, smiles=smiles)

            torch.save(data, os.path.join(self.processed_dir, f'graph_{index}.gp'))

    def get_node_features(self, mol, kinase):
        all_node_feats = []

        for atom in mol.GetAtoms():
            node_feats = []
            # We can include various node features
            # Reference:
            #   https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html

            node_feats.append(atom.GetAtomicNum())
            node_feats.append(atom.GetDegree())
            node_feats.append(atom.GetTotalValence())
            node_feats.append(atom.GetTotalDegree())
            node_feats.append(atom.GetExplicitValence())
            node_feats.append(atom.GetImplicitValence())
            node_feats.append(atom.GetMass())
            node_feats.append(atom.GetNumImplicitHs())
            node_feats.append(atom.GetFormalCharge())
            node_feats.append(atom.GetHybridization())
            node_feats.append(atom.GetIsAromatic())
            node_feats.append(atom.GetTotalNumHs())
            node_feats.append(atom.GetNumRadicalElectrons())
            node_feats.append(atom.IsInRing())
            node_feats.append(atom.GetChiralTag())
            node_feats.append(self.get_kinase_type_id(kinase))

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def get_edge_features(self, mol):
        all_edge_feats = []

        for bond in mol.GetBonds():
            edge_feats = []
            edge_feats.append(bond.GetIsAromatic())
            edge_feats.append(bond.GetIsConjugated())
            edge_feats.append(bond.GetBondTypeAsDouble())
            edge_feats.append(bond.IsInRing())

            all_edge_feats += [edge_feats, edge_feats]

        all_edge_feats = np.asarray(all_edge_feats)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    def get_adjacency_info(self, mol):
        edge_indices = []

        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def get_y(self, measurement_value):
        measurement_value = np.asarray([measurement_value])
        return torch.tensor(measurement_value, dtype=torch.float)

    def len(self):
        return self.data.shape[0]

    def get(self, index):
        return torch.load(os.path.join(self.processed_dir, f'graph_{index}.gp'))


if __name__ == '__main__':
    root_dir = '../data/input'
    raw_file_name = 'kinase_data.csv'

    KinaseDataset(root_dir, raw_file_name)
