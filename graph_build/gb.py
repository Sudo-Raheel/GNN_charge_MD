import torch
from torch_geometric.data import Data
from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import IsayevNN
import numpy as np
from pathlib import Path
from utils.elements import atomic_properties
from utils.descriptors import get_dist_features, get_shell_features

class Graph_build:
    def __init__(self, cif_path):
        self.cif_path = Path(cif_path)
        self.data = self.process_cif(self.cif_path)

    def process_cif(self, src_path):
        cif_parser = CifParser(src_path)
        #cif_struct = cif_parser.get_structures(primitive=False).pop()

        cif_struct = cif_parser.parse_structures().pop()
        # Use the custom function to extract charges
        charges = self.extract_charges_from_cif(src_path)


        atom_symbols = [atom.specie.symbol for atom in cif_struct]
        # giving elments numbering 
        # atom_labels = []
        # label_counter = {element: 0 for element in set(atom_symbols)}
        # for symbol in atom_symbols:
        #     label_counter[symbol] += 1
        #     atom_labels.append(f"{symbol}{label_counter[symbol]}")

        # generate the bond table as edges 
        # 0.5 represents the 0.5 Angstorm tolerance refer to 
        bond_table_func = IsayevNN(tol=0.5, allow_pathological=True).get_bonded_structure
        #bond_table_func class contains a lot of stuff 
        # for our case we need the graphs corresponding to the structure 
        #bond_table_func(cif_struct).as_dict()['graphs'] (Structure Graph) this contains all the 
        tmp=bond_table_func(cif_struct).as_dict()['graphs']["adjacency"]
        cif_bond_info = bond_table_func(cif_struct).as_dict()["graphs"]["adjacency"]
        #bond_table_func goes over each atom and adds the interactions of i th atom with the jth('id')
        # it doesnt repeat i.e it only computes the ij part of adjacency ...we will have to create it using ij (its symmetric)
        bonds_ij = []
        for i, bonds in enumerate(cif_bond_info):  # Loop through each atom and its bonds
            for bond in bonds:                     # Loop through each bond of the current atom
                bond_tuple = (i, bond["id"], *bond["to_jimage"]) #(i,j,h,k,l)  
                #imp point 
                # the bond table is computed locally over each atom i 
                # hence the h,k,l indices are the computed for j relative to i not the origin 
                bonds_ij.append(bond_tuple)
        bonds_ij = np.array(bonds_ij)
        # just swapping i with j ... [bonds_ij[:, [1, 0]] 
        # by swapping i with j(bonds_ij[:, [1, 0]]) we are reversing the direction hence the miller indices will be reversed(need to create shell features)
        bonds_ji = np.hstack([bonds_ij[:, [1, 0]], -bonds_ij[:, 2:]]) #(j,i,-h,-k,-l)
        bonds_full = np.vstack([bonds_ij, bonds_ji])
        edges = bonds_full[:, [0, 1]].T
        # edge matrix will be 27*Nx27*N where N are total number of atoms ..27 comes from miller indices [000],[001]...
        # we are saving the indices which are bonded only, to save memory...not the full edge matrix ..since it is very sparse 
        # pytorch geometric data also takes the indices only not the full matrix 

        n_atoms = len(cif_struct)
        f_atom = np.array([atomic_properties[symbol] for symbol in atom_symbols])

        f_dist = get_dist_features(cif_struct, atom_symbols, n_atoms) #structure required to generate
        f_shell = get_shell_features(bonds_full, f_atom, n_atoms) #bond table required to generate
        node_features = np.hstack([f_atom, f_dist, f_shell])
        # # Convert to torch tensors
        # x = torch.tensor(node_features, dtype=torch.float)
        # y = torch.tensor(charges, dtype=torch.float)  # Charges extracted from custom parsing
        # edge_index = torch.tensor(edges, dtype=torch.long)

        return node_features,charges,edges
        #return Data(x=x, y=y, edge_index=edge_index)

    def extract_charges_from_cif(self,src_path):
        charges = []
        with open(src_path, 'r') as file:
            read_charges = False
            for line in file:
                if '_atom_site_charge' in line:
                    read_charges = True
                elif read_charges and line.strip():
                    try:
                        # Assuming charge is the last column in the atom site listing
                        charge = float(line.split()[-1])
                        charges.append(charge)
                    except ValueError:
                        # Handle cases where conversion to float fails
                        read_charges = False
                else:
                    read_charges = False
        return np.array(charges)


    def get_data(self):
        # Method to access the processed graph data
        return self.data
