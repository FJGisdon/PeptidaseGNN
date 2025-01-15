import igraph as ig
import pandas as pd
import numpy as np
import pickle

import torch
from torch_geometric.data import Data

from sklearn.preprocessing import StandardScaler, LabelEncoder


def create_gnn_input(config):
	"""
    Prepares the input for the GNN training.

    Args:
        config: A dictionary configuration file containing configuration parameters
    """
	
	# Read the preprocessed data
	with open(config["data"]["graph_information_dictionary_path"], 'rb') as f:
		graph_dictionary = pickle.load(f)
	print(f"Graph dictionary read from {config['data']['graph_information_dictionary_path']}.")
    
	graphs = [ig.Graph.Read_GML(item[0]) for item in graph_dictionary.values()]
    
	with open(config["data"]["graphs_path"], 'rb') as f:
		graphs = pickle.load(f)
	print(f"Graphs read from {config['data']['graphs_path']}.")
	
	# Extract data from graph_dictionary using list comprehensions
	actSiteIndexList: list = [item[1] for item in graph_dictionary.values()]
	actSiteResidueName: list = [item[2] for item in graph_dictionary.values()]
	
	# Generate the target_variables
	amino_acid_mapping = {
						   'SER': 1,
						   'HIS': 2,
						   'ASP': 3,
						   'ASN': 3,
						   'GLU': 3,
						   'GLN': 3,
						   # Add other amino acids and their target values as needed
					   		}
	
	target_variables: dict = {}
	for i, graph in enumerate(graphs):
		target_variables[i] = np.zeros(len(graph.vs), dtype=int)  # Initialize with zeros
		for j, node_id in enumerate(graph.vs['id']):
		    node_id_int = int(node_id)
		    if node_id_int in actSiteIndexList[i]:  # Check if node is an active site
		        residue_index = actSiteIndexList[i].index(node_id_int)
		        residue_name = actSiteResidueName[i][residue_index]
		        target_variables[i][j] = amino_acid_mapping.get(residue_name, 0)
	print("Target variables created.")
		
	# Create the PyTorch Geometric Data object
	pyg_data_list = igraph_to_pytorch_geometric(graphs, target_variables)
	print("PyTorch geometric data object created.")
	
    # Save the data
	with open(config["data"]["pyg_data_path"], 'wb') as f:
		pickle.dump(pyg_data_list, f)
	print(f"PyTorch geometric graph object saved to {config['data']['pyg_data_path']}.")

	return
	
	
def igraph_to_pytorch_geometric(igraph_list, target_variables):
    """Converts a list of igraph graphs to PyTorch Geometric Data objects.
       Optimized for faster execution using vectorization and precomputation.
    """
    pyg_data_list = []

    # Define all 20 standard amino acids
    all_amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK']

    # Fit LabelEncoder on all 20 amino acids
    label_encoder = LabelEncoder()
    label_encoder.fit(all_amino_acids)

    # Define all possible values for chemProp5
    all_chemProp5_values = ['H', 'A', 'P', '+', '-', 'n.d.']

    # Fit LabelEncoder on all chemProp5 values
    label_encoder_chemProp5 = LabelEncoder()
    label_encoder_chemProp5.fit(all_chemProp5_values)

	# Precompute mappings
    residue_mapping = {residue: i for i, residue in enumerate(all_amino_acids)}
    chemProp5_mapping = {value: i for i, value in enumerate(all_chemProp5_values)}

    for i, graph in enumerate(igraph_list):
        if i % 50 == 0:
            print(f"Processing graph {i} of {len(igraph_list)}.")

        # Precompute feature values
        strengths = np.array(graph.strength(weights='weight'))
        eigenvector_centralities = np.array(graph.eigenvector_centrality(directed=False, scale=True, return_eigenvalue=False, weights='weight'))
        betweenness_centralities = np.array(graph.betweenness(vertices=None, directed=False, cutoff=None, weights='weight'))

        # Vectorize feature normalization
        strengths_normalized = (strengths - strengths.min()) / (strengths.max() - strengths.min())
        eigenvector_centralities_normalized = (eigenvector_centralities - eigenvector_centralities.min()) / (eigenvector_centralities.max() - eigenvector_centralities.min())
        betweenness_centralities_normalized = (betweenness_centralities - betweenness_centralities.min()) / (betweenness_centralities.max() - betweenness_centralities.min())

        # Encodings
        residues = [v['residue'] for v in graph.vs]
        chemProp5_values = [v['chemProp5'] for v in graph.vs]
        residues_encoded = np.array([residue_mapping.get(r, residue_mapping['UNK']) for r in residues])
        chemProp5_encoded = np.array([chemProp5_mapping.get(val, chemProp5_mapping['n.d.']) for val in chemProp5_values])

        # Create node features
        node_features = np.zeros((len(graph.vs), 5), dtype=np.float32)
        node_features[:, 0] = residues_encoded
        node_features[:, 1] = strengths_normalized
        node_features[:, 2] = eigenvector_centralities_normalized
        node_features[:, 3] = betweenness_centralities_normalized
        node_features[:, 4] = chemProp5_encoded
        node_features = torch.tensor(node_features, dtype=torch.float32)


        # Get edge weights and distances, and normalize distances
        edge_weights = np.array(graph.es["weight"])
        edge_weights_normalized = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min())
        distances = np.array(graph.es["distance"])
        distances_normalized = (distances - distances.min()) / (distances.max() - distances.min())

        # Combine edge features
        edge_features = np.stack([edge_weights_normalized, distances_normalized], axis=1)
        # access with edge_features[:, 0] for weights and edge_features[:, 1] for distances

        # Create edge index
        edges = graph.get_edgelist()
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()


        # Create target tensor
        y = torch.tensor(target_variables[i], dtype=torch.long)

        # Create PyTorch Geometric Data object
        data = Data(x=node_features, edge_index=edge_index, y=y, edge_attr=torch.tensor(edge_features[:, 1], dtype=torch.float32))
        pyg_data_list.append(data)

    return pyg_data_list
	
	
if __name__ == "__main__":
	import yaml
	with open("config.yaml", 'r') as stream:
		config = yaml.safe_load(stream)
	create_gnn_input(config)
