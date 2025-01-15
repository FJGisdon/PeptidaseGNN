import os
import pandas as pd
import igraph as ig
import pickle

def prepare_data(config):
	"""
    Prepares the graph data for analysis.

    Args:
        config: A dictionary configuration file containing configuration parameters
    """
	
	# Read the raw data
	data_peptidases = pd.read_csv(config["data"]["raw_data_csv_path"])
	print("Data read.")
    
	# Create peptidase/graph ditionary
	graph_dictionary = create_peptidase_dictionary(data_peptidases)
	print("Graph dictionary created.")
    
	with open(config["data"]["graph_information_dictionary_path"], 'wb') as f:
		pickle.dump(graph_dictionary, f)
	print(f"Graph dictionary saved to {config['data']['graph_information_dictionary_path']}.")
    
	# Construct the graphs and save them
	graphs = [ig.Graph.Read_GML(item[0]) for item in graph_dictionary.values()]
    
	with open(config["data"]["graphs_path"], 'wb') as f:
		pickle.dump(graphs, f)
	print(f"Graphs saved to {config['data']['graph_information_dictionary_path']}.")

	return
    
    
def create_peptidase_dictionary(data_peptidases):
	"""
	Creates a dictionary of peptidases with no mutations and all three active site residues.

	Args:
		data_peptidases: Pandas DataFrame containing peptidase data.

	Returns:
		A dictionary where keys are PDB IDs and values are lists containing:
			- Path to the graph file
			- List of active site residue IDs
			- List of active site residue names
	"""
	graph_dictionary = {}
	for structure in data_peptidases['pdb_id'].unique():
		mask = data_peptidases['pdb_id'] == structure
		active_site_count = data_peptidases.loc[mask]['active_site'].sum()
		act_site_mut_count = data_peptidases.loc[mask]['act_site_mut'].sum()

		if active_site_count == 3 and act_site_mut_count == 0:  
			# Optimized using boolean indexing and .tolist()
			activeSiteResiduesIndices = data_peptidases.loc[mask & data_peptidases['active_site']].index.tolist()
            
			# Extracting IDs and names using list comprehensions
			activeSiteResidueIDs = [int(data_peptidases.loc[index]['node_id'].split('-')[0]) for index in activeSiteResiduesIndices]
			activeSiteResidues = data_peptidases.loc[activeSiteResiduesIndices, 'residue'].tolist()
            
			#Simplified condition using set intersection
			required_residues = {'SER', 'HIS'}
			optional_residues = {'ASN', 'ASP', 'GLU', 'GLN'}
			if required_residues.issubset(activeSiteResidues) and any(res in activeSiteResidues for res in optional_residues):
				subfamily = data_peptidases.loc[mask, "subfamily"].iloc[0]
				peptidase = data_peptidases.loc[mask, "peptidase"].iloc[0]
				gml_file = data_peptidases.loc[mask, "gml_file"].iloc[0]

				graph_path = os.path.join(config["data"]["raw_data_path"],f"{subfamily}/{peptidase}/{structure}/modularization/ptgl_output/{gml_file.split('_')[0]}_aagraph-network_properties.gml")
				graph_dictionary[structure] = [graph_path, activeSiteResidueIDs, activeSiteResidues]

	return graph_dictionary



if __name__ == "__main__":
	import yaml
	with open("config.yaml", 'r') as stream:
		config = yaml.safe_load(stream)
	prepare_data(config)
