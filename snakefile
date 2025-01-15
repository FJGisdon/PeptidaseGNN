import yaml

# Load configuration
with open("config.yaml", 'r') as stream:
	config = yaml.safe_load(stream)
   
# Define rules 
rule all:
    input:
        config["data"]["graph_information_dictionary_path"],
        config["data"]["graphs_path"],
        config["data"]["pyg_data_path"],
        config["data"]["data_masks"],
        config["data"]["gcn_model_trained"]
        # ... any other final outputs ...


rule prepare_data:
    input:
        config["data"]["raw_data_csv_path"]
    output:
        [config["data"]["graph_information_dictionary_path"], config["data"]["graphs_path"]]
    conda:
        "envs/env_gnn.yaml"  # Path to the environment file
    script:
        "scripts/prepare_data.py"


rule create_gnn_input:
    input:
        [config["data"]["graph_information_dictionary_path"], config["data"]["graphs_path"]]
    output:
        config["data"]["pyg_data_path"]
    conda:
        "envs/env_gnn.yaml"  # Path to the environment file
    script:
        "scripts/create_gnn_input.py"


rule train_gcn:
    input:
        config["data"]["pyg_data_path"]
    output:
        [config["data"]["data_masks"],config["data"]["gcn_model_trained"]]
    conda:
        "envs/env_gnn.yaml"  # Path to the environment file
    script:
        "scripts/train_gcn.py"
        
        
        
