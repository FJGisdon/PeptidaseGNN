# PeptidaseGNN
Learning from peptidase protein structure graphs to detect regions for potential peptidase active sites.

The project is available as a snakemake workflow.

In this project, I want to obtain information from active sites to detect regions in proteins where these active sites might be designed.
Further, I want to see if there are regions in proteins, which might have such potential active sites present but without activity, which
could give interesting evolutionary insights.
So far, the data is collected with scripts, which are not yet implemented here, but the raw data is available.
I am currently developing this project, so the functionality is limited.

````
PeptidaseGNN/  # Project root
├── config.yaml  # Configuration file (hyperparameters, paths, etc.)
├── scripts/  # Python scripts containing workflow logic
│   ├── prepare_data.py
│   ├── create_gnn\_input.py
│   ├── train_gcn.py
│   ├── evaluate_gcn.py
│   └── analyze_results.py
├── Snakefile  # Main Snakemake workflow file
└── data/  # Input and output data
    ├── raw/  # Initial data files (compiled_peptidase_modularization_data.csv, etc.)
    └── processed/  # Intermediate and final results
````


    - scripts/prepare_data.py: Handles loading initial data, filtering for specific structures, and creating the graph_dictionary.
    - scripts/create_gnn_input.py: Converts the information in graph_dictionary into igraph graph objects and stores them.
    - scripts/train_gcn.py: Trains the GCN model using the PyTorch Geometric data and hyperparameters from the config file.
    - scripts/evaluate_gcn.py: Evaluates the trained GCN model, calculating metrics like accuracy, and saves the results.
    - scripts/analyze_results.py: Performs further analysis like identifying false positives and their properties.



During the development of the code, I used using Google Colab. The notebooks will also be provided at some point.
