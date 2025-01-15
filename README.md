# PeptidaseGNN
Learning from peptidase protein structure graphs to detect regions for potential peptidase active sites.

````
PeptidaseGNN\  # Project root
├── config.yaml  # Configuration file (hyperparameters, paths, etc.)
├── scripts/  # Python scripts containing workflow logic
│   ├── prepare\_data.py
│   ├── create\_gnn\_input.py
│   ├── train\_gcn.py
│   ├── evaluate\_gcn.py
│   └── analyze\_results.py
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



This code was developed using Google Colab.
