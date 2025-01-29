import sys
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import torch
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from gnn_model_classes import GCN_h1, GCN_h2



def evaluate_gcn(config):
	"""
    Evaluates the trained model.

    Args:
        config: A dictionary configuration file containing configuration parameters
    """
    
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define the hyperparameters.
	in_channels = 6  # Number of input features per node
	out_channels = 4 # Number of output classes (adjust to your task)
	hidden_layers = config["hyperparameters"]["hidden_layers"]
	hidden_channels_1 = config["hyperparameters"]["hidden_channels_1"] # Number of hidden channels in the first hidden layer
	hidden_channels_2 = config["hyperparameters"]["hidden_channels_2"] # Number of hidden channels in the second hidden layer
	learning_rate = config["hyperparameters"]["learning_rate"] # Learning rate for the optimizer
	dropout_rate_1 = config["hyperparameters"]["dropout_rate_1"]
	dropout_rate_2 = config["hyperparameters"]["dropout_rate_2"]    
	batch_size = config["hyperparameters"]["batch_size"]

	# Read the preprocessed data
	with open(config["data"]["pyg_data_path"], 'rb') as f:
		pyg_data_list = pickle.load(f)
	print("PyTorch geometric graph object read.")
	
    # Read the data masks
	with open(config["data"]["data_masks"], 'rb') as f:
		masks = pickle.load(f)
	print("Masks loaded.")
	
	# Load the model
	model_path = config["data"]["gcn_model_trained"]
	
	# Create an instance of your GCN model (same architecture as when saving)
	if hidden_layers == 1:
		modelGCN = GCN_h1(in_channels, hidden_channels_1, out_channels, dropout_rate_1).to(device)
	elif hidden_layers == 2:
		modelGCN = GCN_h2(in_channels, hidden_channels_1, hidden_channels_2, out_channels, dropout_rate_1, dropout_rate_2).to(device) # Create the GNN model
	else: sys.exit("Only 1 or 2 hidden layers are supported.")

	# Load the saved state dictionary into the model
	modelGCN.load_state_dict(torch.load(model_path))

	# Get the test data
	test_loader = DataLoader([pyg_data_list[i] for i in range(len(pyg_data_list)) if masks['test_mask'][i]], batch_size=batch_size, shuffle=False)

	# Call the test function and store the returned values
	all_labels, all_preds = test_with_report(modelGCN, test_loader, device)

	# Now you can use all_labels and all_preds outside the function
	print(classification_report(all_labels, all_preds))

	# Calculate and print accuracy
	accuracy = accuracy_score(all_labels, all_preds)
	print(f"Accuracy: {accuracy:.4f}")

	# Confusion Matrix
	conf_mat = confusion_matrix(all_labels, all_preds)
	fig, ax = plt.subplots(figsize=(8,6))
	sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
	plt.xlabel('Predicted Labels')
	plt.ylabel('True Labels')
	plt.title('Confusion Matrix')
	plt.savefig(config["data"]["confusion_matrix"])
	plt.close()



def test_with_report(model, loader, device, threshold_class1=0.8, threshold_class2=0.8, threshold_class3=0.8):

	# Threshold adjustment was used to modify the decision boundary for classifying
	# predictions into different classes. By default, the model assigns a prediction
	# to the class with the highest probability. However, the threshold to make the model 
	# more selective in its predictions, potentially reducing false positives can be adapted.

	model.eval()
	all_preds = []
	all_labels = []
	for data in loader:
	    data = data.to(device)
	    out = model(data.x, data.edge_index, data.edge_attr) # Model outputs (logits)
	    probabilities = torch.exp(out)  # Convert logits to probabilities using softmax
	    pred = out.argmax(dim=1)
	    # Apply thresholds to adjust predictions
	    predicted_classes = []
	    for probs in probabilities:
	        predicted_class = 0 # Initialize the predicted class as 0 before thresholds
	        if probs[1] >= threshold_class1:
	            predicted_class = 1
	        elif probs[2] >= threshold_class2:
	            predicted_class = 2
	        elif probs[3] >= threshold_class3:
	            predicted_class = 3
	        predicted_classes.append(predicted_class)

	    all_preds.extend(predicted_classes)  # Update with adjusted predictions
	    all_labels.extend(data.y.cpu().numpy()) # append the true labels for the current batch

	print(classification_report(all_labels, all_preds))
	# Return the values to be used outside the function
	return all_labels, all_preds




if __name__ == "__main__":
	import yaml
	with open("config.yaml", 'r') as stream:
		config = yaml.safe_load(stream)
	evaluate_gcn(config)
