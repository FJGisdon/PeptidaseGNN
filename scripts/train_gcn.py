import sys
import pickle
import random

import torch
from torch.optim import lr_scheduler
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv



def train_gcn(config):
	"""
    Splits the data, creates and trains the GCN model.

    Args:
        config: A dictionary configuration file containing configuration parameters
    """
    
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	    
    # Define the hyperparameters.
	in_channels = 5  # Number of input features per node
	out_channels = 4 # Number of output classes (adjust to your task)
	hidden_layers = config["hyperparameters"]["hidden_layers"]
	hidden_channels_1 = config["hyperparameters"]["hidden_channels_1"] # Number of hidden channels in the first hidden layer
	hidden_channels_2 = config["hyperparameters"]["hidden_channels_2"] # Number of hidden channels in the second hidden layer
	learning_rate = config["hyperparameters"]["learning_rate"] # Learning rate for the optimizer
	dropout_rate_1 = config["hyperparameters"]["dropout_rate_1"]
	dropout_rate_2 = config["hyperparameters"]["dropout_rate_2"]
	epochs = config["hyperparameters"]["epochs"] # Number of epochs for training
	batch_size = config["hyperparameters"]["batch_size"]
    
	# Read the preprocessed data
	with open(config["data"]["pyg_data_path"], 'rb') as f:
		pyg_data_list = pickle.load(f)
	print("PyTorch geometric graph object read.")
	
    # Split the data
	masks = split_training_data(config, pyg_data_list)
	with open(config["data"]["data_masks"], 'wb') as f:
		pickle.dump(masks, f)
	print("Masks saved.")
	
	# Create data loaders
	train_loader = DataLoader([pyg_data_list[i] for i in range(len(pyg_data_list)) if masks['train_mask'][i]], batch_size=batch_size, shuffle=True)
	val_loader = DataLoader([pyg_data_list[i] for i in range(len(pyg_data_list)) if masks['val_mask'][i]], batch_size=batch_size, shuffle=False)
	test_loader = DataLoader([pyg_data_list[i] for i in range(len(pyg_data_list)) if masks['test_mask'][i]], batch_size=batch_size, shuffle=False)
	print(pyg_data_list[1])
	
	# Normalize class weights to account for imbalanced classes
	class_weights = normalize_class_weights(out_channels, train_loader)
	
	# Create your weighted loss function using the calculated class_weights
	criterion = torch.nn.NLLLoss(weight=class_weights.to(device)) # Make sure weights are on the same device as your model
	
	# Initialize the model, optimizer, and loss function
	if hidden_layers == 1:
		modelGCN = GCN_h1(in_channels, hidden_channels_1, out_channels, dropout_rate_1).to(device) # Create the GNN model
	elif hidden_layers == 2:
		modelGCN = GCN_h2(in_channels, hidden_channels_1, hidden_channels_2, out_channels, dropout_rate_1, dropout_rate_2).to(device) # Create the GNN model
	else: sys.exit("Only 1 or 2 hidden layers are supported.")
	optimizer = torch.optim.Adam(modelGCN.parameters(), lr=learning_rate)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

	
	# Hyperparameter optimization was performed external
	
	# Early stopping
	best_val_loss = float('inf')  # Initialize with a very high value
	patience = 10  # Number of epochs to wait for improvement
	epochs_without_improvement = 0

	# Prepare a dictionary of arguments
	args_dict = {
		'device': device,
		'optimizer': optimizer,
		'modelGCN': modelGCN,
		'criterion': criterion,
	}

	# Training loop with validation
	train_losses = []
	for epoch in range(epochs):
		train(train_loader, train_losses, **args_dict)
		val_loss, val_acc = validate(val_loader, args_dict['device'], args_dict['modelGCN'], args_dict['criterion'])  # Evaluate on validation set
		# Early stopping
		if val_loss < best_val_loss:
		    best_val_loss = val_loss
		    epochs_without_improvement = 0
		    # Save the model
		    # Save the model's state dictionary
		    torch.save(modelGCN.state_dict(), config["data"]["gcn_model_trained"])
		scheduler.step() # Update the learning rate (if using a scheduler that updates every epoch)
		train_acc = test(train_loader, args_dict['device'], args_dict['modelGCN'])
		test_acc = test(test_loader, args_dict['device'], args_dict['modelGCN'])
		print(f'Epoch: {epoch+1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
	
	return
	
	
	

# Define the GCN model classes
class GCN_h1(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, out_channels, dropout_rate_1):
        super().__init__()
        self.dropout_rate_1 = dropout_rate_1
        self.conv1 = GCNConv(in_channels, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate_1, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1)

class GCN_h2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_1, hidden_channels_2, out_channels, dropout_rate_1, dropout_rate_2):
        super().__init__()
        self.dropout_rate_1 = dropout_rate_1
        self.dropout_rate_2 = dropout_rate_2
        self.conv1 = GCNConv(in_channels, hidden_channels_1)
        self.conv2 = GCNConv(hidden_channels_1, hidden_channels_2)
        self.conv3 = GCNConv(hidden_channels_2, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate_1, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate_2, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        return F.log_softmax(x, dim=1) 
    
    
def split_training_data(config, pyg_data_list):
	"""
	Splits the data into training, validation and test set.

	Args:
		pyg_data_list:

	Returns:
		A dictionary with the different masks for the data.
	"""

	# Calculate the number of samples for each set
	num_samples = len(pyg_data_list)
	num_train = int(config["hyperparameters"]["train_ratio"] * num_samples)
	num_val = int(config["hyperparameters"]["val_ratio"] * num_samples)
	num_test = num_samples - num_train - num_val

	# Shuffle the data randomly
	random.seed(42)  # Set a seed for reproducibility
	random.shuffle(pyg_data_list)

	# Create masks
	train_mask = torch.zeros(num_samples, dtype=torch.bool)
	val_mask = torch.zeros(num_samples, dtype=torch.bool)
	test_mask = torch.zeros(num_samples, dtype=torch.bool)

	# Assign True values to the masks based on the split ratios
	train_mask[:num_train] = True
	val_mask[num_train:num_train + num_val] = True
	test_mask[num_train + num_val:] = True

	return {'train_mask': train_mask, 'val_mask': val_mask, 'test_mask': test_mask}
	
	
def normalize_class_weights(out_channels, loader):
	""" """
	# Use only the training set for normalization
	train_loader = loader
	
	# Initialize class counts with zeros
	class_counts = torch.zeros(out_channels, dtype=torch.int64) # out_channels is the number of classes

	# Iterate through the entire training dataset
	for data in train_loader:
		# Get the class labels for the current batch
		labels = data.y

		# Update the class counts using bincount
		counts = torch.bincount(labels)
		class_counts[:len(counts)] += counts  # Account for cases where some classes might not be in a batch

	# Calculate class weights (inverse of class frequencies)
	class_weights = 1.0 / class_counts.float()

	# Normalize class weights
	class_weights = class_weights / class_weights.sum()

	# Print the calculated class weights
	print("Class Weights:", class_weights)
	
	return class_weights
	
	
# Training loop validation and testing
def train(loader, train_losses, device, optimizer, modelGCN, criterion):
    modelGCN.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad() # Clear gradients
        out = modelGCN(data.x, data.edge_index, data.edge_attr)  # Forward pass
        loss = criterion(out, data.y)  # Compute loss
        loss.backward()  # Backpropagation
        train_losses.append(loss.item())
        optimizer.step() # Update model parameters

def validate(loader, device, modelGCN, criterion):
    modelGCN.eval()  # Set the model to evaluation mode
    total_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient calculation during validation
        for data in loader:  # Iterate over validation data
            data = data.to(device)
            out = modelGCN(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            total_loss += loss.item() * data.num_graphs  # Accumulate loss
            pred = out.argmax(dim=1)  # Get predicted labels
            correct += int((pred == data.y).sum())  # Count correct predictions
    return total_loss / len(loader.dataset), correct / len(loader.dataset)  # Return average loss and accuracy

def test(loader, device, modelGCN):
    modelGCN.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = modelGCN(data.x, data.edge_index, data.edge_attr)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)



if __name__ == "__main__":
	import yaml
	with open("config.yaml", 'r') as stream:
		config = yaml.safe_load(stream)
	train_gcn(config)
