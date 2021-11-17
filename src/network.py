import torch
from torch.utils.data import DataLoader
from math import sqrt


class FCNN(torch.nn.Module):
	def __init__(self, layers :list):
		super().__init__()

		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		assert len(layers) >= 2 # "At least two layers are required (incl. input and output layer)"

		self.layers = layers

		linear_layers = []

		for i in range(len(self.layers) -1):
			n_in = self.layers[i]
			n_out = self.layers[i+1]
			layer = torch.nn.Linear(n_in, n_out)

            # Initialize weights and biases
			a = 1 if i == 0 else 2
			layer.weight.data = torch.randn((n_out, n_in)) * sqrt(a/n_in) # HE initialization for i != 0.
			layer.bias.data = torch.zeros(n_out)

			# Add to list
			linear_layers.append(layer)

		# Modules/layers must be registered to enable saving of model
		self.linear_layers = torch.nn.ModuleList(linear_layers)

		# Non-linearity (e.g. ReLU, ELU, or SELU)
		self.act = torch.nn.ReLU(inplace=False)

	def forward(self, input :torch.Tensor):
		x = input
		for l in self.linear_layers[:-1]:
			x = l(x)
			x = self.act(x)

		output_layer = self.linear_layers[-1]
		return output_layer(x)

	def get_num_parameters(self):
		return sum(p.numel() for p in self.parameters())
    
	def save(self, path: str):
		torch.save({
			'model_state_dict': self.state_dict(),
		}, path)

	def load(self, path: str):
		checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
		self.load_state_dict(checkpoint['model_state_dict'])


def train(
        net: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        lr: float,
        l2_reg: float,
) -> torch.nn.Module:
    """
    Train model using mini-batch SGD
    After each epoch, we evaluate the model on validation data

    :param net: initialized neural network
    :param train_loader: DataLoader containing training set
    :param n_epochs: number of epochs to train
    :param lr: learning rate (default: 0.001)
    :param l2_reg: L2 regularization factor (default: 0)
    :return: torch.nn.Module: trained model.
    """

    # Define loss and optimizer
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Train Network
    for epoch in range(n_epochs):
        for inputs, labels in train_loader:
            # Zero the parameter gradients (from last iteration)
            optimizer.zero_grad()

            # Forward propagation
            outputs = net(inputs)
            
            # Compute cost function
            batch_mse = criterion(outputs, labels)
            
            reg_loss = 0
            for param in net.parameters():
                reg_loss += param.pow(2).sum()

            cost = batch_mse + l2_reg * reg_loss

            # Backward propagation to compute gradient
            cost.backward()
            
            # Update parameters using gradient
            optimizer.step()
        
        # Evaluate model on validation data
        mse_val = 0
        for inputs, labels in val_loader:
            mse_val += torch.sum(torch.pow(labels - net(inputs), 2)).item()
        mse_val /= len(val_loader.dataset)
        print(f'Epoch: {epoch + 1}: Val MSE: {mse_val}')
        
    return net
