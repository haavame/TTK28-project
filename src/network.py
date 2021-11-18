import torch
from torch.utils.data import DataLoader
from dataHandler import DataHandler
from math import sqrt


class FCNN(torch.nn.Module):
    def __init__(self, layers :list, dropout :float):
        super().__init__()

        if dropout:
            assert dropout > 0 # Ensure positive dropout value
            self.enable_dropout = True
            self.dropout = torch.nn.Dropout(dropout)

        else:
            self.enable_dropout = False

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert len(layers) >= 2 # "At least two layers are required (incl. input and output layer)"

        self.layers = layers

        # Initialize with random weights and zero biases
        self._initialize_weights()

        # Non-linearity (e.g. ReLU, ELU, or SELU)
        self.act = torch.nn.ReLU(inplace=False)

    def _initialize_weights(self):
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

    def _randomize_weights(self):
        for i in range(len(self.layers) -1):
            n_in = self.layers[i]
            n_out = self.layers[i+1]
            layer = self.linear_layers[i]

            # Initialize weights and biases
            a = 1 if i == 0 else 2
            layer.weight.data = torch.randn((n_out, n_in)) * sqrt(a/n_in) # HE initialization for i != 0.
            layer.bias.data = torch.zeros(n_out)

        # for i in range(len(self.linear_layers)):
        #     l = self.linear_layers[i]

        #     a = 1 if i == 0 else 2
        #     l.weight.data = torch.randn((l.out_features, l.in_features)) * sqrt(a/l.in_features)
        #     l.bias.data = torch.zeros(l.out_parameters)

    def forward(self, input :torch.Tensor) -> torch.nn.Module:
        x = input
        for l in self.linear_layers[:-1]:
            x = l(x)
            
            if self.enable_dropout:
                x = self.dropout(x)

            x = self.act(x)

        output_layer = self.linear_layers[-1]
        return output_layer(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def save(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.load_state_dict(checkpoint['model_state_dict'])


class Trainer(FCNN):
    def __init__(self, layers :list, data :DataHandler, l1 :bool=False, l2 :bool=False, early_stopping :bool=False, dropout :float=0):
        super().__init__(layers, dropout)

        self.l1 = l1
        self.l2 = l2
        self.early_stopping = early_stopping
        self.progress = []

        # Generate_loaders
        self.datahandler = data

        self.train_loader = self.datahandler.generate_dataloader(set='train')
        self.val_loader = self.datahandler.generate_dataloader(set='val')
        self.train_val_loader = self.datahandler.generate_dataloader(set='train_val')

    def train_network(self, n_epochs :int, lr :float=0.001, l1_reg :float=None, l2_reg :float=None, patience :int=None, retrain :bool=False) ->None:
        self._verify_inputs(l1_reg=l1_reg, l2_reg=l2_reg, patience=patience)

        self.train()

        # Define loss and optimizer
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        if self.early_stopping:
            j = i = 0
            optimal_mse = float('inf')
            best_state = self.state_dict()
            optimal_iterations = i

            while j < patience:
                self._training_loop(n_epochs=n_epochs, l1_reg=l1_reg, l2_reg=l2_reg, loader=self.train_loader, criterion=criterion, optimizer=optimizer, validate=True)
            
                i += n_epochs

                mse_val = self._calculate_mse(self.val_loader)

                if mse_val < optimal_mse:
                    j = 0
                    best_state = self.state_dict()
                    optimal_iterations = i
                    optimal_mse = mse_val
                
                else:
                    j += 1
                
            if retrain:
                self._randomize_weights()

                print(f"training for {optimal_iterations} iterations")

                self._training_loop(n_epochs=optimal_iterations, l1_reg=l1_reg, l2_reg=l2_reg, loader=self.train_val_loader, criterion=criterion, optimizer=optimizer)

            else:
                self.load_state_dict(best_state)    
              
        else:
            self._training_loop(n_epochs=n_epochs, l1_reg=l1_reg, l2_reg=l2_reg, loader=self.train_loader, criterion=criterion, optimizer=optimizer, validate=True)

    def _verify_inputs(self, l1_reg :float, l2_reg :float, patience :int) -> None:
        if self.l1 and (l1_reg is None):
            raise TypeError(f"Cannot perform l1 regularization with l1_reg {l1_reg}")

        if self.l2 and (l2_reg is None):
            raise TypeError(f"Cannot perform l1 regularization with l2_reg {l2_reg}")

        if self.early_stopping and (patience is None):
            raise TypeError(f"Cannot perform early stopping with patience {patience}")
        
        # if self.enable_dropout and (dropout is None):
        #     raise TypeError(f"Cannot perform early stopping with dropout {dropout}")

    def _calculate_mse(self, loader :DataLoader):
        mse_val = 0
        self.eval()

        for inputs, labels in loader:
            mse_val += torch.sum(torch.pow(labels - self(inputs), 2)).item()

        mse_val /= len(loader.dataset)

        self.train()

        return mse_val

    def _training_loop(self, n_epochs :int, l1_reg :float, l2_reg :float, loader :DataLoader, criterion, optimizer, validate :bool=False):
        for epoch in range(n_epochs):
            for inputs, labels in loader:
                # Zero the parameter gradients (from last iteration)
                optimizer.zero_grad()

                # Forward propagation
                outputs = self(inputs)
                
                # Compute cost function
                batch_mse = criterion(outputs, labels)
                
                reg_loss = 0
                
                if self.l1:
                    l1_loss = 0
                    for param in self.parameters():
                        l1_loss += param.abs().sum()
                    
                    reg_loss += l1_reg * l1_loss

                if self.l2:
                    l2_loss = 0
                    for param in self.parameters():
                        l2_loss += param.pow(2).sum()
                    
                    reg_loss += l2_reg * l2_loss

                cost = batch_mse + reg_loss

                # Backward propagation to compute gradient
                cost.backward()
                
                # Update parameters using gradient
                optimizer.step()

            if validate:
                mse_val = self._calculate_mse(self.val_loader)    
                print(f'Epoch: {epoch + 1}: Val MSE: {mse_val}')

                self.progress.append(mse_val)
            
            # else:
            #     print()

    def evaluate(self, input :torch.Tensor, value :torch.Tensor) -> tuple:
        # x, y = self.datahandler.generate_tensor(set=mode)

        self.eval()

        pred = self(input)

        mse = torch.mean(torch.pow(pred - value, 2))

        mae = torch.mean(torch.abs(pred - value))

        mape = 100*torch.mean(torch.abs(torch.div(pred - value, value)))

        return mse.item(), mae.item(), mape.item()

    def predict(self, input :torch.Tensor) -> torch.Tensor:
        self.eval()

        return self(input).detach().numpy()

    def get_progress(self):
        return self.progress

def train(
        net: torch.nn.Module,
        train_loader: DataLoader,
        n_epochs: int,
        lr: float,
        l2_reg: float,
        val_loader: DataLoader=None,
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
        

        # Evaluate model on validation data if exists
        if val_loader is not None:
            mse_val = 0
            for inputs, labels in val_loader:
                mse_val += torch.sum(torch.pow(labels - net(inputs), 2)).item()
            mse_val /= len(val_loader.dataset)
            print(f'Epoch: {epoch + 1}: Val MSE: {mse_val}')
        
    return net
