from random import seed
import pandas as pd
import torch
from torch.utils.data import DataLoader
from plot import DataPlotter
from dataHandler import DataHandler
from network import FCNN, train

def main():
	datahandler = DataHandler("../data/well_data.csv", index_col=0, seed=12345)

	df = datahandler.get_dataframe()

	plotter = DataPlotter(subplots=True, subplot_size=(2,2))

	plotter.fill_subplot(pos=(0,0), data=df['CHK'], label='CHK')
	plotter.fill_subplot(pos=(0,1), data=df['TWH'], label='TWH')
	plotter.fill_subplot(pos=(1,0), data=df['PWH'] - df['PDC'], label='PWH - PDC')
	plotter.fill_subplot(pos=(1,1), data=df['FOIL'], label='FOIL')
	plotter.fill_subplot(pos=(1,1), data=df['FGAS'], label='FGAS', opt_args='--r')

	plotter.show()

	datahandler.generate_data_sets(test_indices=(2000, 2500), validation_frac=0.1)

	scatter = DataPlotter(subplots=False)
	scatter.fill_scatter(indices=datahandler.get_train_set().index, values=datahandler.get_train_set()['QTOT'], color='black', label='Train')
	scatter.fill_scatter(indices=datahandler.get_val_set().index, values=datahandler.get_val_set()['QTOT'], color='green', label='Val')
	scatter.fill_scatter(indices=datahandler.get_test_set().index, values=datahandler.get_test_set()['QTOT'], color='red', label='Test')

	scatter.show()

	input_cols = ['CHK', 'PWH', 'PDC', 'TWH', 'FGAS', 'FOIL']
	output_cols = ['QTOT']

	datahandler.set_input_cols(input_cols)
	datahandler.set_output_cols(output_cols)

	train_loader = datahandler.generate_dataloader(set='train')
	val_loader = datahandler.generate_dataloader(set='val')

	layers = [len(input_cols), 50, 50, len(output_cols)]
	net = FCNN(layers)

	n_epochs = 100
	lr = 0.001
	l2_reg = 0.001  # 10
	net = train(net, train_loader, val_loader, n_epochs, lr, l2_reg)

if __name__ == "__main__":
	main()