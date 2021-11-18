from random import seed
import pandas as pd
import torch
from torch.utils.data import DataLoader
from plot import DataPlotter
from dataHandler import DataHandler
from network import Trainer

def main():
	datahandler = DataHandler("../data/well_data.csv", index_col=0, seed=12345, test_indices=(2000, 2500), validation_frac=0.1)
	# datahandler.generate_data_sets(test_indices=(2000, 2500), validation_frac=0.1)

	#draw_plots(datahandler)

	input_cols = ['CHK', 'PWH', 'PDC', 'TWH', 'FGAS', 'FOIL']
	output_cols = ['QTOT']

	datahandler.set_input_cols(input_cols)
	datahandler.set_output_cols(output_cols)

	layers = [len(input_cols), 50, 50, len(output_cols)]

	n_epochs = 100
	n_steps = 5
	lr = 0.001
	l2_reg = 0.001  # 10
	l1_reg = 0.001  # 10
	patience = 10
	retrain = False

	trainer = Trainer(layers=layers, data=datahandler, early_stopping=True)

	trainer.train(n_epochs=n_steps, lr=lr, l2_reg=l2_reg, l1_reg=l1_reg, patience=patience, retrain=retrain)

	mse_value, mae_value, mape_value = trainer.evaluate(mode='test')

	print_evaluation(mse_value, mae_value, mape_value)


def draw_plots(datahandler: DataHandler):
	df = datahandler.get_dataframe()

	plotter = DataPlotter(subplots=True, subplot_size=(2,2))

	plotter.fill_subplot(pos=(0,0), data=df['CHK'], label='CHK')
	plotter.fill_subplot(pos=(0,1), data=df['TWH'], label='TWH')
	plotter.fill_subplot(pos=(1,0), data=df['PWH'] - df['PDC'], label='PWH - PDC')
	plotter.fill_subplot(pos=(1,1), data=df['FOIL'], label='FOIL')
	plotter.fill_subplot(pos=(1,1), data=df['FGAS'], label='FGAS', opt_args='--r')

	plotter.show()

	scatter = DataPlotter(subplots=False)
	scatter.fill_scatter(indices=datahandler.get_train_set().index, values=datahandler.get_train_set()['QTOT'], color='black', label='Train')
	scatter.fill_scatter(indices=datahandler.get_val_set().index, values=datahandler.get_val_set()['QTOT'], color='green', label='Val')
	scatter.fill_scatter(indices=datahandler.get_test_set().index, values=datahandler.get_test_set()['QTOT'], color='red', label='Test')

	scatter.show()

def print_evaluation(mse_value :float, mae_value :float, mape_value :float):
	print(f'MSE: {mse_value}')

	print(f'MAE: {mae_value}')

	print(f'MAPE: {mape_value} %')

if __name__ == "__main__":
	main()