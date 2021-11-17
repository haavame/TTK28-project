import pandas as pd
import torch
from torch.utils.data import DataLoader

class DataHandler(object):
	def __init__(self, file_name :str, index_col :int, seed: int, input_cols :list = None, output_cols :list = None):
		self.random_seed = seed
		self.df = pd.read_csv(file_name, index_col=index_col)

		self.input_cols = input_cols
		self.output_cols = output_cols

	def get_dataframe(self):
		return self.df

	def generate_data_sets(self, test_indices :tuple, validation_frac :float):
		self._extract_test_set(test_indices)

		self._extract_validation_set(frac=validation_frac)

		self._validate_set_sizes()

	def _extract_test_set(self, indices :tuple):
		self.test_set = self.df.iloc[indices[0]:indices[1]]

		self.train_val_set = self.df.copy().drop(self.test_set.index)

	def _extract_validation_set(self, frac :float):
		self.val_set = self.train_val_set.sample(frac=frac, replace=False, random_state=self.random_seed)

		self.train_set = self.train_val_set.copy().drop(self.val_set.index)
		
	def _validate_set_sizes(self):
		num_points = len(self.train_set) + len(self.val_set) + len(self.test_set)

		if num_points != len(self.df):
			raise AssertionError("Faulty data set lengths.")

	def get_test_set(self):
		return self.test_set

	def get_train_val_set(self):
		return self.train_val_set

	def get_train_set(self):
		return self.train_set

	def get_val_set(self):
		return self.val_set

	def set_input_cols(self, input_cols):
		self.input_cols = input_cols

	def set_output_cols(self, output_cols):
		self.output_cols = output_cols

	def get_input_cols(self):
		return self.input_cols
	
	def get_output_cols(self):
		return self.output_cols

	def generate_dataloader(self, set :str = 'train') -> tuple:
		if set == 'val':
			x, y = self.generate_tensor('val')
			shuffle = False

		elif set == 'train':
			x, y = self.generate_tensor('train')
			shuffle = True

		else:
			raise ValueError('{error} undefined'.format(error=set))

		dataset = torch.utils.data.TensorDataset(x, y)
		loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=shuffle)

		return loader

	def generate_tensor(self, set :str = 'train') -> tuple:
		if set == 'test':
			input_tensor = torch.from_numpy(self.test_set[self.input_cols].values).to(torch.float)
			output_tensor = torch.from_numpy(self.test_set[self.output_cols].values).to(torch.float)

		elif set == 'val':
			input_tensor = torch.from_numpy(self.val_set[self.input_cols].values).to(torch.float)
			output_tensor = torch.from_numpy(self.val_set[self.output_cols].values).to(torch.float)

		elif set == 'train':
			input_tensor = torch.from_numpy(self.train_set[self.input_cols].values).to(torch.float)
			output_tensor = torch.from_numpy(self.train_set[self.output_cols].values).to(torch.float)

		else:
			raise ValueError('{error} undefined'.format(error=set))

		return input_tensor, output_tensor


