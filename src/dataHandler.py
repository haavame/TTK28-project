import pandas as pd

class DataHandler(object):
	def __init__(self, file_name :str, index_col :int, seed: int):
		self.random_seed = seed
		self.df = pd.read_csv(file_name, index_col=index_col)

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




