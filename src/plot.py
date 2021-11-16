import matplotlib.pyplot as plt

_num = 0


class DataPlotter(object):
	def __init__(self, subplots :bool=False, subplot_size :tuple=None, size :tuple=(16, 9)):
		global _num
		self.num = _num
		_num += 1

		self.subplots = subplots
		self.fig_size = size

		if self.subplots:
			self.subplot_size = subplot_size
			self.fig, self.ax = plt.subplots(self.subplot_size[0], self.subplot_size[1], figsize=self.fig_size, num=self.num)
		
		else:
			self.fig = plt.figure(num=self.num, figsize=size)

	### TODO: Add xlabel and ylabel as optional arguments 
	def fill_subplot(self, pos :tuple, data, label :str, legend :bool=True, opt_args :str=None):
		assert self.subplots

		self.activate()

		self._check_subplot_ranges(pos)

		if opt_args is not None:
			self.ax[pos[0], pos[1]].plot(data, opt_args, label=label)

		else:
			self.ax[pos[0], pos[1]].plot(data, label=label)

		if legend:
			self.ax[pos[0], pos[1]].legend()
	
	def _check_subplot_ranges(self, pos :tuple):
		self.activate()

		if pos[0] < 0 or pos[1] < 0:
			raise IndexError('Subplot position negative.')
		if pos[0] >= self.subplot_size[0] or pos[1] >= self.subplot_size[1]:
			raise IndexError('Subplot position out of range.')

	### TODO: Add xlabel and ylabel as optional arguments
	def fill_plot(self, data, label :str, legend :bool=True, opt_args :str=None):
		self.activate()

		if opt_args is not None:
			plt.plot(data, opt_args, label=label)
		
		else:
			plt.plot(data, label=label)

		if legend:
			plt.legend()

	def fill_scatter(self, indices, values, label :str, color :str='black', legend :bool=True):
		self.activate()

		plt.scatter(indices, values, color=color, label=label)

		if legend:
			plt.legend()

	def show(self):
		self.activate()
		plt.show()

	def activate(self):
		plt.figure(num=self.num)

