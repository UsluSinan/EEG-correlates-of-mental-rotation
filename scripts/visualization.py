import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from os import path, makedirs
import warnings



class ICAPlotsIPython:
	def __init__(self, ica, source, samples_window, exclusive_components, output_dir):
		self.ica = ica
		self.source = source
		self.samples_window = samples_window
		self.exclusive_components = exclusive_components
		self.output_dir = output_dir
		if not path.exists(output_dir):
			makedirs(output_dir)
		self.figures = []
		self.show()

	def save_component_plot(self, component):
		matplotlib.use('agg')
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			f_names, titles = self.save_component_subplots(component)
			fig = plt.figure(figsize=(15, 5))
			grid = fig.add_gridspec(nrows=1, ncols=6)
			col_ranges = [[0, 1], [1, 4], [4, 6]]
			for name, col_range, title in zip(f_names, col_ranges, titles):
				ax = fig.add_subplot(grid[0, col_range[0]:col_range[1]])
				ax.imshow(mpimg.imread(name))
				ax.set_title(title, fontsize=10)
				ax.set_frame_on(False)
				[axis.set_visible(False) for axis in [ax.xaxis, ax.yaxis]]
			fig.suptitle(f'Component #{component}', fontsize=20)
			f_name = path.join(self.output_dir, f'_{component}.png')
			fig.savefig(f_name)
			fig.clear()
			del fig
		return f_name

	def save_component_subplots(self, component):
		topography = self.ica.plot_components(picks=component, title='')
		topography.axes[0].set_title('')
		sources = self.ica.get_sources(self.source)
		psd = sources.plot_psd(picks=component, fmax=80, dB=True,
							   estimate='power')
		psd.axes[0].set_title('')
		time, ax = plt.subplots()
		time_window = [self.samples_window[0] / self.source.info['sfreq'],
					   (self.samples_window[1] - 1) / self.source.info['sfreq']]
		xs = np.linspace(time_window[0], time_window[1],
						 self.samples_window[1] - self.samples_window[0])
		ys = sources.get_data(picks=component)[0, self.samples_window[0]:
												  self.samples_window[1]]
		ax.plot(xs, ys, color='k', linewidth=.5)
		ax.set_title('')
		ax.set_xlabel('time [s]')
		ax.set_ylabel('a.u.')
		plots = [topography, psd, time]
		f_names = [path.join(self.output_dir, f'_{component}_{i}.png') for i, _ in enumerate(plots)]
		titles = ['Scalp Field Topography', 'Power Spectral Density', 'Source Time Series']
		[plt.close(p) for p in plots]
		[p.savefig(name, dpi=600) for name, p in zip(f_names, plots)]
		[p.clear() for p in plots]
		del topography, psd, time, plots
		return f_names, titles

	def show(self):
		# n_figs = self.ica.n_components
		# matplotlib.rcParams['figure.max_open_warning'] = n_figs + 1
		f_names = []
		if self.exclusive_components is None:
			components = range(self.ica.n_components)
		else:
			components = self.exclusive_components

		for component in components:
			f_name = self.save_component_plot(component)
			f_names.append(f_name)
