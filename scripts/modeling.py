from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from os import path, remove
import numpy as np
from mne.decoding import SPoC
import pickle as pkl
from .constants import COV_METHOD, SPOC_COMPONENTS, BANDS, RT_MAD_FACTOR
import shap
import matplotlib.pyplot as plt
from scipy.stats import median_abs_deviation


def sd(x):
    return np.var(x, ddof=1) ** .5


def aggregate(x, func, by):
    by = np.array(by)
    groups = np.unique(by)
    x = np.array(x)
    aggs = {group: func(x[by == group]) for group in groups}
    return aggs


def get_spocs(Xs, zs):
    x_spocs = {}
    for band_name, eeg_epochs in Xs.items():
        spoc = SPoC(
            n_components=len(SPOC_COMPONENTS),
            reg=COV_METHOD,
            rank='full'
            )        
        spoc.fit(eeg_epochs, zs)
        x_spocs[band_name] = spoc
    return x_spocs


def evaluate_mae_per_angle(observations, predictions, angles):
    observations, predictions, angles = [np.array(_) for _ in [observations, predictions, angles]]
    mae = {'all': mean_absolute_error(observations, predictions)}
    for angle in np.unique(angles):
        mae[angle] = mean_absolute_error(observations[angles == angle], predictions[angles == angle])
    return mae


class CrossValidation(object):
    def __init__(self, n_samples, n_windows, rel_train_size):
        self.n_samples = int(n_samples)
        self.n_windows = int(n_windows)
        self.rel_train_size = rel_train_size
        self._n_samples_remaining = int(n_samples)
        self.preprocessors = []

    def define_preprocessors(self, Xs, ys, angles):
        for n_windows_remaining in range(self.n_windows, 0, -1):
            preprocessor = self._define_preprocessor(n_windows_remaining, Xs, ys, angles)
            self.preprocessors.append(preprocessor)
        self.reset()

    def _define_preprocessor(self, n_windows, Xs, ys, angles):
        train, test = self._get_next_slicers(n_windows)
        Xs_train = {band_name: epochs[train] for band_name, epochs in Xs.items()}
        preprocessor = Preprocessor()
        preprocessor.define(Xs_train, ys[train], angles[train])
        return preprocessor

    def apply(self, model, Xs, ys, angles):
        if not any(self.preprocessors):
            self.define_preprocessors(Xs, ys, angles)
        maes = self._measure_maes(model, Xs, ys, angles)
        mean_mae = np.mean(maes)
        self.reset()
        return mean_mae

    def _measure_maes(self, model, Xs, ys, angles):
        maes = []
        for n_windows_remaining, preprocessor in zip(range(self.n_windows, 0, -1), self.preprocessors):
            mae = self._measure_mae(n_windows_remaining, model, Xs, ys, angles, preprocessor)
            maes.append(mae)
        return maes

    def _measure_mae(self, n_windows_remaining, model, Xs, ys, angles, preprocessor):
        train, test = self._get_next_slicers(n_windows_remaining)
        Xs_train = {band_name: epochs[train] for band_name, epochs in Xs.items()}
        model.train(Xs_train, ys[train], angles[train], preprocessor)
        Xs_test = {band_name: epochs[test] for band_name, epochs in Xs.items()}
        mae = model.evaluate(Xs_test, ys[test], angles[test])
        return mae['all']

    def _get_next_slicers(self, n_windows):
        n_train, n_test = self._get_next_win_sizes(n_windows)
        train = self._get_next_train_slice(n_train)
        test_stop = self.n_samples if n_windows == 1 else train.stop + n_test
        test = slice(train.stop, test_stop)
        self._n_samples_remaining -= n_train
        return train, test

    def _get_next_win_sizes(self, n_windows):
        win_size = self._n_samples_remaining / (((n_windows - 1) * self.rel_train_size) + 1)
        n_train = int(win_size * self.rel_train_size)
        n_test = int(win_size - n_train)
        return n_train, n_test

    def _get_next_train_slice(self, n_train):
        train_start = self.n_samples - self._n_samples_remaining
        train_stop = train_start + n_train
        train = slice(train_start, train_stop)
        return train

    def reset(self):
        self._n_samples_remaining = self.n_samples


class Tuner(object):
    def __init__(self, epochs, cv_n_windows, cv_rel_train_size):
        self.epochs = epochs
        self.n_samples = len(epochs['rt'])
        self.cv_n_windows = cv_n_windows
        self.cv_rel_train_size = cv_rel_train_size
        self.cv = self.__load_cv()
        self.tuned_lambda = None
        self.tuned_error = None
        self.error_measure = 'MAE'

    def __load_cv(self):
        cv = CrossValidation(self.n_samples, self.cv_n_windows, self.cv_rel_train_size)
        cv.define_preprocessors(
            Xs=self.epochs['eeg'], 
            ys=self.epochs['rt'], 
            angles=self.epochs['angle']
            )  # use same SPoC components and RT standardizers for all LAMBDAS
        return cv

    def find_lambda(self, model, lambdas):
        errors = [self._measure_error(model, lambda_val) for lambda_val in lambdas]
        tuned_error = min(errors)
        tuned_lambda = lambdas[errors.index(tuned_error)]
        self.tuned_lambda = tuned_lambda
        self.tuned_error = tuned_error
        return tuned_lambda

    def _measure_error(self, model, lambda_val):
        m = model(lambda_val)
        # apply cross-validation and measure mean predictive performance across validation sets
        error = self.cv.apply(
            m, 
            Xs=self.epochs['eeg'], 
            ys=self.epochs['rt'], 
            angles=self.epochs['angle']
            )
        return error

    def save_tuned_lambda(self, output_path, new_file, **info):
        if new_file:
            with open(output_path, 'w') as f:
                header = ','.join(list(info.keys())) + f',n_samples,tuned_lambda,{self.error_measure}'
                f.write(header)
                print(f'{output_path} written')
        with open(output_path, 'a') as f:
            line = list(info.values()) + [str(val) for val in [self.n_samples, self.tuned_lambda, self.tuned_error]]
            line = '\n' + ','.join(line)
            f.write(line)
            print(f'{output_path} appended')


class EEGmodel(Ridge):
    def __init__(self, alpha):
        self.preprocessor = None
        self.shap_values = None
        super().__init__(alpha)

    def train(self, Xs, ys, angles, preprocessor=None):
        if preprocessor is None:
            preprocessor = Preprocessor()
            preprocessor.define(Xs, ys, angles)
        samples_to_include = preprocessor.which_samples_to_include(ys, angles)
        Xs = {band: epochs[samples_to_include][:][:] for band, epochs in Xs.items()}
        ys = np.array(ys)[samples_to_include]
        angles = np.array(angles)[samples_to_include]
        features = preprocessor.extract_features(Xs)
        labels = preprocessor.extract_labels(ys, angles)
        self.set_preprocessor(preprocessor)
        self.fit(features, labels)

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def evaluate(self, Xs, ys, angles):
        angles = np.array(angles)
        predicted_labels = self.get_predicted_labels(Xs)
        labels = self.preprocessor.extract_labels(ys, angles)
        mae = evaluate_mae_per_angle(labels, predicted_labels, angles)
        return mae

    def get_predicted_labels(self, Xs):
        features = self.preprocessor.extract_features(Xs)
        predicted_labels = self.predict(features)
        return predicted_labels

    def evaluate_raw(self, Xs, ys, angles):
        angles = np.array(angles)
        predicted_labels = self.get_predicted_labels(Xs)
        predicted_ys = self.preprocessor.revert_labels_to_original_scale(predicted_labels, angles)
        mae = evaluate_mae_per_angle(ys, predicted_ys, angles)
        return mae

    def plot_feature_importance(self, Xs_train, Xs_test, output_path, **fig_kwargs):
        if self.shap_values is None:
            self.shap_values = self.get_shap_values(Xs_train, Xs_test)

        plot_pars = {
            'shap_values': self.shap_values,
            'features': self.preprocessor.extract_features(Xs_test),
            'feature_names': list(Xs_test.keys()),
            'show': False,
            'plot_size': (12, 6)
        }
        shap.summary_plot(**plot_pars)
        plt.gcf().axes[-1].set_aspect(100)
        plt.gcf().axes[-1].set_box_aspect(100)
        if path.isfile(output_path):
            remove(output_path)
        plt.savefig(output_path, **fig_kwargs)
        plt.close()
        print(f'{output_path} written')

    def get_shap_values(self, Xs_train, Xs_test):
        train_features = self.preprocessor.extract_features(Xs_train)
        explainer = shap.KernelExplainer(self.predict, train_features)
        test_features = self.preprocessor.extract_features(Xs_test)
        shap_values = explainer.shap_values(test_features, silent=True)
        return shap_values

    def save_shap_values_per_feature(self, Xs_train, Xs_test, output_path, write_header=True, **info):
        if write_header:
            header = list(info.keys()) + ['feature', 'mean_abs_shap']
            header = ','.join(header)
            with open(output_path, 'w') as f:
                f.write(header)

        if self.shap_values is None:
            self.shap_values = self.get_shap_values(Xs_train, Xs_test)
        shap_values_abs_mean = np.abs(self.shap_values).mean(axis=0)
        lines = ''
        for feature, shap_value in zip(Xs_test.keys(), shap_values_abs_mean):
            line = list(info.values()) + [feature, str(shap_value)]
            lines += '\n' + ','.join(line)
        with open(output_path, 'a') as f:
            f.write(lines)

    def plot_spoc_components(self, output_dir, info, _id, **fig_kwargs):
        for band, spoc in self.preprocessor.x_spocs.items():
            output_path = path.join(output_dir, f'{_id}_{band.replace(" ", "-")}_spoc-topomap.png')
            p = spoc.plot_patterns(
                info, 
                components=SPOC_COMPONENTS,
                scalings=100, 
                name_format='#%01d',
                title=f'{band} ({BANDS[band]["l_freq"]}-{BANDS[band]["h_freq"]}Hz)\nSPoC components',
                show=False
                )
            if path.isfile(output_path):
                remove(output_path)
            p.savefig(output_path, **fig_kwargs)
            plt.close()
            print(f'{output_path} written')



class RTmodel(object):
    def __init__(self):
        self.preprocessor = None

    def define_preprocessor(self, ys, angles):
        preprocessor = Preprocessor()
        preprocessor.define_for_ys(ys, angles)
        self.set_preprocessor(preprocessor)

    def set_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def evaluate(self, Xs, ys, angles):
        angles = np.array(angles)
        labels = self.preprocessor.extract_labels(ys, angles)
        predicted_labels = np.zeros_like(labels)  # predict the average RT
        mae = evaluate_mae_per_angle(labels, predicted_labels, angles)
        return mae

    def evaluate_raw(self, Xs, ys, angles):
        angles = np.array(angles)
        predicted_labels = np.zeros_like(ys)
        predicted_ys = self.preprocessor.revert_labels_to_original_scale(predicted_labels, angles)
        mae = evaluate_mae_per_angle(ys, predicted_ys, angles)
        return mae


class Preprocessor(object):
    def __init__(self):
        self.y_means = None
        self.y_sds = None
        self.z_cutoffs = None
        self.x_spocs = {}

    def define(self, Xs, ys, angles):
        self.define_for_ys(ys, angles)
        self.define_for_xs(Xs, ys, angles)

    def define_for_ys(self, ys, angles):
        self.y_means = aggregate(np.log(ys), np.mean, by=angles)
        self.y_sds = aggregate(np.log(ys), sd, by=angles)
        zs = self.extract_labels(ys, angles)
        mad = median_abs_deviation(zs, scale=1/RT_MAD_FACTOR)
        self.z_cutoffs = [np.median(zs) - mad, np.median(zs) + mad]

    def define_for_xs(self, Xs, ys, angles):
        zs = self.extract_labels(ys, angles)
        self.x_spocs = get_spocs(Xs, zs)

    def which_samples_to_include(self, ys, angles):
        zs = self.extract_labels(ys, angles)
        is_sample_included = np.logical_and(
            zs > min(self.z_cutoffs), 
            zs < max(self.z_cutoffs)
            )
        samples_to_include = np.argwhere(is_sample_included).flatten().tolist()
        return samples_to_include

    def extract_labels(self, ys, angles):
        ys_log = np.log(ys)
        zs = self.z_transform(ys_log, angles)
        return zs

    def revert_labels_to_original_scale(self, zs, angles):
        ys_log = self.revert_z_transform(zs, angles)
        ys = np.exp(ys_log)
        return ys

    def z_transform(self, ys, angles):
        z = np.zeros_like(ys)
        angles = np.array(angles)
        for angle in np.unique(angles):
            z_angle = (ys[angles == angle] - self.y_means[angle]) / self.y_sds[angle]
            z[angles == angle] = z_angle
        return z

    def revert_z_transform(self, zs, angles):
        ys_log = np.zeros_like(zs)
        angles = np.array(angles)
        for angle in np.unique(angles):
            ys_log_angle = (zs[angles == angle] * self.y_sds[angle]) + self.y_means[angle]
            ys_log[angles == angle] = ys_log_angle
        return ys_log

    def extract_features(self, Xs):
        n_epochs = len(list(Xs.values())[0])
        n_bands = len(Xs)
        n_components = len(SPOC_COMPONENTS)
        features = np.empty((n_epochs, n_bands * n_components))
        for nth_band, (band_name, epochs) in enumerate(Xs.items()):
            band = slice(nth_band * n_components, (nth_band * n_components) + n_components)
            band_features = self.extract_feature(band_name, epochs)
            features[:, band] = band_features
        return features

    def extract_feature(self, band_name, epochs):
        spoc = self.x_spocs[band_name]
        spatial_filters = np.stack([spoc.filters_[component] for component in SPOC_COMPONENTS])
        components = np.asarray([np.dot(spatial_filters, epoch) for epoch in epochs])
        feature = np.log(np.var(components, ddof=1, axis=2))
        return feature

    def save(self, f_path):
        with open(f_path, 'wb') as f:
            pkl.dump(self, f)
        print(f'{f_path} written')
