# evaluation/BlandAltmanPy.py

import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy.stats import gaussian_kde


class BlandAltman():

    def __init__(self, gold_std, new_measure, config, averaged=False):
        # set averaged to True if multiple observations from each participant are averaged together to get one value
        import pandas as pd

        # Check that inputs are list or pandas series, convert to series if list
        if isinstance(gold_std, list) or isinstance(gold_std, (np.ndarray, np.generic)):
            df = pd.DataFrame()  # convert to pandas series
            df['gold_std'] = gold_std
            gold_std = df.gold_std
        elif not isinstance(gold_std, pd.Series):
            print('Error: Data type of gold_std is not a list or a Pandas series or Numpy array')

        if isinstance(new_measure, list) or isinstance(new_measure, (np.ndarray, np.generic)):
            df2 = pd.DataFrame()  # convert to pandas series
            df2['new_measure'] = new_measure
            new_measure = df2.new_measure
        elif not isinstance(new_measure, pd.Series):
            print('Error: Data type of new_measure is not a list or a Pandas series or Numpy array')

        self.gold_std = gold_std
        self.new_measure = new_measure

        # Calculate Bland-Altman statistics
        diffs = gold_std - new_measure
        self.mean_error = diffs.mean()
        self.std_error = diffs.std()
        self.mean_absolute_error = diffs.abs().mean()
        self.mean_squared_error = (diffs ** 2).mean()
        self.root_mean_squared_error = np.sqrt((diffs ** 2).mean())
        r = np.corrcoef(self.gold_std, self.new_measure)
        self.correlation = r[0, 1]  # correlation coefficient
        diffs_std = diffs.std()  # 95% Confidence Intervals
        corr_std = np.sqrt(2 * (diffs_std ** 2))  # if observations are averaged, used corrected standard deviation
        sqrt_sample_size = math.sqrt(self.gold_std.shape[0])
        if averaged:
            self.CI95 = [self.mean_error + 1.96 * corr_std, self.mean_error - 1.96 * corr_std]
        else:
            self.CI95 = [self.mean_error + 1.96 * diffs_std, self.mean_error - 1.96 * diffs_std]

        # Define save path
        if config.TOOLBOX_MODE == 'train_and_test' or config.TOOLBOX_MODE == 'only_test':
            self.save_path = os.path.join(config.LOG.PATH, config.TEST.DATA.EXP_DATA_NAME, 'bland_altman_plots')
        elif config.TOOLBOX_MODE == 'unsupervised_method':
            self.save_path = os.path.join(config.LOG.PATH, config.UNSUPERVISED.DATA.EXP_DATA_NAME, 'bland_altman_plots')
        else:
            raise ValueError('TOOLBOX_MODE only supports train_and_test, only_test, or unsupervised_method!')

        # Make the save path, if needed
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def print_stats(self, round_amount=5):
        print("Mean error = {}".format(round(self.mean_error, round_amount)))
        print("Mean absolute error = {}".format(round(self.mean_absolute_error, round_amount)))
        print("Mean squared error = {}".format(round(self.mean_squared_error, round_amount)))
        print("Root mean squared error = {}".format(round(self.root_mean_squared_error, round_amount)))
        print("Standard deviation error = {}".format(round(self.std_error, round_amount)))
        print("Correlation = {}".format(round(self.correlation, round_amount)))
        print("+95% Confidence Interval = {}".format(round(self.CI95[0], round_amount)))
        print("-95% Confidence Interval = {}".format(round(self.CI95[1], round_amount)))

    def return_stats(self):
        # return dict of statistics
        stats_dict = {'mean_error': self.mean_error,
                      'mean_absolute_error': self.mean_absolute_error,
                      'mean_squared_error': self.mean_squared_error,
                      'root_mean_squared_error': self.root_mean_squared_error,
                      'correlation': self.correlation,
                      'CI_95%+': self.CI95[0],
                      'CI_95%-': self.CI95[1]}

        return stats_dict

    def rand_jitter(self, arr):
        stdev = .01 * (max(arr) - min(arr))
        return arr + np.random.randn(len(arr)) * stdev

    def scatter_plot(self,
                     x_label='Gold Standard', y_label='New Measure',
                     figure_size=(4, 4), show_legend=True,
                     the_title=' ',
                     file_name='BlandAltman_ScatterPlot.pdf',
                     is_journal=False, measure_lower_lim=40, measure_upper_lim=150):

        if is_journal:  # avoid use of type 3 fonts for journal paper acceptance
            import matplotlib
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42

        x = self.rand_jitter(self.gold_std)
        y = self.rand_jitter(self.new_measure)

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_axes([0, 0, 1, 1])
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
        ax.scatter(x, y, c=z, s=50)
        x_vals = np.array(ax.get_xlim())
        ax.plot(x_vals, x_vals, '--', color='black', label='Line of Slope = 1')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(the_title)
        ax.grid()
        plt.xlim(measure_lower_lim, measure_upper_lim)
        plt.ylim(measure_lower_lim, measure_upper_lim)
        # file_name이 절대경로면 그대로, 아니면 save_path와 결합
        if os.path.isabs(file_name) or file_name.startswith('./') or os.path.dirname(file_name):
            save_to = file_name
        else:
            save_to = os.path.join(self.save_path, file_name)
        plt.savefig(save_to, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved scatter plot at: {save_to}")

    def difference_plot(self, x_label='Difference between rPPG HR and ECG HR [bpm]',
                        y_label='Average of rPPG HR and ECG HR [bpm]', averaged=False,
                        figure_size=(4, 4), show_legend=True,
                        the_title='', file_name='BlandAltman_DifferencePlot.pdf',
                        is_journal=False):

        if is_journal:  # avoid use of type 3 fonts for journal paper acceptance
            matplotlib.rcParams['pdf.fonttype'] = 42
            matplotlib.rcParams['ps.fonttype'] = 42

        diffs = self.gold_std - self.new_measure
        avgs = (self.gold_std + self.new_measure) / 2

        fig = plt.figure(figsize=figure_size)
        ax = fig.add_axes([0, 0, 1, 1])
        xy = np.vstack([avgs, diffs])
        z = gaussian_kde(xy)(xy)
        ax.scatter(avgs, diffs, c=z, label='Observations')
        x_vals = np.array(ax.get_xlim())
        ax.axhline(self.mean_error, color='black', label='Mean Error')
        ax.axhline(self.CI95[0], color='black', linestyle='--', label='+95% Confidence Interval')
        ax.axhline(self.CI95[1], color='black', linestyle='--', label='-95% Confidence Interval')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(the_title)
        if show_legend:
            ax.legend()
        ax.grid()
        # file_name 처리 로직 동일하게 적용
        if os.path.isabs(file_name) or file_name.startswith('./') or os.path.dirname(file_name):
            save_to = file_name
        else:
            save_to = os.path.join(self.save_path, file_name)
        plt.savefig(save_to, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Saved difference plot at: {save_to}")
