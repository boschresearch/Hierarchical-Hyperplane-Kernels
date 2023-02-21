# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Matthias Bitzer, matthias.bitzer3@de.bosch.com

from enum import Enum
import matplotlib.pyplot as plt
import numpy as np
import os


class Plotter:
    def __init__(self, num_axes, share_x=False, share_y=False):
        self.num_axes = num_axes
        self.fig, self.axes = plt.subplots(num_axes, 1, sharex=share_x, sharey=share_y)

    def add_gt_function(self, x, ground_truth, color, ax_num, sort_x=True):
        if sort_x:
            sorted_indexes = np.argsort(x)
            self.give_axes(ax_num).plot(x[sorted_indexes], ground_truth[sorted_indexes], color=color)
        else:
            self.give_axes(ax_num).plot(x, ground_truth, color=color)

    def add_datapoints(self, x_data, y_data, color, ax_num):
        self.give_axes(ax_num).plot(x_data, y_data, "o", color=color)

    def give_axes(self, ax_num):
        if self.num_axes > 1:
            return self.axes[ax_num]
        else:
            return self.axes

    def add_posterior_functions(self, x, predictions, ax_num):
        num_predictions = predictions.shape[0]
        for i in range(0, num_predictions):
            self.give_axes(ax_num).plot(x, predictions[i], color="r", linewidth="0.5")

    def add_predictive_dist(self, x, pred_mu, pred_sigma, ax_num, sort_x=True):
        if sort_x:
            sorted_index = np.argsort(x)
            x = x[sorted_index]
            pred_mu = pred_mu[sorted_index]
            pred_sigma = pred_sigma[sorted_index]
        axes = self.give_axes(ax_num)
        axes.plot(x, pred_mu, color="g")
        axes.fill_between(x, pred_mu - pred_sigma, pred_mu + pred_sigma, alpha=0.8, color="b")
        axes.fill_between(x, pred_mu - 2 * pred_sigma, pred_mu + 2 * pred_sigma, alpha=0.3, color="b")

    def add_confidence_bound(self, x, pred_mu, bound_width, ax_num, sort_x=True):
        if sort_x:
            sorted_index = np.argsort(x)
            x = x[sorted_index]
            pred_mu = pred_mu[sorted_index]
            bound_width = bound_width[sorted_index]
        axes = self.give_axes(ax_num)
        axes.plot(x, pred_mu, color="g")
        axes.fill_between(x, pred_mu - bound_width, pred_mu + bound_width, alpha=0.3, color="b")

    def add_multiple_confidence_bound(self, x, pred_mu, bound_width, ax_num, sort_x=True):
        assert np.shape(pred_mu) == np.shape(bound_width)
        if len(np.shape(pred_mu)) == 1 or np.shape(pred_mu)[1] == 1:
            self.add_confidence_bound(x, np.squeeze(pred_mu), np.squeeze(bound_width), ax_num, sort_x)
        if sort_x:
            sorted_index = np.argsort(x)
            x = x[sorted_index]
            pred_mu = pred_mu[sorted_index]
            bound_width = bound_width[sorted_index]
        axes = self.give_axes(ax_num)
        for i in range(pred_mu.shape[1]):
            axes.plot(x, pred_mu[..., i], color=f"C{i}")
            axes.fill_between(x, pred_mu[..., i] - bound_width[..., i], pred_mu[..., i] + bound_width[..., i], alpha=0.3, color=f"C{i}")

    def add_hline(self, y_value, color, ax_num):
        if y_value != np.inf and y_value != -np.inf:
            self.give_axes(ax_num).axhline(y_value, color=color, linewidth=0.8, linestyle="--")

    def add_multiple_hline(self, y_values, ax_num):
        for i, y in enumerate(y_values):
            if y == np.inf or y == -np.inf:
                continue
            self.add_hline(y, f"C{i}", ax_num)

    def add_vline(self, x_value, color, ax_num):
        if x_value != np.inf and x_value != -np.inf:
            self.give_axes(ax_num).axvline(x_value, color=color, linestyle="--")

    def add_safety_region(self, safe_x, ax_num):
        min_y = self.give_axes(ax_num).get_ylim()[0]
        self.give_axes(ax_num).plot(safe_x, np.repeat(min_y, safe_x.shape[0]), "_", linewidth=10.0, color="green")

    def add_query_region(self, query_x_grid, ax_num):
        min_y = self.give_axes(ax_num).get_ylim()[0]
        self.give_axes(ax_num).plot(query_x_grid, np.repeat(min_y, query_x_grid.shape[0]), "_", linewidth=10.0, color="purple")

    def save_fig(self, file_path, file_name):
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()

    def show(self):
        plt.show()
