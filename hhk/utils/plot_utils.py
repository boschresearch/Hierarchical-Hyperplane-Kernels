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

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import os
import matplotlib.pyplot as plt
from hhk.kernels.rbf_kernel import RBFKernel
from hhk.models.gp_model import GPModel
from hhk.kernels.hierarchical_hyperplane_kernel import HierarchicalHyperplaneKernel
from hhk.utils.plotter import Plotter
from hhk.utils.plotter2D import Plotter2D


def active_learning_nd_plot(x_data, y_data, save_plot=False, file_name=None, file_path=None):
    column_names = ["x" + str(i) for i in range(1, x_data.shape[1] + 1)] + ["y"]
    data = np.concatenate((x_data, y_data), axis=1)
    df = pd.DataFrame(data=data, columns=column_names)
    scatter_matrix(df, alpha=1.0, figsize=(6, 6), diagonal="kde")
    if save_plot:
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()
    else:
        plt.show()


def active_learning_1d_plot(
    x_grid,
    pred_mu_grid,
    pred_sigma_grid,
    x_data,
    y_data,
    x_query,
    y_query,
    gt_available=False,
    gt_x=None,
    gt_f=None,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    plotter_object = Plotter(1)
    if gt_available:
        plotter_object.add_gt_function(np.squeeze(gt_x), np.squeeze(gt_f), "black", 0)
    plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(y_data), "r", 0)
    plotter_object.add_datapoints(x_query, y_query, "green", 0)
    plotter_object.add_predictive_dist(np.squeeze(x_grid), np.squeeze(pred_mu_grid), np.squeeze(pred_sigma_grid), 0)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def active_learning_2d_plot(
    x_grid, acquisition_values_grid, pred_mu_grid, y_over_grid, x_data, x_query, save_plot=False, file_name=None, file_path=None
):
    plotter_object = Plotter2D(3)
    plotter_object.add_gt_function(x_grid, np.squeeze(acquisition_values_grid), "RdBu_r", 14, 0)
    plotter_object.add_datapoints(x_data, "black", 0)
    if len(x_query.shape) == 1:
        x_query = np.expand_dims(x_query, axis=0)

    plotter_object.add_datapoints(x_query, "green", 0)
    min_y = np.min(y_over_grid)
    max_y = np.max(y_over_grid)
    levels = np.linspace(min_y, max_y, 100)
    plotter_object.add_gt_function(x_grid, np.squeeze(y_over_grid), "seismic", levels, 1)
    plotter_object.add_datapoints(x_data, "black", 1)
    plotter_object.add_datapoints(x_query, "green", 1)
    plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu_grid), "seismic", levels, 2)
    plotter_object.add_datapoints(x_data, "black", 2)
    plotter_object.add_datapoints(x_query, "green", 2)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def plot_model_specifics(x_grid, x_data, model, save_plot=False, file_name=None, file_path=None):
    input_dimension = x_grid.shape[1]
    # cmaps = ['Greens','Reds','Blues','Purples']
    if isinstance(model, GPModel):
        if isinstance(model.model.kernel, HierarchicalHyperplaneKernel):
            hhk_specific_plot(x_grid, x_data, model, save_plot, file_name, file_path, input_dimension)


def hhk_specific_plot(x_grid, x_data, model, save_plot, file_name, file_path, input_dimension):
    if input_dimension == 2:
        classified_local_kernel = np.concatenate(model.model.kernel.gate(x_grid), axis=1)
        sorted_classes = np.argsort(-1 * np.sum(classified_local_kernel, axis=0))
        topology = model.model.kernel.get_topology()
        if topology == 3:
            plotter_object = Plotter2D(4, 2)
        else:
            plotter_object = Plotter2D(classified_local_kernel.shape[1])
        levels = np.linspace(0, 1, 100)
        if topology == 3:
            counter = 0
            v_index = 0
            for class_index in sorted_classes:
                if counter == 4:
                    v_index = 1
                    counter = 0
                plotter_object.add_gt_function(
                    x_grid, np.squeeze(classified_local_kernel[:, class_index]), "plasma", levels, counter, v_ax=v_index
                )
                plotter_object.add_datapoints(x_data, "red", counter, v_ax=v_index)
                if isinstance(model.model.kernel.kernel_list[class_index], RBFKernel):
                    lengthscales = model.model.kernel.kernel_list[class_index].kernel.lengthscales.numpy()
                    variance = model.model.kernel.kernel_list[class_index].kernel.variance.numpy()
                    plotter_object.add_text_box("ls_x1=" + "{:.2f}".format(lengthscales[0]), 0.30, 0.88, 0.5, 17, counter, v_ax=v_index)
                    plotter_object.add_text_box("ls_x2=" + "{:.2f}".format(lengthscales[1]), 0.30, 0.78, 0.5, 17, counter, v_ax=v_index)
                    plotter_object.add_text_box("var=" + "{:.2f}".format(variance[0]), 0.30, 0.68, 0.5, 17, counter, v_ax=v_index)
                counter += 1
        else:
            counter = 0
            for class_index in sorted_classes:
                plotter_object.add_gt_function(x_grid, np.squeeze(classified_local_kernel[:, class_index]), "plasma", levels, counter)
                plotter_object.add_datapoints(x_data, "red", counter)
                if isinstance(model.model.kernel.kernel_list[class_index], RBFKernel):
                    lengthscales = model.model.kernel.kernel_list[class_index].kernel.lengthscales.numpy()
                    variance = model.model.kernel.kernel_list[class_index].kernel.variance.numpy()
                    plotter_object.add_text_box("ls_x1=" + "{:.2f}".format(lengthscales[0]), 0.30, 0.88, 0.5, 17, counter)
                    plotter_object.add_text_box("ls_x2=" + "{:.2f}".format(lengthscales[1]), 0.30, 0.78, 0.5, 17, counter)
                    plotter_object.add_text_box("var=" + "{:.2f}".format(variance[0]), 0.30, 0.68, 0.5, 17, counter)
                counter += 1
        if save_plot:
            plotter_object.save_fig(file_path, file_name)
        else:
            plotter_object.show()


if __name__ == "__main__":
    pass
