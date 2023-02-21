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
import random
from hhk.utils.plot_utils import active_learning_1d_plot, active_learning_2d_plot, active_learning_nd_plot, plot_model_specifics
from scipy.stats import norm
from typing import Tuple
from hhk.enums.active_learner_enums import AcquisitionFunctionType, ValidationType
from hhk.models.base_model import BaseModel
from hhk.active_learner.base_pool_active_learner import BasePoolActiveLearner


class ActiveLearner(BasePoolActiveLearner):
    """
    Main class for non-batch pool-based active learning - one query at a time - inherits from BasePoolActiveLearner

    Attributes:
        acquisition_function_type : AcquisitionFunctionType - Enum which acquisiton function type should be performed e.g. entropy, random,...
        validation_type : ValidationType - Enum which validation metric should be used e.g. RMSE, NegLoglikeli
        model : BaseModel - Model object that is an instance of a child class of BaseModel such as e.g. GPModel
        use_smaller_acquisition_set : bool - Bool if only a sampled subset of the pool should used for query selection (saves computational budget)
        smaller_set_size : int - Number of samples from the pool used for acquisition calculation
    """

    def __init__(
        self,
        acquisition_function_type: AcquisitionFunctionType,
        validation_type: ValidationType,
        use_smaller_acquistion_set: bool = False,
        smaller_set_size: int = 200,
    ):
        super().__init__()
        self.acquisition_function_type = acquisition_function_type
        self.validation_type = validation_type
        self.use_smaller_acquistion_set = use_smaller_acquistion_set
        self.smaller_set_size = smaller_set_size

    def set_model(self, model: BaseModel):
        """
        Sets model which is used for prediction and acquisition
        """
        self.model = model

    def reduce_grid(self, grid: np.ndarray, new_grid_size: int):
        """
        Gets a grid of points and reduces the grid - is used to reduce the pool for acquisition function calculation
        """
        print("-Reduce acquisition set ")
        grid_size = grid.shape[0]
        if grid_size > new_grid_size:
            indexes = np.random.choice(list(range(0, grid_size)), new_grid_size, replace=False)
            new_grid = grid[indexes]
            return new_grid
        else:
            return grid

    def update(self):
        """
        Main update function - infers the model on the current dataset, calculates the acquisition function and returns the query location
        """
        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)
        x_grid = self.pool.possible_queries()
        output_dimension = self.y_data.shape[1]

        if self.use_smaller_acquistion_set:
            x_grid = self.reduce_grid(x_grid, self.smaller_set_size)

        if self.acquisition_function_type == AcquisitionFunctionType.RANDOM:
            index = np.random.choice(list(range(0, x_grid.shape[0])), 1)[0]
            return x_grid[index]

        if self.acquisition_function_type == AcquisitionFunctionType.PRED_VAR:
            assert output_dimension == 1
            pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
            acquisition_function = pred_sigma
            new_query = x_grid[np.argmax(acquisition_function)]
            return new_query

        if self.acquisition_function_type == AcquisitionFunctionType.PRED_ENTROPY:
            acquisition_function = self.model.entropy_predictive_dist(x_grid)
            new_query = x_grid[np.argmax(acquisition_function)]
            return new_query

        return None

    def learn(self, n_steps: int, start_index=0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main Active learning loop - makes n_steps queries and updates the model after each query with the new x_data, y_data
        - validation is done after each query

        Arguments:
            n_steps : int - number of active learning iteration/number of queries
            start_index : int - important for plotting and logging - indicates that already start_index-1 AL steps where done previously
        Returns:
            np.array - validation metrics values over the iterations
            np.array - selected queries over the iterations
        """
        self.n_steps = n_steps
        for i in range(start_index, start_index + self.n_steps):
            query = self.update()
            print("Query")
            print(query)
            new_y = self.pool.query(query)
            if self.do_plotting:
                self.plot(query, new_y, i)
            self.x_data = np.vstack((self.x_data, query))
            self.y_data = np.vstack((self.y_data, new_y))
            self.validate()
        return np.array(self.validation_metrics), self.x_data

    def validate(self):
        """
        Validates the currently infered model on the test set - uses the validation_type to choose which metric should be used
        """
        if self.validation_type == ValidationType.RMSE:
            pred_mu, pred_sigma = self.model.predictive_dist(self.x_test)
            rmse = np.sqrt(np.mean(np.power(pred_mu - np.squeeze(self.y_test), 2.0)))
            self.validation_metrics.append(rmse)
        elif self.validation_type == ValidationType.NEG_LOG_LIKELI:
            log_likelis = self.model.predictive_log_likelihood(self.x_test, self.y_test)
            neg_log_likeli = np.mean(-1 * log_likelis)
            self.validation_metrics.append(neg_log_likeli)

    def plot(self, query: np.ndarray, new_y: np.ndarray, step: int):
        """
        Plotting function - gets the actual query and the AL step index und produces plots depending on the input and output dimension
        if self.plots is True the plots are saved to self.plot_path (both variables are set in the parent class)
        """
        dimension = self.x_data.shape[1]
        output_dimension = self.y_data.shape[1]
        x_grid = np.vstack((self.pool.possible_queries(), self.x_data))
        y_over_grid = np.vstack((self.pool.get_y_data(), self.y_data))
        if output_dimension == 1:
            if dimension == 1:
                self.plot_1d(query, new_y, step, x_grid)

            elif dimension == 2:
                pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
                if self.save_plots:
                    plot_name = "query_" + str(step) + ".png"
                    active_learning_2d_plot(
                        x_grid,
                        pred_sigma,
                        pred_mu,
                        y_over_grid,
                        self.x_data,
                        query,
                        save_plot=self.save_plots,
                        file_name=plot_name,
                        file_path=self.plot_path,
                    )
                else:
                    active_learning_2d_plot(x_grid, pred_sigma, pred_mu, y_over_grid, self.x_data, query)
            else:
                if self.save_plots:
                    plot_name = "query_" + str(step) + ".png"
                    active_learning_nd_plot(self.x_data, self.y_data, self.save_plots, plot_name, self.plot_path)
                else:
                    active_learning_nd_plot(self.x_data, self.y_data)

            if self.save_plots:
                plot_name = "model_specific" + str(step) + ".png"
                plot_model_specifics(
                    x_grid, self.x_data, self.model, save_plot=self.save_plots, file_name=plot_name, file_path=self.plot_path
                )
            else:
                plot_model_specifics(x_grid, self.x_data, self.model)

    def plot_1d(self, query, new_y, step, x_grid):
        pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
        if self.save_plots:
            plot_name = "query_" + str(step) + ".png"
            if self.ground_truth_available:
                active_learning_1d_plot(
                    x_grid,
                    pred_mu,
                    pred_sigma,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    self.ground_truth_available,
                    self.gt_X,
                    self.gt_function_values,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
            else:
                active_learning_1d_plot(
                    x_grid,
                    pred_mu,
                    pred_sigma,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    save_plot=self.save_plots,
                    file_name=plot_name,
                    file_path=self.plot_path,
                )
        else:
            if self.ground_truth_available:
                active_learning_1d_plot(
                    x_grid,
                    pred_mu,
                    pred_sigma,
                    self.x_data,
                    self.y_data,
                    query,
                    new_y,
                    self.ground_truth_available,
                    self.gt_X,
                    self.gt_function_values,
                )
            else:
                active_learning_1d_plot(x_grid, pred_mu, pred_sigma, self.x_data, self.y_data, query, new_y)
                # pred_mu,pred_cov = self.model.predictive_dist(x_grid)


if __name__ == "__main__":
    pass
