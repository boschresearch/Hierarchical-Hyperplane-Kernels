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

from hhk.active_learner.standard_pool import Pool
from typing import List, Tuple
from hhk.models.base_model import BaseModel
import numpy as np


class BasePoolActiveLearner:
    """
    Base class for all pool active learner - holds a pool oject and provides setter and getter for initial data and test data
    Main functionality such as update of the model, the learning procedure or the valudation needs to be implemented by the child classes

    Attributes:
        pool - Pool : Pool object that is populated by data and that can be queried
        validation_metrics - List : list of values of the validation over the queries
        x_data - np.array : input data - to initialize can be sampled from the pool or set manually
        y_data - np.array : output data
        x_test - np.array : test input data - needs to be set manually
        y_test - np.array : test output data

    """

    def __init__(self):
        self.pool: Pool = Pool()
        self.validation_metrics: List = []
        self.ground_truth_available: bool = False
        self.do_plotting: bool = False
        self.save_plots: bool = False
        self.plot_path: str = ""
        self.x_data: np.array
        self.y_data: np.array
        self.x_test: np.array
        self.y_test: np.array

    def set_do_plotting(self, do_plotting: bool):
        """
        Method for specifying if plotting should be done

        Arguments:
            do_plotting - bool : flag if plotting should be done
        """
        self.do_plotting = do_plotting

    def set_ground_truth(self, gt_X: np.ndarray, gt_function_values: np.ndarray):
        self.ground_truth_available = True
        self.gt_X = gt_X
        self.gt_function_values = gt_function_values

    def set_test_set(self, x_test: np.ndarray, y_test: np.ndarray):
        """
        Method for setting the test set
        """
        self.x_test = x_test
        self.y_test = y_test

    def set_pool(self, complete_x_data: np.ndarray, complete_y_data: np.ndarray):
        """
        Method for population the pool with data

        Arguments:
            complete_x_data - np.array : complete input data of the pool
            complete_y_data - np.array : complete output data of the pool
        """
        self.pool.set_data(complete_x_data, complete_y_data)

    def get_data_sets(self):
        """
        Method to get the data set currently present in the active learner
        """
        return self.x_data, self.y_data

    def sample_initial_data(self, n_data, seed=100, set_seed=False):
        """
        Method to sample initial data from the pool - can be seeded for reproducible experiments
        """
        self.x_data, self.y_data = self.pool.sample_random(n_data, seed, set_seed)

    def set_initial_queries_manually(self, x_data: np.ndarray):
        """
        Method for only setting the initial x values manually - those are then queried from the pool
        """
        self.x_data = x_data
        query = self.x_data[0]
        self.y_data = np.array([self.pool.query(query)])
        for i in range(1, self.x_data.shape[0]):
            query = self.x_data[i]
            new_y = self.pool.query(query)
            self.y_data = np.append(self.y_data, [new_y])
        self.y_data = np.expand_dims(self.y_data, axis=1)

    def set_initial_dataset_manually(self, x_data: np.ndarray, y_data: np.ndarray):
        """
        Method for setting the initial dataset manually (x and y data)
        """
        self.x_data = x_data.copy()
        self.y_data = y_data.copy()

    def save_plots_to_path(self, plot_path: str):
        """
        method to specify that plots are not shown but saved to a path
        """
        self.save_plots = True
        self.plot_path = plot_path

    def learn(self, n_steps: int, **kwargs) -> Tuple[np.ndarray, ...]:
        """
        Main method that needs to be implemented by child classes
        Gets as input the number of active learning steps and returns validation_metrics and the collected queries
        Arguments:
            n_steps - int : number of active learning iterations

        Returns:
            np.array : validation_metrics collected over the iterations
            np.array : queried datapoints (only x values)
            ... some active learner specfic returns
        """
        raise NotImplementedError
