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

from typing import List, Optional
import gpflow
import numpy as np


class GPParameterCache:
    """
    Class to cache the parameters of gpflow models over multiple iterations (e.g. over multiple optimization steps)
    Parameters can be associated with loss values and best parameters values can be reloaded to the model
    """

    def __init__(self):
        self.parameters_list = []
        self.loss_list = []

    def store_parameters_from_model(
        self, model: gpflow.models.BayesianModel, associated_loss_value: Optional[float] = None, add_loss_value=False
    ):
        parameter_values = self.get_parameter_numpy_values(model)
        self.parameters_list.append(parameter_values)
        if add_loss_value:
            self.loss_list.append(associated_loss_value)

    def load_parameters_to_model(self, model: gpflow.models.BayesianModel, index: int):
        parameter_values = self.parameters_list[index]
        self.set_parameters_to_values(model, parameter_values)

    def load_best_parameters_to_model(self, model: gpflow.models.BayesianModel):
        assert len(self.loss_list) > 0
        best_index = np.argmin(np.array(self.loss_list))
        self.load_parameters_to_model(model, best_index)

    def set_parameters_to_values(self, model: gpflow.models.BayesianModel, parameter_values: List[np.array]):
        for i, parameter in enumerate(model.trainable_parameters):
            if hasattr(model, "likelihood") and parameter is model.likelihood.variance:
                if parameter_values[i] < 1.000001e-06:
                    parameter.assign(1.000001e-06)
                else:
                    parameter.assign(parameter_values[i])
            else:
                parameter.assign(parameter_values[i])

    def get_parameter_numpy_values(self, model: gpflow.models.BayesianModel) -> List[np.array]:
        parameter_values = []
        for parameter in model.trainable_parameters:
            parameter_value = parameter.numpy()
            parameter_values.append(parameter_value)
        return parameter_values

    def clear(self):
        self.parameters_list = []
        self.loss_list = []
