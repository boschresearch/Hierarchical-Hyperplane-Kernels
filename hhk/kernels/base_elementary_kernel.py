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

import gpflow
import numpy as np
import tensorflow as tf


class BaseElementaryKernel(gpflow.kernels.Kernel):
    def __init__(
        self,
        input_dimension: int,
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs,
    ):
        self.input_dimension = input_dimension
        self.active_dimension = active_dimension
        self.active_on_single_dimension = active_on_single_dimension
        if active_on_single_dimension:
            super().__init__(name=name + "_on_" + str(active_dimension))
            self.num_active_dimensions = 1
        else:
            super().__init__(name=name)
            self.num_active_dimensions = input_dimension
        self.kernel = None

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        if self.active_on_single_dimension:
            X = tf.expand_dims(X[:, self.active_dimension], axis=1)
            X2 = tf.expand_dims(X2[:, self.active_dimension], axis=1)
        assert X.shape[1] == self.num_active_dimensions
        return self.kernel.K(X, X2)

    def K_diag(self, X):
        if self.active_on_single_dimension:
            X = tf.expand_dims(X[:, self.active_dimension], axis=1)
        return self.kernel.K_diag(X)

    def get_input_dimension(self):
        return self.input_dimension
