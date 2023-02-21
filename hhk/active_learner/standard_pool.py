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


class Pool:
    def __init__(self):
        self.with_replacement = False

    def set_data(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x_data = x_data.copy()
        self.y_data = y_data.copy()

    def set_replacement(self, with_replacement: bool):
        self.with_replacement = with_replacement

    def sample_random(self, n: int, seed: int = 0, set_seed: bool = False):
        if set_seed:
            np.random.seed(seed)
        indexes = np.random.choice(list(range(0, self.x_data.shape[0])), n, replace=False)
        x = self.x_data[indexes]
        y = self.y_data[indexes]
        if not self.with_replacement:
            self.x_data = np.delete(self.x_data, (indexes), axis=0)
            self.y_data = np.delete(self.y_data, (indexes), axis=0)
        return x, y

    def possible_queries(self):
        return self.x_data.copy()

    def get_y_data(self):
        return self.y_data.copy()

    def query(self, x: np.ndarray):
        indexes = []
        for index in range(0, self.x_data.shape[0]):
            if np.array_equal(self.x_data[index], x):
                indexes.append(index)
        if len(indexes) > 0:
            random_index = random.choice(indexes)
            y = self.y_data[random_index]
            if not self.with_replacement:
                self.x_data = np.delete(self.x_data, (random_index), axis=0)
                self.y_data = np.delete(self.y_data, (random_index), axis=0)
            return y
        assert False
