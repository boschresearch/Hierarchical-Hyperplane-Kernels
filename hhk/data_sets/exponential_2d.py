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
import matplotlib.pyplot as plt


class Exponential2D:
    def __init__(self, observation_noise=0.01):
        self.__a = -2
        self.__b = 5
        self.__dimension = 2
        self.observation_noise = observation_noise

    def f(self, x1, x2):
        return x1 * np.exp(-1 * np.power(x1, 2.0) - np.power(x2, 2.0)) * 0.5

    def query(self, x, noisy=True, scale_factor=1.0):
        function_value = self.f(x[0], x[1]) * scale_factor
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def get_random_data(self, n, noisy=True):
        X = np.random.uniform(low=self.__a, high=self.__b, size=(n, self.__dimension))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        return X, np.expand_dims(np.array(function_values), axis=1)

    def get_scaled_random_data(self, n, noisy=True, f_scale_factor=10):
        X = np.random.uniform(low=self.__a, high=self.__b, size=(n, self.__dimension))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy, f_scale_factor)
            function_values.append(function_value)
        X = (X - self.__a) / (self.__b - self.__a)
        return X, np.expand_dims(np.array(function_values), axis=1)

    def get_box_bounds(self):
        return self.__a, self.__b

    def get_dimension(self):
        return self.__dimension

    def plot(self):
        xs, ys = self.get_scaled_random_data(2000, True)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(xs[:, 0], xs[:, 1], ys, marker=".", color="black")
        plt.show()


if __name__ == "__main__":
    function = Exponential2D(0.01)
    function.plot()
