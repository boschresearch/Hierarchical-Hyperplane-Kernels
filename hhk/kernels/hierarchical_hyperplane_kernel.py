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
from gpflow.utilities import positive
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
import matplotlib.pyplot as plt
import copy
from hhk.configs.kernels.hhk_tree_config_matrices import (
    HHK_EIGHT_LEFT_TREE_MATRIX,
    HHK_EIGHT_RIGHT_TREE_MATRIX,
    HHK_FOUR_RIGHT_TREE_MATRIX,
    HHK_FOUR_LEFT_TREE_MATRIX,
    HHK_TWO_LEFT_TREE_MATRIX,
    HHK_TWO_RIGHT_TREE_MATRIX,
)

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
from gpflow.ci_utils import ci_niter

f64 = gpflow.utilities.to_default_float
from typing import Tuple


class HierarchicalHyperplaneKernel(gpflow.kernels.Kernel):
    """
    Main class of the hierarchical-hyperplane kernel. It is a child class of gpflow.kernels.Kernel.

    Main Attributes:
        topology: int - specyfing the topology of the tree (1==One-level HHK with two hyperplanes - 2== Two-level HHK with four hyperplanes etc)
        kernel_list: list of gpflow.kernels.Kernel objects - these are the kernels in the leaf of the HHK
        base_hyperplane_mu: float - mu of hyperplane prior - also acts as initial parameter for optimization
        base_hyperplane_std: float - std of hyperplane prior
        smoothing_prior_parameters: Tuple[float,float] - parameters of the Gamma prior of the smoothing parameter
    """

    def __init__(
        self,
        base_kernel: gpflow.kernels.Kernel,
        base_hyperplane_mu: float,
        base_hyperplane_std: float,
        input_dimension: int,
        base_smoothing: float,
        hyperplanes_learnable: bool,
        learn_smoothing_parameter: bool,
        topology: int,
        smoothing_prior_parameters: Tuple[float, float],
        **kwargs
    ):
        super().__init__()
        self.topology = topology
        self.set_default_topology(topology)
        self.dimension = input_dimension
        self.learn_smoothing_parameter = learn_smoothing_parameter
        self.smoothing_prior_parameters = smoothing_prior_parameters
        self.hyperplanes_learnable = hyperplanes_learnable
        self.base_hyperplane_mu = base_hyperplane_mu
        self.base_hyperplane_std = base_hyperplane_std
        self.base_smoothing = base_smoothing
        self.kernel_list = []
        for i in range(0, self.n_experts):
            kernel = gpflow.utilities.deepcopy(base_kernel)
            self.kernel_list.append(kernel)
        self.initialize_hyperplane_parameters()

    def initialize_hyperplane_parameters(self):
        """
        Method for initializing gpflow parameters and priors associated with the hyperplanes - parameters are stored in self.hyperplane_parameter_list
        """
        self.smoothing_list = []
        self.hyperplane_parameter_list = []
        for j in range(0, self.M):
            smoothing_param = gpflow.Parameter([self.base_smoothing], transform=positive(), trainable=self.learn_smoothing_parameter)
            a_smoothing, b_smoothing = self.smoothing_prior_parameters
            smoothing_param.prior = tfd.Gamma(f64([a_smoothing]), f64([b_smoothing]))
            self.smoothing_list.append(smoothing_param)
            w = gpflow.Parameter(np.repeat(self.base_hyperplane_mu, self.dimension + 1), trainable=self.hyperplanes_learnable)
            w.prior = tfd.Normal(
                np.repeat(self.base_hyperplane_mu, self.dimension + 1), np.repeat(self.base_hyperplane_std, self.dimension + 1)
            )
            self.hyperplane_parameter_list.append(w)

    def gate(self, x: tf.Tensor):
        """
        Main method for calculation lambda_j(x) based on the tree topology see paper for details
        """
        expert_probabilities = []
        for k in range(0, self.n_experts):
            prob = tf.cast(tf.expand_dims(tf.repeat(1.0, x.shape[0]), axis=1), dtype=tf.float64)
            for j in range(0, self.M):
                w = self.hyperplane_parameter_list[j]
                w_0 = w[0]
                w_rest = tf.expand_dims(w[1:], axis=0)
                smoothing = self.smoothing_list[j][0]
                prob_elem = tf.math.pow(
                    self.sigmoid(w_0 + tf.linalg.matmul(x, tf.transpose(w_rest)), smoothing), self.left_matrix[k, j]
                ) * tf.math.pow(1 - self.sigmoid(w_0 + tf.linalg.matmul(x, tf.transpose(w_rest)), smoothing), self.right_matrix[k, j])
                prob = prob * prob_elem
            expert_probabilities.append(prob)
        return expert_probabilities

    def sigmoid(self, x, smoothing):
        return 1 / (1 + tf.math.exp(-1 * x * smoothing))

    def set_default_topology(self, n_depth):
        """
        Method for setting the tree topology matrices for a given depth of the tree (we use symmetric trees only here)
        Arguments:
            n_depth: int - depth of the symmetric tree (only up to depth 3 is implemented)
        """
        if n_depth == 1:
            left_matrix = HHK_TWO_LEFT_TREE_MATRIX
            right_matrix = HHK_TWO_RIGHT_TREE_MATRIX
            self.set_topology_matrices(left_matrix, right_matrix)
        elif n_depth == 2:
            left_matrix = HHK_FOUR_LEFT_TREE_MATRIX
            right_matrix = HHK_FOUR_RIGHT_TREE_MATRIX
            self.set_topology_matrices(left_matrix, right_matrix)
        elif n_depth == 3:
            left_matrix = HHK_EIGHT_LEFT_TREE_MATRIX
            right_matrix = HHK_EIGHT_RIGHT_TREE_MATRIX
            self.set_topology_matrices(left_matrix, right_matrix)

    def set_topology_matrices(self, left_matrix, right_matrix):
        """
        Sets the tree topology matrices (see paper) and deduces number of experts and number of gates from the matrices

        Arguments:
            left_matrix : np.array - left matrix of HHK
            right_matrix: np.array - right matrix of HHK
        """
        self.left_matrix = left_matrix
        self.right_matrix = right_matrix
        self.n_experts = self.left_matrix.shape[0]  ## Number of experts
        self.M = self.left_matrix.shape[1]  ## Number of gate nodes
        assert self.n_experts == self.right_matrix.shape[0]
        assert (self.left_matrix.shape[0] * self.left_matrix.shape[1]) == ((self.left_matrix == 0).sum() + (self.left_matrix == 1).sum())
        assert (self.right_matrix.shape[0] * self.right_matrix.shape[1]) == (
            (self.right_matrix == 0).sum() + (self.right_matrix == 1).sum()
        )

    def get_topology(self):
        return self.topology

    def K(self, X, X2=None):
        """
        Main forward pass through the kernel K(X,X2) - admits to gpflow.kernel.Kernel parent class
        """
        len_x = X.shape[0]
        if X2 is None:
            X2 = X
            len_x2 = len_x
        else:
            len_x2 = X2.shape[0]
        gate_x = self.gate(X)
        gate_x2 = self.gate(X2)
        output = tf.zeros((len_x, len_x2), dtype=tf.dtypes.float64)
        for k in range(0, self.n_experts):
            output += tf.matmul(gate_x[k], tf.transpose(gate_x2[k])) * self.kernel_list[k].K(X, X2)
        return output

    def K_diag(self, X):
        """
        Forward pass through the kernel to get diagonal of gram matrix - admits to gpflow.kernel.Kernel parent class
        """
        len_x = X.shape[0]
        gate_x = self.gate(X)
        output = tf.zeros((len_x,), dtype=tf.dtypes.float64)
        gate_x = self.gate(X)
        for k in range(0, self.n_experts):

            output += tf.math.multiply(tf.squeeze(tf.math.pow(gate_x[k], 2.0)), self.kernel_list[k].K_diag(X))
        return output


def get_num_active_partitions(expert_probabilities: np.array, threshold: float):
    num_active_partitions = 0
    for expert_prob in expert_probabilities:
        if np.any(np.greater(expert_prob, threshold)):
            num_active_partitions += 1
    return num_active_partitions
