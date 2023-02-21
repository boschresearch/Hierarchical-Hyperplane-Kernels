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

import traceback
import numpy as np
from typing import List, Tuple, Optional, Callable
import gpflow
from gpflow.utilities import print_summary, set_trainable
from tensorflow_probability import distributions as tfd
from hhk.utils.gp_paramater_cache import GPParameterCache
from hhk.models.base_model import BaseModel
from scipy.stats import norm
from enum import Enum
from hhk.enums.global_model_enums import PredictionQuantity
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class GPModel(BaseModel):
    """
    Class that implements standard Gaussian process regression with Type-2 ML for kernel hyperparameter infernece. It forms mainly a wrapper around
    the gpflow.models.GPR object

    Attributes:
        kernel: kernel that is used inside the Gaussian process
        model: holds the gpflow.models.GPR instance
        optimize_hps: bool if kernel parameters are trained
        train_likelihood_variance: bool if likelihood variance is trained
        observation_noise: observation noise level - is either set fixed to that value or acts as initial starting value for optimization
        pertube_parameters_at_start: bool if parameters of the kernels should be pertubed before optimization
        perform_multi_start_optimization: bool if multiple initial values should be used for optimization
        set_prior_on_observation_noise: bool if prior should be applied to obvservation noise (Exponential prior with expected value self.observation_noise)
    """

    def __init__(
        self,
        kernel: gpflow.kernels.Kernel,
        observation_noise: float,
        optimize_hps: bool,
        train_likelihood_variance: bool,
        pertube_parameters_at_start=False,
        pertubation_for_multistart_opt: float = 0.5,
        pertubation_at_start: float = 0.1,
        perform_multi_start_optimization=False,
        set_prior_on_observation_noise=False,
        n_starts_for_multistart_opt: int = 5,
        expected_observation_noise: float = 0.1,
        prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y,
        **kwargs,
    ):
        self.kernel = gpflow.utilities.deepcopy(kernel)
        self.kernel_initial_parameter_cache = GPParameterCache()
        self.kernel_initial_parameter_cache.store_parameters_from_model(self.kernel)
        self.observation_noise = observation_noise
        self.model = None
        self.optimize_hps = optimize_hps
        self.train_likelihood_variance = train_likelihood_variance
        self.use_mean_function = False
        self.pertube_parameters_at_start = pertube_parameters_at_start
        self.perform_multi_start_optimization = perform_multi_start_optimization
        self.n_starts_for_multistart_opt = n_starts_for_multistart_opt
        self.pertubation_for_multistart_opt = pertubation_for_multistart_opt
        self.pertubation_at_start = pertubation_at_start
        self.set_prior_on_observation_noise = set_prior_on_observation_noise
        self.expected_observation_noise = expected_observation_noise
        self.prediction_quantity = prediction_quantity
        self.print_summaries = True

    def set_kernel(self, kernel):
        """
        Method to manually set the kernel after initialization of the object

        Arguments:
            kernel gpflow.kernels.Kernel object
        """
        self.kernel = gpflow.utilities.deepcopy(kernel)
        self.kernel_initial_parameter_cache = GPParameterCache()
        self.kernel_initial_parameter_cache.store_parameters_from_model(self.kernel)

    def reset_model(self):
        """
        resets the model to the initial values - kernel parameters and observation noise are reset to initial values - gpflow model is deleted
        """
        if self.model is not None:
            self.kernel_initial_parameter_cache.load_parameters_to_model(self.kernel, 0)
            del self.model

    def set_mean_function(self, constant: float):
        """
        setter function for mean function - sets the use_mean_function flag to True
        @TODO: mean function is not trainable right now - was only used for Safe Active learning

        Arguments:
            constant: constant value for mean function
        """
        self.use_mean_function = True
        self.mean_function = gpflow.mean_functions.Constant(c=constant)

    def set_number_of_restarts(self, n_starts: int):
        """
        Setter method to set the number of restarts in the multistart optimization - some models need more - default in the class is 10

        Arguments:
            n_starts: number of initial values in the multistart optimization
        """
        self.n_starts_for_multistart_opt = n_starts

    def infer(self, x_data: np.array, y_data: np.array):
        """
        Main entrance method for learning the model - training methods are called if hp should be done otherwise only gpflow model is build

        Arguments:
            x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: Label array with shape (n,1) where n is the number of training points
        """
        self.build_model(x_data, y_data)
        if self.optimize_hps:
            if self.perform_multi_start_optimization:
                self.multi_start_optimization(self.n_starts_for_multistart_opt, self.pertubation_for_multistart_opt)
            else:
                self.optimize(self.pertube_parameters_at_start, self.pertubation_at_start)

    def build_model(self, x_data: np.array, y_data: np.array):
        """
        Method to build to gpflow model object

        Arguments:
            x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: Label array with shape (n,1) where n is the number of training points
        """
        assert len(y_data.shape) == 2

        if self.use_mean_function:
            self.model = gpflow.models.GPR(
                data=(x_data, y_data),
                kernel=self.kernel,
                mean_function=self.mean_function,
                noise_variance=np.power(self.observation_noise, 2.0),
            )
            set_trainable(self.model.mean_function.c, False)
        else:
            self.model = gpflow.models.GPR(
                data=(x_data, y_data),
                kernel=self.kernel,
                mean_function=None,
                noise_variance=np.power(self.observation_noise, 2.0),
            )
        set_trainable(self.model.likelihood.variance, self.train_likelihood_variance)
        if self.set_prior_on_observation_noise:
            self.model.likelihood.variance.prior = tfd.Exponential(1 / np.power(self.expected_observation_noise, 2.0))

    def optimize(self, pertube_initial_parameters: bool, pertubation_factor: float):
        """
        Method for performing Type-2 ML infernence - optimization is repeated if convergence was not succesfull or cholesky was not possible
        pertubation of initial values is applied in this case.
        If kernel parameters have prior this method automatically turns to MAP estimation!!

        Arguments:
            pertube_initial_parameters: bool if the initial parameters should be pertubed at the beginning
            pertubation_factor: factor how much the pertubation of initial parameters should be (also used for pertubtion if convergence was not reached)
        """
        if pertube_initial_parameters:
            self.pertube_parameters(pertubation_factor)
            if self.print_summaries:
                logger.debug("Initial parameters:")
                self.print_model_summary()
        optimizer = gpflow.optimizers.Scipy()
        optimization_success = False
        while not optimization_success:
            try:
                opt_res = optimizer.minimize(self.training_loss, self.model.trainable_variables)
                optimization_success = opt_res.success
            except:
                logger.error("Error in optimization - try again")
                traceback.print_exc()
                self.pertube_parameters(pertubation_factor)
            if not optimization_success:
                logger.warning("Not converged - try again")
                self.pertube_parameters(pertubation_factor)
            else:
                logger.debug("Optimization successful - learned parameters:")
                if self.print_summaries:
                    self.print_model_summary()

    def training_loss(self) -> tf.Tensor:
        return self.model.training_loss()

    def multi_start_optimization(self, n_starts: int, pertubation_factor: float):
        """
        Method for performing optimization (Type-2 ML) of kernel hps with multiple initial values - self.optimzation method is
        called multiple times and the log_posterior_density (falls back to log_marg_likeli for ML) is collected for all initial values.
        Model/kernel is set to the trained parameters with largest log_posterior_density

        Arguments:
            n_start: number of different initialization/restarts
            pertubation_factor: factor how much the initial_values should be pertubed in each restart
        """
        optimization_success = False
        self.parameter_cache = GPParameterCache()
        while not optimization_success:
            self.parameter_cache.clear()
            assert len(self.parameter_cache.parameters_list) == 0
            assert len(self.parameter_cache.loss_list) == 0
            try:
                if self.pertube_parameters_at_start:
                    self.pertube_parameters(self.pertubation_at_start)
                self.multi_start_losses = []
                for i in range(0, n_starts):
                    logger.info(f"Optimization repeat {i+1}/{n_starts}")
                    self.optimize(True, pertubation_factor)
                    loss = self.training_loss()
                    self.parameter_cache.store_parameters_from_model(self.model, loss, add_loss_value=True)
                    self.multi_start_losses.append(loss)
                    log_marginal_likelihood = -1 * loss
                    logger.info(f"Log marginal likeli for run: {log_marginal_likelihood}")
                self.parameter_cache.load_best_parameters_to_model(self.model)
                logger.debug("Chosen parameter values:")
                self.print_model_summary()
                logger.debug(f"Marginal Likelihood of chosen parameters: {self.model.log_posterior_density()}")
                optimization_success = True
            except:
                logger.error("Error in multistart optimization - repeat")

    def print_model_summary(self):
        if logger.isEnabledFor(logging.DEBUG):
            print_summary(self.model)

    def estimate_model_evidence(self, x_data: Optional[np.array] = None, y_data: Optional[np.array] = None) -> np.float:
        """
        Estimates the model evidence - always retrieves marg likelihood, also when HPs are provided with prior!!

        Arguments:
            x_data: (only necessary if infer was not yet called) Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: (only necessary if infer was not yet called) Label array with shape (n,1) where n is the number of training points

        Returns:
            marginal likelihood value of infered model
        """
        if self.model is None and x_data is not None and y_data is not None:
            self.infer(x_data, y_data)
        model_evidence = self.model.log_marginal_likelihood().numpy()
        return model_evidence

    def pertube_parameters(self, factor_bound: float):
        """
        Method for pertubation of the current kernel parameters - internal method that is used before optimization

        Arguments:
            factor_bound: old value is mutliplied with (1+factor) where the factor is random and the factor_bound defines the interval of that variable
        """
        self.kernel_initial_parameter_cache.load_parameters_to_model(self.model.kernel, 0)
        if self.train_likelihood_variance:
            self.model.likelihood.variance.assign(np.power(self.observation_noise, 2.0))
        logger.debug("Pertube parameters - parameters before pertubation")
        self.print_model_summary()
        for variable in self.model.trainable_variables:
            unconstrained_value = variable.numpy()
            factor = 1 + np.random.uniform(-1 * factor_bound, factor_bound, size=unconstrained_value.shape)
            if np.isclose(unconstrained_value, 0.0, rtol=1e-07, atol=1e-09).all():
                new_unconstrained_value = (unconstrained_value + np.random.normal(0, 0.05, size=unconstrained_value.shape)) * factor
            else:
                new_unconstrained_value = unconstrained_value * factor
            variable.assign(new_unconstrained_value)

    def get_number_of_trainable_parameters(self):
        total_number = 0
        for variable in self.model.trainable_variables:
            total_number += tf.size(variable).numpy()
        return total_number

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (n,)
        sigma array with shape (n,)
        """
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            pred_mus, pred_vars = self.model.predict_f(x_test)
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            pred_mus, pred_vars = self.model.predict_y(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        return np.squeeze(pred_mus), np.squeeze(pred_sigmas)

    def predictive_log_likelihood(self, x_test: np.array, y_test: np.array) -> np.array:
        """
        Method for calculating the log likelihood value of the the predictive distribution at the test input points (evaluated at the output values)
        - method is therefore for validation purposes only

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points
        y_test: Array of test output points with shape (n,1)

        Returns:
        array of shape (n,) with log liklihood values
        """
        pred_mus, pred_vars = self.model.predict_y(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        log_likelis = norm.logpdf(np.squeeze(y_test), np.squeeze(pred_mus), np.squeeze(pred_sigmas))
        return log_likelis

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        """
        Method for calculating the entropy of the predictive distribution for test sequence - used for acquistion function in active learning

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        entropy array with shape (n,1)
        """
        if self.prediction_quantity == PredictionQuantity.PREDICT_F:
            pred_mus, pred_vars = self.model.predict_f(x_test)
        elif self.prediction_quantity == PredictionQuantity.PREDICT_Y:
            pred_mus, pred_vars = self.model.predict_y(x_test)
        pred_sigmas = np.sqrt(pred_vars)
        entropies = self.normal_entropy(pred_sigmas)
        return entropies

    def normal_entropy(self, sigma):
        entropy = np.log(sigma * np.sqrt(2 * np.pi * np.exp(1)))
        return entropy

    def deactivate_summary_printing(self):
        self.print_summaries = False
