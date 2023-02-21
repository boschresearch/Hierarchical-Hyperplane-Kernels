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

import logging
import gpflow
from typing import Iterator
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import deep_copy
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from gpflow.utilities import print_summary, set_trainable, to_default_float
from enum import Enum
from scipy.stats import multivariate_normal
from hhk.models.base_model import BaseModel
from hhk.utils.gaussian_mixture_density import EntropyApproximation, GaussianMixtureDensity
from hhk.enums.global_model_enums import InitializationType, PredictionQuantity

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
f64 = gpflow.utilities.to_default_float
logger = logging.getLogger(__name__)


class GPModelMarginalized(BaseModel):
    """
    Class for GP Regression with fully bayesian inference of the kernel hyperparameters using Hamiltonian Monte Carlo. It uses gpflow.models.GPR and the HMC and MCMC
    methods of Tensorflow probability (as suggested in the gpflow notebooks)

    Attributes:
        kernel: an instance of a subclass of gpflow.kernels.Kernel - the kernel parameters need to be equipped with priors!
        model: holds the gpflow.models.GPR instance
        train_likelihood_variance: bool if likelihood variance is trained
        observation_noise: observation noise level - is either set fixed to that value or acts as initial starting value for optimization
        num_samples: number of posterior samples that should be retrieved by HMC
        thin_trace: bool if the HMC trace should be thinned to ensure independence of posterior draws
        thin_steps: (number of samples-1) in between draws that are kept after thinning
        num_burnin_steps: number of samples in the beginning of the chain that are thrown away - to ensure samples from the stationary distribution
        initialization_type: InitializationType Enum that specifies how the starting point of the HMC chain should be generated
        prediction_quantity: PredictionQuantity Emum that specifies if P(y|x,D) or P(f|x,D) should be approximated for prediction
        samples: list of posterior_draws for the unconstrained parameters (called variables in this context - as they are the tensorflow variables)
        parameter_samples: list of posterior draws transformend to the constrained parameters
    """

    def __init__(
        self,
        kernel: gpflow.kernels.Kernel,
        observation_noise: float,
        expected_observation_noise: float,
        train_likelihood_variance: bool,
        num_samples: int = 100,
        num_burnin_steps: int = 500,
        thin_trace: bool = True,
        thin_steps: int = 50,
        initialization_type: InitializationType = InitializationType.PRIOR_DRAW,
        prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y,
        entropy_approx_type: EntropyApproximation = EntropyApproximation.QUADRATURE,
        **kwargs
    ):
        self.observation_noise = observation_noise
        self.expected_observation_noise = expected_observation_noise
        self.model = None
        self.plot_posterior = False
        self.print_posterior_summary = False
        self.thin_trace = thin_trace
        self.num_samples = num_samples
        self.num_burnin_steps = num_burnin_steps
        if self.thin_trace:
            self.thin_steps = thin_steps
        else:
            self.thin_steps = 0
        self.kernel = gpflow.utilities.deepcopy(kernel)
        self.train_likelihood_variance = train_likelihood_variance
        self.initialization_type = initialization_type
        self.prediction_quantity = prediction_quantity
        self.target_acceptance_prob = 0.75
        self.use_mean_function = False
        self.samples = []
        self.parameter_samples = []
        self.entropy_approx_type = entropy_approx_type

    def set_number_of_samples(self, num_samples: int):
        """
        Setter method for the number of posterior samples that should be drawn

        Arguments:
            num_samples: number of posterior samples
        """
        self.num_samples = num_samples

    def set_entropy_approx_type(self, entropy_approx_type: EntropyApproximation):
        """
        Setter method for the type of entropy approximation e.g. quadrature or gaussian moment-matches

        Arguments:
            entropy_approx_type: enum of type EntropyApproximation
        """
        self.entropy_approx_type = entropy_approx_type

    def set_mean_function(self, constant: float):
        """
        Setter method, to set the mean function to a constant

        Arguments:
            constant: mean function constant
        """
        self.use_mean_function = True
        self.mean_function = gpflow.mean_functions.Constant(c=constant)

    def build_model(self, x_data: np.array, y_data: np.array):
        """
        Method that builds the initial gpflow.model.GPR model

        Arguments:
            x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
            y_data: Label array with shape (n,1) where n is the number of training points
        """
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

        if self.train_likelihood_variance:
            self.model.likelihood.variance.prior = tfd.Exponential(1.0 / np.power(self.expected_observation_noise, 2.0))
        else:
            set_trainable(self.model.likelihood.variance, False)

    def initialize_first_sample(self):
        """
        Method to initialize the initial sample for HMC - either MAP Estimate or a draw from the prior
        """
        if self.initialization_type == InitializationType.MAP_ESTIMATE:
            self.optimize_hyperparameters()
        elif self.initialization_type == InitializationType.PRIOR_DRAW:
            self.draw_from_hyperparameter_prior()

    def optimize_hyperparameters(self, draw_initial_from_prior: bool = True):
        """
        Method for performing Type-2 ML infernence - optimization is repeated if convergence was not succesfull or cholesky was not possible
        pertubation of initial values is applied in this case.
        If kernel parameters have prior this method automatically turns to MAP estimation!!

        Arguments:
            draw_initial_from_prior: bool if the initial parameters should be drawn from prior at the beginning
        """
        logger.info("-Start initial optimization")
        if draw_initial_from_prior:
            logger.debug("Initial parameters:")
            self.draw_from_hyperparameter_prior()
            # print_summary(self.model)
        optimizer = gpflow.optimizers.Scipy()
        optimization_success = False
        while not optimization_success:
            try:
                opt_res = optimizer.minimize(self.training_loss, self.model.trainable_variables)
                optimization_success = opt_res.success
            except:
                logger.error("Error in optimization - try again")
                self.draw_from_hyperparameter_prior()
            if not optimization_success:
                logger.warning("Not converged - try again")
                self.draw_from_hyperparameter_prior()
            else:
                logger.debug("Optimization succesful - learned parameters:")
                self.print_model_summary()
        logger.info("-Optimization done")

    def draw_from_hyperparameter_prior(self, show_draw: bool = True):
        """
        Method for drawing a sample from the prior and setting the kernel parameters to this sample.

        Arguments:
            show_draw: bool if the drawn parameters should be printed
        """
        logger.info("-Draw from hyperparameter prior")
        for parameter in self.model.trainable_parameters:
            new_value = parameter.prior.sample()
            parameter.assign(new_value)
        if show_draw:
            self.print_model_summary()

    def reset_model(self):
        pass

    def infer(self, x_data: np.array, y_data: np.array):
        """
        Main training method. First builds a gpflow.model.GPR model and initializes the HP's (either to MAP or prior draw). It then start the HMC procedure
        and collects self.num_samples posterior draws of the HP's given the data and stores the samples in self.parameter_samples

        Arguments:
        x_data: Input array with shape (n,d) where d is the input dimension and n the number of training points
        y_data: Label array with shape (n,1) where n is the number of training points
        """
        logger.info("-Infer")
        self.build_model(x_data, y_data)
        self.hmc_helper = gpflow.optimizers.SamplingHelper(self.log_posterior_density, self.model.trainable_parameters)
        hmc = tfp.mcmc.HamiltonianMonteCarlo(self.hmc_helper.target_log_prob_fn, num_leapfrog_steps=10, step_size=0.01)
        adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            hmc, num_adaptation_steps=10, target_accept_prob=f64(self.target_acceptance_prob), adaptation_rate=0.1
        )
        # @TODO: test if this function declaration is guilty for memory leaks
        @tf.function
        def run_chain_fn():
            return tfp.mcmc.sample_chain(
                num_results=self.num_samples,
                num_burnin_steps=self.num_burnin_steps,
                current_state=self.hmc_helper.current_state,
                num_steps_between_results=self.thin_steps,
                kernel=adaptive_hmc,
                trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            )

        inference_success = False
        while not inference_success:
            self.initialize_first_sample()
            try:
                logger.info("-Start sampling")
                self.samples, traces = run_chain_fn()
                inference_success = True
            except:
                logger.error("ERROR in MCMC sampling - repeat")
        self.parameter_samples = self.hmc_helper.convert_to_constrained_values(self.samples)
        self.param_to_name = {param: name for name, param in gpflow.utilities.parameter_dict(self.model).items()}
        logger.info("-Inference done")
        if self.print_posterior_summary:
            self.print_parameter_sample_summary()
        if self.plot_posterior:
            self.plot_sample_trace()

    def log_posterior_density(self) -> tf.Tensor:
        return self.model.log_posterior_density()

    def training_loss(self) -> tf.Tensor:
        return -1 * self.log_posterior_density()

    def print_model_summary(self):
        if logger.isEnabledFor(logging.DEBUG):
            print_summary(self.model)

    def plot_posterior_hp_histograms(self):
        """
        @TODO: Method can only do that for one dimensional parameters -> change
        Plot historgram of all sampled parameters
        """
        fig, ax = plt.subplots(1, len(self.param_to_name), figsize=(15, 3), constrained_layout=True)
        for axes, sample, parameter in zip(ax, self.parameter_samples, self.model.trainable_parameters):
            axes.hist(sample)
            axes.set_title(self.param_to_name[parameter])
        plt.show()

    def print_parameter_sample_summary(self):
        """
        @TODO: Method can only do that for one dimensional parameters -> change
        Prints mean and std of the drawn parameter samples
        """
        for sample, parameter in zip(self.parameter_samples, self.model.trainable_parameters):
            logger.debug(self.param_to_name[parameter])
            logger.debug(sample.shape)
            logger.debug("Posterior Mean:")
            logger.debug(np.mean(sample, axis=0))
            logger.debug("Posterior Std:")
            logger.debug(np.std(sample, axis=0))

    def plot_sample_trace(self):
        """
        @TODO: Method can only do that for one dimensional parameters -> change
        Shows trace of parameter samples
        """
        plt.figure(figsize=(10, 4))
        for val, param in zip(self.parameter_samples, self.model.trainable_parameters):
            plt.plot(tf.squeeze(val), label=self.param_to_name[param])
        plt.legend()
        plt.xlabel("HMC iteration")
        plt.show()

    def get_posterior_summary(self):
        """
        @TODO: Method can only do that for one dimensional parameters -> change
        """
        mean_dict = {}
        std_dict = {}
        for sample, parameter in zip(self.parameter_samples, self.model.trainable_parameters):
            param_name = self.param_to_name[parameter]
            mean_dict[param_name] = np.mean(sample, axis=0)
            std_dict[param_name] = np.std(sample, axis=0)
        return mean_dict, std_dict

    def get_kernel_with_posterior_mean_estimates(self) -> gpflow.kernels.Kernel:
        """
        Retrieves a kernel with the mean of the posterior samples as HP's

        Returns:
            gpflow.kernels.Kernel - Kernel with mean posterior parameteres
        """
        for sample, parameter in zip(self.parameter_samples, self.model.trainable_parameters):
            value = np.mean(sample, axis=0)
            parameter.assign(value)
            # std_dict[param_name]=np.std(sample,axis=0)
        return gpflow.utilities.deepcopy(self.model.kernel)

    def yield_posterior_models(self) -> Iterator[gpflow.models.GPR]:
        for i in range(0, self.num_samples):
            for var, var_sample in zip(self.hmc_helper.current_state, self.samples):
                var.assign(var_sample[i])
            yield self.model

    def get_posterior_samples(self) -> Tuple[List[np.array], List[str]]:
        """
        Getter method for the posterior HMC draws of the kernel HP's

        Returns:
        List[np.array] - list of np.arrays containing the posterior samples with shape [n,] or [n,p] where n is the number of posterior samples and p the dimension of the parameter
        List[str] - list with names of the parameters
        """
        param_names = []
        for parameter in self.model.trainable_parameters:
            param_name = self.param_to_name[parameter]
            param_names.append(param_name)
        return self.parameter_samples, param_names

    def estimate_model_evidence(self, x_data: np.array, y_data: np.array, sample_size=1000) -> np.float:
        """
        Sample estimate of the model evidence p(y|M,x)=intergal(p(y|theta,x,M)p(theta|M)d theta) - very bad estimate
        @TODO: provide better estimate - maybe fall back to laplace method

        Arguments:
        x_data - Input array with shape (n,d) where d is the input dimension and n the number of training points
        y_data - Label array with shape (n,1) where n is the number of training points

        Returns:
        evidence value - single value
        """
        ##bad estimate!
        self.build_model(x_data, y_data)
        marginal_likelihoods = []
        for i in range(0, sample_size):
            self.draw_from_hyperparameter_prior(show_draw=False)
            log_marginal_likelihood = self.model.log_marginal_likelihood()
            marginal_likeli = np.exp(log_marginal_likelihood)
            marginal_likelihoods.append(marginal_likeli)
        return np.mean(marginal_likelihoods)

    def predict(self, x_test: np.array, prediction_quantity: PredictionQuantity) -> Tuple[np.array, np.array]:
        """
        Inner method for getting predictive distributions associated with each HMC sample - collected in summarized mean and sigma arrays

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (self.num_samples,n)
        sigma array with shape (self.num_samples,n)
        """
        pred_mus_over_samples = []
        pred_sigmas_over_samples = []
        for i in range(0, self.num_samples):
            for var, var_sample in zip(self.hmc_helper.current_state, self.samples):
                var.assign(var_sample[i])
            if prediction_quantity == PredictionQuantity.PREDICT_F:
                pred_mus, pred_vars = self.model.predict_f(x_test)
            elif prediction_quantity == PredictionQuantity.PREDICT_Y:
                pred_mus, pred_vars = self.model.predict_y(x_test)
            pred_sigmas = np.sqrt(pred_vars)
            pred_mus_over_samples.append(pred_mus)
            pred_sigmas_over_samples.append(pred_sigmas)
        pred_mus_complete = np.array(pred_mus_over_samples)
        pred_sigmas_complete = np.array(pred_sigmas_over_samples)
        return pred_mus_complete, pred_sigmas_complete

    def predictive_dist(self, x_test: np.array) -> Tuple[np.array, np.array]:
        """
        Method for retrieving the predictive mean and sigma for a given array of the test points

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        mean array with shape (n,)
        sigma array with shape (n,)
        """
        logger.info("Predict")
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test, self.prediction_quantity)
        n = x_test.shape[0]
        mus_over_inputs = []
        sigmas_over_inputs = []
        for i in range(0, n):
            mu = np.mean(pred_mus_complete[:, i])
            var = np.mean(np.power(pred_mus_complete[:, i], 2.0) + np.power(pred_sigmas_complete[:, i], 2.0) - np.power(mu, 2.0))
            mus_over_inputs.append(mu)
            sigmas_over_inputs.append(np.sqrt(var))
        return np.array(mus_over_inputs), np.array(sigmas_over_inputs)

    def entropy_predictive_dist(self, x_test: np.array) -> np.array:
        """
        Calculates entropies of the predictive distributions at the test points - is the entropy of a mixture of gaussian - entropy is approximated via quadrature

        Arguments:
        x_test: Array of test input points with shape (n,d) where d is the input dimension and n the number of test points

        Returns:
        np.array with shape (n,) containing the entropies for each test point
        """
        logger.info("Calculate entropy")
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test, self.prediction_quantity)
        n = x_test.shape[0]
        m_posterior_draws = len(pred_mus_complete[:, 0])
        assert m_posterior_draws == self.num_samples
        weights = np.repeat(1 / m_posterior_draws, m_posterior_draws)
        entropies = []
        for i in range(0, n):
            gmm_at_test_point = GaussianMixtureDensity(weights, pred_mus_complete[:, i], pred_sigmas_complete[:, i])
            gmm_at_test_point.set_entropy_approx_type(self.entropy_approx_type)
            entropy = gmm_at_test_point.entropy()
            logger.debug(str(i) + "/" + str(n) + ":")
            logger.debug(entropy)
            entropies.append(entropy)
        return np.array(entropies)

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
        pred_mus_complete, pred_sigmas_complete = self.predict(x_test, PredictionQuantity.PREDICT_Y)
        n = x_test.shape[0]
        m_posterior_draws = len(pred_mus_complete[:, 0])
        assert m_posterior_draws == self.num_samples
        weights = np.repeat(1 / m_posterior_draws, m_posterior_draws)
        log_likelis = []
        for i in range(0, n):
            gmm_at_test_point = GaussianMixtureDensity(weights, pred_mus_complete[:, i], pred_sigmas_complete[:, i])
            log_likeli = gmm_at_test_point.log_likelihood(np.squeeze(y_test[i]))
            log_likelis.append(log_likeli)
        return np.squeeze(np.array(log_likelis))
