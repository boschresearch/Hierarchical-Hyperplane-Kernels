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
from hhk.configs.kernels.hhk_configs import HHKFourLocalDefaultConfig, HHKEightLocalDefaultConfig, HHKTwoLocalDefaultConfig
from hhk.configs.kernels.rbf_configs import RBFWithPriorConfig
from hhk.data_sets.exponential_2d import Exponential2D
from hhk.configs.models.gp_model_config import GPModelWithNoisePriorConfig
from hhk.configs.models.gp_model_marginalized_config import GPModelMarginalizedConfigMAPInitialized
from hhk.models.model_factory import ModelFactory
from hhk.active_learner.active_learner import ActiveLearner
from hhk.configs.active_learner.active_learner_configs import PredEntropyActiveLearnerConfig
import numpy as np
import os

from hhk.utils.gaussian_mixture_density import EntropyApproximation

matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG)

########### Settings - to be set manually #############

USE_HHK = True
DO_PLOTTING = True
USE_FULLY_BAYESIAN_INFERENCE = False
USE_FAST_ENTROPY_APPROX_IN_FULLY_BAYESIAN = True
N_RESTARTS_MAP = 5
N_AL_STEPS = 20
DATA_SEED = 100
N_TEST_DATA = 200
N_INITIAL_DATA = 10
SAVE_SCORES = False
OUTPUT_FOLDER = ""
RUN_NAME = "HHK"
HHK_CONFIG_CLASS = HHKFourLocalDefaultConfig  # change to different config class to use a different HHK version

###############  Set up initial data and pool ####################
data_loader = Exponential2D()

n_dim = data_loader.get_dimension()

# create pool data (we simulate having an oracle via a pool)
x_pool, y_pool = data_loader.get_scaled_random_data(1000)

# Sample test data
x_test, y_test = data_loader.get_scaled_random_data(N_TEST_DATA)

############### Configure model  #################################

if USE_HHK:
    kernel_config = HHK_CONFIG_CLASS(input_dimension=n_dim)
else:
    kernel_config = RBFWithPriorConfig(input_dimension=n_dim)

if USE_FULLY_BAYESIAN_INFERENCE:
    model_config = GPModelMarginalizedConfigMAPInitialized(kernel_config=kernel_config)
    if USE_FAST_ENTROPY_APPROX_IN_FULLY_BAYESIAN:
        model_config.entropy_approx_type = EntropyApproximation.MOMENT_MATCHED_GAUSSIAN
    else:
        model_config.entropy_approx_type = EntropyApproximation.QUADRATURE
else:
    model_config = GPModelWithNoisePriorConfig(kernel_config=kernel_config)
    model_config.n_starts_for_multistart_opt = N_RESTARTS_MAP

model = ModelFactory.build(model_config)

############## Initialize active learner #########################

active_learner = ActiveLearner(**PredEntropyActiveLearnerConfig().dict())

active_learner.set_pool(x_pool, y_pool)

active_learner.set_test_set(x_test, y_test)

active_learner.sample_initial_data(N_INITIAL_DATA, DATA_SEED, True)

active_learner.set_model(model)

active_learner.set_do_plotting(DO_PLOTTING)

validation_scores, chosen_x = active_learner.learn(N_AL_STEPS)

score_type = active_learner.validation_type.get_name()

############# Save or print scores #######################

if SAVE_SCORES:
    file_name = f"{RUN_NAME}_{score_type}.txt"
    np.savetxt(os.path.join(OUTPUT_FOLDER, file_name), validation_scores)
else:
    print(score_type)
    print(validation_scores)
