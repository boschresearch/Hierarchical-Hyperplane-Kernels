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

from hhk.configs.models.base_model_config import BaseModelConfig
from hhk.configs.kernels.base_kernel_config import BaseKernelConfig
from hhk.configs.kernels.hhk_configs import HHKEightLocalDefaultConfig
from hhk.models.gp_model_marginalized import InitializationType, PredictionQuantity
from hhk.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE
from hhk.utils.gaussian_mixture_density import EntropyApproximation


class BasicGPModelMarginalizedConfig(BaseModelConfig):
    kernel_config: BaseKernelConfig
    observation_noise: float = 0.01
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    train_likelihood_variance: bool = True
    num_samples: int = 100
    num_burnin_steps: int = 500
    thin_trace: bool = True
    thin_steps: int = 50
    initialization_type: InitializationType = InitializationType.PRIOR_DRAW
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    entropy_approx_type: EntropyApproximation = EntropyApproximation.QUADRATURE
    name = "GPModelMarginalized"


class GPModelMarginalizedConfigMAPInitialized(BasicGPModelMarginalizedConfig):
    initialization_type: InitializationType = InitializationType.MAP_ESTIMATE
    name = "GPModelMarginalizedMAPInitialized"


if __name__ == "__main__":
    pass
