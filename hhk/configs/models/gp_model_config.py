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
from hhk.enums.global_model_enums import PredictionQuantity
from hhk.configs.prior_parameters import EXPECTED_OBSERVATION_NOISE


class BasicGPModelConfig(BaseModelConfig):
    kernel_config: BaseKernelConfig
    observation_noise: float = 0.01
    optimize_hps: bool = True
    train_likelihood_variance: bool = True
    pertube_parameters_at_start: bool = True
    pertubation_for_multistart_opt: float = 0.5
    pertubation_at_start: float = 0.1
    perform_multi_start_optimization: bool = True
    n_starts_for_multistart_opt: int = 10
    set_prior_on_observation_noise: bool = False
    expected_observation_noise: float = EXPECTED_OBSERVATION_NOISE
    prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y
    name = "GPModel"


class GPModelWithNoisePriorConfig(BasicGPModelConfig):
    set_prior_on_observation_noise: bool = True
    name = "GPModelWithNoisePrior"


if __name__ == "__main__":
    pass
