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

from typing import Tuple
from hhk.configs.kernels.base_kernel_config import BaseKernelConfig
from hhk.configs.kernels.rbf_configs import RBFWithPriorConfig
from hhk.configs.prior_parameters import HHK_SMOOTHING_PRIOR_GAMMA


class BasicHHKConfig(BaseKernelConfig):
    base_kernel_config = RBFWithPriorConfig(input_dimension=0)
    base_smoothing: float = 1.0
    smoothing_prior_parameters: Tuple[float, float] = HHK_SMOOTHING_PRIOR_GAMMA
    hyperplanes_learnable: bool = True
    learn_smoothing_parameter: bool = True
    base_hyperplane_mu: float = 0.0
    base_hyperplane_std: float = 1.0
    topology: int
    name: str = "BasicHHK"


class HHKEightLocalDefaultConfig(BasicHHKConfig):
    topology: int = 3
    name: str = "HHKEightLocalDefault"


class HHKFourLocalDefaultConfig(BasicHHKConfig):
    topology: int = 2
    name: str = "HHKFourLocalDefault"


class HHKTwoLocalDefaultConfig(BasicHHKConfig):
    topology: int = 1
    name: str = "HHKTwoLocalDefault"


if __name__ == "__main__":
    # HHKEightLocalDefaultConfig.base_lengthscale = 0.1
    print(isinstance(HHKEightLocalDefaultConfig(input_dimension=2), BasicHHKConfig))
    print(HHKEightLocalDefaultConfig(input_dimension=2).dict())
