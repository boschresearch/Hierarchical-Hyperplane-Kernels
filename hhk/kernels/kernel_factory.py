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

from hhk.configs.kernels.base_kernel_config import BaseKernelConfig
from hhk.configs.kernels.hhk_configs import BasicHHKConfig
from hhk.configs.kernels.rbf_configs import BasicRBFConfig
from hhk.kernels.hierarchical_hyperplane_kernel import HierarchicalHyperplaneKernel
from hhk.kernels.rbf_kernel import RBFKernel


class KernelFactory:
    @staticmethod
    def build(kernel_config: BaseKernelConfig):

        if isinstance(kernel_config, BasicHHKConfig):
            base_kernel_config = kernel_config.base_kernel_config
            base_kernel_config.input_dimension = kernel_config.input_dimension
            base_kernel = KernelFactory.build(base_kernel_config)
            kernel = HierarchicalHyperplaneKernel(base_kernel=base_kernel, **kernel_config.dict())
            return kernel

        elif isinstance(kernel_config, BasicRBFConfig):
            kernel = RBFKernel(**kernel_config.dict())
            return kernel


if __name__ == "__main__":
    pass
