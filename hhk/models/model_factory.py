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
from hhk.configs.models.gp_model_config import BasicGPModelConfig
from hhk.configs.models.gp_model_marginalized_config import BasicGPModelMarginalizedConfig
from hhk.models.gp_model import GPModel
from hhk.models.gp_model_marginalized import GPModelMarginalized
from hhk.kernels.kernel_factory import KernelFactory


class ModelFactory:
    @staticmethod
    def build(model_config: BaseModelConfig):
        if isinstance(model_config, BasicGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = GPModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicGPModelMarginalizedConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = GPModelMarginalized(kernel=kernel, **model_config.dict())
            return model


if __name__ == "__main__":
    pass
