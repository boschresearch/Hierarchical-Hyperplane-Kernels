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

from hhk.enums.active_learner_enums import AcquisitionFunctionType, ValidationType
from pydantic import BaseSettings
import json


class BasicActiveLearnerConfig(BaseSettings):
    acquisition_function_type: AcquisitionFunctionType
    validation_type: ValidationType = ValidationType.RMSE
    use_smaller_acquistion_set: bool = False
    smaller_set_size: int = 0


class PredVarActiveLearnerConfig(BasicActiveLearnerConfig):
    acquisition_function_type: AcquisitionFunctionType = AcquisitionFunctionType.PRED_VAR


class PredEntropyActiveLearnerConfig(BasicActiveLearnerConfig):
    acquisition_function_type: AcquisitionFunctionType = AcquisitionFunctionType.PRED_ENTROPY
    use_smaller_acquistion_set: bool = True
    smaller_set_size: int = 200


class RandomActiveLearnerConfig(BasicActiveLearnerConfig):
    acquisition_function_type: AcquisitionFunctionType = AcquisitionFunctionType.RANDOM


if __name__ == "__main__":
    print(type(json.loads(PredEntropyActiveLearnerConfig().json())))
