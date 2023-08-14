# Copyright 2024 Crown in Right of Canada
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
# Define variants of the GraphCast models
@dataclass
class graphcast_model:
    name: str
    path: str
    description: str

models = [graphcast_model(name='era5_025',
                          path='params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz',
                          description='ERA5 model with 37 levels at 1/4-degree, requiring 6h precipitation input'),
          graphcast_model(name='hres_025',
                          path='params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz',
                          description='"HRES" model with 13 levels at 1/4-degree, without precipitation input'),
          graphcast_model(name='era5_100',
                          path='params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz',
                          description='ERA5 model with 13 levels at 1-degree, requiring 6h precipitation input')
        ]
models_dict = {m.name : m for m in models}
model_descriptions = '\n'.join([f'    {m.name}: {m.description}' for m in models])
