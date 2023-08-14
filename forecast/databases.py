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
from forecast.data_reader import XArrayReader

# Define source databases
@dataclass
class dbase_short:
    name: str # Short name of the database
    path: str # (Long) path of the database
    description: str # User-friendly description
    reader: ... # Class providing a readdate method

dbase = [dbase_short(name='wb_era5_025_37',
                     path='gs://weatherbench2/datasets/era5/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2',
                     description='ERA5/WB2, 1/4 degree, 37 levels, hourly, 1959-2021',
                     reader = XArrayReader),
        dbase_short(name='wb_era5_025_13',
                    path='gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr',
                    description='ERA5/WB2, 1/4 degree, 13 levels, 6hr, 1959-2021',
                    reader = XArrayReader),
        dbase_short(name='wb_era5_100_13',
                    path='gs://weatherbench2/datasets/era5/1959-2022-6h-512x256_equiangular_conservative.zarr',
                    description='ERA5/WB2, 0.7-degree (512x256), 13 levels, 6hr, 1959-2021',
                    reader = XArrayReader),
        dbase_short(name='none',path=None,description='No database (use cache only)',reader=XArrayReader),
        dbase_short(name='xarr',path=None,description='Use dbase_path to specify database (xarr)', reader=XArrayReader)
        ]

try:
    from forecast.fstd_reader import FSTDReader

    dbase.append(dbase_short(name='fstd',path=None,description='Use dbase_path to specify database (fstd)', reader=FSTDReader))
except ImportError:
    print('Warning: FSTD libraries not available, disabling FSTD database import')

dbase_dict = {d.name : d for d in dbase}
dbase_descriptions = '\n'.join([f'   {d.name}: {d.description}' for d in dbase])
