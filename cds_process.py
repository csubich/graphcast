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


import argparse
from pathlib import Path
import xarray as xr
from forecast import encabulator
import numpy as np
import pickle
import multiprocessing

import datetime
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--base-dir',type=str,dest='base_dir',required=True,default=None)
parser.add_argument('--cache-dir',type=str,dest='cache_dir',required=True,default=None)
parser.add_argument('year',type=int)

args = parser.parse_args()

year = args.year
base_dir = args.base_dir
cache_dir = args.cache_dir


base_path=base_dir

print('Opening surface datasets')
surf_ds = xr.open_mfdataset(sorted(list(glob.glob(f'{base_path}/*/*/era5_2d.nc'))),
                            coords='minimal', data_vars='minimal', compat='override')

print('Opening precipitation datasets')
precip_ds = xr.open_mfdataset(sorted(list(glob.glob(f'{base_path}/*/*/era5_precip.nc'))),
                            coords='minimal', data_vars='minimal', compat='override')



renames_3d = {'u' : 'u_component_of_wind',
           'v' : 'v_component_of_wind',
           'w' : 'vertical_velocity',
           'z' : 'geopotential',
           't' : 'temperature',
           'q' : 'specific_humidity',
           'isobaricInhPa' : 'level'}

renames_2d = {'u10' : '10m_u_component_of_wind',
              'v10' : '10m_v_component_of_wind',
              't2m' : '2m_temperature',
              'z' : 'geopotential_at_surface',
              'lsm' : 'land_sea_mask',
              'msl' : 'mean_sea_level_pressure',
              'tisr' : 'toa_incident_solar_radiation'}

renames_precip = {'tp' : 'total_precipitation_6hr'}

no_time_vars = {'geopotential_at_surface', 'land_sea_mask'}

# def load_ds(name):
#     res = xr.load_dataset(name)
#     return (pickle.dumps(res,-1))

def compute_set(ds): return pickle.dumps(ds.compute(),-1)

import dask
from multiprocessing.pool import ThreadPool
process_pool = multiprocessing.Pool(20)
dask.config.set(pool=ThreadPool(20))


for month in range(1,13):
    print(f'Processing month {month}')

    date_start = datetime.datetime(year,month,1,0)
    dt = datetime.timedelta(hours=6)
    date_end = datetime.datetime(year + (1 if month == 12 else 0),(month + 1 if month < 12 else 1),1,0)-dt

    print(f'Opening 3D datasets for month {month}')
    
    # Everything GRIB has terrible parallel performance via dask; my hunch is that something
    # is not releasing Python's global interpreter lock.  To work around this, we'll use
    # a process pool via multiprocessing.Pool, explode each file into a (file,var) set, then
    # dispatch each tuple to a process in the pool to be read from disk separately.

    # The only real annoyance is that we have to use some kind of serialization (pickling) to
    # return the fully-realized datasets back from the processes, and the unpickling step
    # is inherently serial since pickle does not play nicely with threading.

    files_3d = sorted(list(glob.glob(f'{base_path}/*/{month:02d}/era5_3d*.grib')))
    sets_3d = [xr.open_dataset(f) for f in files_3d]
    vars_3d = sum([[s[v] for v in s.data_vars] for s in sets_3d],[])
    # with multiprocessing.Pool(20) as p:
    levels_ds = xr.combine_by_coords([pickle.loads(v) for v in process_pool.map(compute_set,vars_3d)],
                                        coords='minimal',data_vars='minimal')

    levels_sel = levels_ds.sel(time=slice(date_start,date_end))
    del levels_ds
    levels_sel = levels_sel.isel(latitude=slice(-1,None,-1))
    levels_sel = levels_sel.rename(renames_3d)
    levels_sel = levels_sel.drop_vars(('number','step','valid_time'))
    levels_sel['level'] = levels_sel.level.astype(np.uint64)
    print(f'Found {levels_sel.time.size=}')

    surf_sel = surf_ds.sel(time=slice(date_start,date_end))
    surf_sel = surf_sel.isel(latitude=slice(-1,None,-1))
    surf_sel = surf_sel.rename(renames_2d)
    print(f'Found {surf_sel.time.size=}')
    for var in no_time_vars:
        surf_sel[var] = surf_sel[var][0,:,:]

    # Grab precipitation from T0-5h to Tend, since it must be accumulated over 6h
    dt_hr = datetime.timedelta(hours=1)
    precip_sel = precip_ds.sel(time=slice(date_start-5*dt_hr,date_end))
    precip_sel = precip_sel.isel(latitude=slice(-1,None,-1))
    print(f'Found {precip_sel.time.size=}')

    # Perform grouping and summation, then replace the binned 'time_bins' variable
    # with plain 'time'
    precip_sel = precip_sel.groupby_bins('time',precip_sel.time.size//6).sum()
    precip_sel = precip_sel.rename_dims(time_bins = 'time')
    precip_sel = precip_sel.drop_vars('time_bins')
    precip_sel = precip_sel.rename(renames_precip)

    month_ds = xr.merge([precip_sel,surf_sel,levels_sel])

    # Re-chunk the dataset to limit time chunk-size to one
    for var in month_ds.data_vars:
        if 'chunks' in month_ds[var].encoding:
            del month_ds[var].encoding['chunks']
    month_ds = month_ds.chunk(time=1,level=-1,latitude=-1,longitude=-1)

    compressed_vars = set(month_ds.data_vars) - {'geopotential_at_surface','land_sea_mask','total_precipitation_6hr'}
    encoding = {v : {'compressor' : encabulator.LayerQuantizer(nbits=16)} for v in compressed_vars}

    out_fname= f'{cache_dir}/{year}/{month:02d}'
    print(f'Writing to {out_fname=}')
    month_ds.to_zarr(out_fname,encoding=encoding,compute=True)
    del(month_ds,levels_sel,surf_sel,precip_sel)
