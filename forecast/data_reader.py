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


# Functions to read data into an XArray
import numpy as np
import xarray as xr
import xesmf as xe
import datetime
from forecast.forecast_variables import DERIVED_VARIABLES
from forecast.cache_manager import CacheManager

from typing import Union

import threading
db_sel_lock = threading.Lock()

def model_latlon(model_config):
    '''Construct a lat/lon grid (1d coordinates) based on a specified model_config (resolution)'''
    latitude = np.linspace(-90,90,1+int(180/model_config.resolution),dtype=np.float32)
    longitude = np.linspace(0,360-model_config.resolution,int(360/model_config.resolution),dtype=np.float32)
    return(latitude,longitude)

# def input_from_xarray(intime,task_config,latitude,longitude,in_dbase,interpolator=None):

def update_from_xarray(input_dataset,intime,task_config,latitude,longitude,in_dbase,interpolator=None,input_variables = None):
    '''Select a single input time from a given xarray, '''
    '''Select a single input time from a given xarray, then process it to provide the
    fields required for GraphCast input or validation.  Gives output 'time' as absolute 
    rather than relative time (which would be 0).  Variables in DERIVED_VARIABLES will be
    retained if present but not otherwise added.'''
    if (input_variables is None):
        input_variables = set(task_config['input_variables']) # things required as input to GraphCast
    # Remove any that already exist in the input dataset
    input_variables = set.difference(input_variables,set(input_dataset.data_vars))

    analysis_time = np.datetime64(intime).astype('datetime64[ns]')
    # print('Making input fields for',analysis_time)

    input_variables_in_dbase = list(set.intersection(input_variables,set(in_dbase.data_vars)))
    # print('input_variables_in_dbase', input_variables_in_dbase)
    # Select which input variables have a time dimension
    input_time_variables_in_dbase = [v for v in input_variables_in_dbase if 'time' in in_dbase[v].dims]
    # print('input_time_variables_in_dbase', input_time_variables_in_dbase)
    # ... and which don't
    input_notime_variables_in_dbase = [v for v in input_variables_in_dbase if 'time' not in in_dbase[v].dims]
    # print('input_notime_variables_in_dbase', input_notime_variables_in_dbase)
    # Select variables that aren't in the database at all
    input_variables_not_in_dbase = list(set.difference(input_variables,set(in_dbase.data_vars)))
    
    # ## Construct input dataset
    # input_dataset = xr.Dataset(coords={'time':[analysis_time],
    #                                    'level':list(task_config.pressure_levels),
    #                                    'latitude':latitude,
    #                                    'longitude':longitude}) 

    # If not interpolating, check whether lat/lon values are in the supplied dataset
    if (interpolator is None):
        if ((not input_dataset.latitude.isin(in_dbase.latitude).all()) or \
            (not input_dataset.longitude.isin(in_dbase.longitude).all())):
            raise(ValueError('Target coordinates are not in input database, and no interpolator is supplied'))

    # Check whether all levels are in the database
    if (not input_dataset.level.isin(in_dbase.level).all()):
        raise(ValueError('Not all target levels are in the input database'))

    # Construct level-limited database set
    db_sel_levels = in_dbase.isel(level=in_dbase.level.isin(task_config.pressure_levels))
        
    # Select static (not time-varying) input variables that are in the database)
    if (len(input_notime_variables_in_dbase) > 0):
        if (interpolator is None):
            input_dataset[input_notime_variables_in_dbase] = db_sel_levels[input_notime_variables_in_dbase]
        else:
            input_dataset[input_notime_variables_in_dbase] = interpolator(db_sel_levels[input_notime_variables_in_dbase])
    
    # Limit to analysis times
    if ('time' not in db_sel_levels.dims or not (db_sel_levels.time == analysis_time).any()):
        # The analysis time is not in the 
        return input_dataset
        # raise(ValueError(f'Analysis time {analysis_time} is not in database'))
    
    # When run with multiple threads, xarray seems to have a race condition related to
    # selecting the correct time from the dataset.  The most frequent manifestation is an
    # internal error when building a tuple, but one experiment gave a Pandas exception related
    # to a non-unique selection here.  https://github.com/pandas-dev/pandas/issues/21150 suggests
    # that pandas is internally not thread-safe even for some reading operations, so putting this
    # selection behind a threading lock might do the trick.
    db_sel_lock.acquire()
    db_sel_atimes = db_sel_levels.sel(time=[analysis_time]) # Keep time dimension
    db_sel_lock.release()
    
    # ## Add day/year progress
    # add_progress(input_dataset)

    ## Select time-varying input variables that are in the database
    if (len(input_time_variables_in_dbase) > 0):
        if (interpolator is None):
            input_dataset[input_time_variables_in_dbase] = db_sel_atimes[input_time_variables_in_dbase]
        else:
            input_dataset[input_time_variables_in_dbase] = interpolator(db_sel_atimes[input_time_variables_in_dbase])

    # Detect any variables that were missing from the input
    present_input_variables = set(input_dataset.data_vars).union(set(DERIVED_VARIABLES)) # Values we have or can get
    missing_input_variables = list(set.difference(input_variables,present_input_variables))

    if (len(missing_input_variables) > 0):
        # print('Missing input values: ',missing_input_variables)
        if ('total_precipitation_6hr' in missing_input_variables):
            # print('Calculating 6h precipitation')
            # Calculate 6h precipitation (presumably from a 1h dataset) by adding total_precipitation values
            # input_dataset['total_precipitation_6hr'] = xr.DataArray(np.zeros((1,
            #                                                                  latitude.size,
            #                                                                  longitude.size),dtype=np.float32),
            #                                                         dims=['time','latitude','longitude'])
            newt = analysis_time - np.arange(6)*np.timedelta64(1,'h')
            precip_1hr = db_sel_levels.isel(time=db_sel_levels.time.isin(newt))['total_precipitation']
            if (precip_1hr.time.size != 6):
                pass
                # raise(ValueError(f'Could not get 6 hours of precipitation from database to construct total_precipitation_6hr for date {analysis_time}'))
            else:
                # Accumulate precipitation on the source grid
                precip = precip_1hr.sum(dim='time')
                precip = precip.expand_dims('time')

                # Assign or interpolate accumulated precipitation to target grid
                if (interpolator is None):
                    input_dataset['total_precipitation_6hr'] = precip
                else:
                    input_dataset['total_precipitation_6hr'] = interpolator(precip)
        # if ('toa_incident_solar_radiation' in missing_input_variables):
        #     print('Calculating TOA radiation')
        #     intput_dataset['toa_incident_solar_radiation'] = xr.DataArray(toa_radiation(input_dataset),dims=['time','latitude','longitude'])

    # Re-check missing variables
    # present_input_variables = set(input_dataset.data_vars).union(set(DERIVED_VARIABLES)) # Values we have or can get
    # missing_input_variables = list(set.difference(set(input_variables),present_input_variables))
    # if (len(missing_input_variables) > 0):
    #     raise(ValueError(f'Error: required input variables {missing_input_variables} are missing from the database'))

    return(input_dataset)

class XArrayReader():
    '''Class to wrap the XArray reader function, presenting a uniform readdate(date) interface'''
    def __init__(self,dbase_path: Union[str,None], cache: Union[CacheManager,None], task_config, model_config, dbase_type='zarr',cache_type='netcdf_dir',verbose=False):
        self.dbase_path = dbase_path # URL/path of main database (might be None)
        self.cache = cache
        self.task_config = task_config
        self.model_config = model_config
        self.dbase_type = dbase_type # Type of storage used for main database (default zarr)
        self.cache_type = cache_type # Type of storage used for cache (default netcdf_dir)
        self.verbose = verbose

        if (dbase_path is not None):
            if (dbase_type == 'zarr'):
                if (self.verbose): print(f'Opening database {dbase_path}')
                # Pass trust_env = True to the GCS storage layer in order to pick up
                # HTTP(S)_PROXY environment variables
                storage_options = {'session_kwargs': {'trust_env':True}}
                # Use storage_options only if we have a remote path, determined by a ':' in
                # the path name
                if ':' in dbase_path:
                    self.dbase = xr.open_zarr(dbase_path,storage_options=storage_options)
                else:
                    self.dbase = xr.open_zarr(dbase_path)
            else:
                raise ValueError(f'Unsupported database type {dbase_type}')
        else:
            self.dbase = None

        # Get lat/lon coordinates required by the model
        (self.latitude, self.longitude) = model_latlon(self.model_config)

        # Check if an interpolator is necessary
        self.regridder = None
        if (self.dbase is not None):
            # Create an interpolator unless lat/lon is exactly the same between
            # the generated (model) coordinates and the input database
            if ( (not self.dbase.latitude.isin(self.latitude).all()) or \
                 (not self.dbase.longitude.isin(self.longitude).all()) or \
                 (not np.in1d(self.latitude,self.dbase.latitude.data).all()) or \
                 (not np.in1d(self.longitude,self.dbase.longitude.data).all())):
                dbase_res = np.abs((self.dbase.latitude[1] - self.dbase.latitude[0]).data)

                # xESMF takes a schematic output dataset to derive its interpolation weights,
                # so we must create one.  The conservative regridder requires grid-cell corner
                # information, so we must also construct that set.
                lat = self.latitude
                lon = self.longitude
                # The GraphCast grids include poles; treat the polar points as triangles that extend from
                # the pole to (±90 ∓ model_resolution/2) degrees latitude.  All other cell corners lie halfway
                # between grid points
                lat_corner = np.concatenate((lat[:1],0.5*(lat[1:] + lat[:-1]),lat[-1:]))
                lon_corner = np.concatenate((lon - self.model_config.resolution/2, lon[-1:] + self.model_config.resolution/2))
                dset_out = xr.Dataset({'latitude' : ('latitude',self.latitude),
                                       'longitude' : ('longitude',self.longitude),
                                       'lat_b' : (lat_corner),
                                       'lon_b' : (lon_corner)})
                if (self.model_config.resolution < dbase_res):
                    # Higher-resolution target, use bilinear interpolation
                    if (self.verbose): print(f'Computing bilinear interpolator, from {dbase_res} to {self.model_config.resolution}')
                    self.regridder = xe.Regridder(self.dbase,dset_out,method='bilinear',periodic=True)
                else: 
                    # Lower-resolution target, use conservative interpolation
                    # Update: the conservative_normed method avoids problems at the poles; it seems that xesmf doesn't
                    # natively like something about the triangular cells there.
                    if (self.verbose): print(f'Computing conservative interpolator, from {dbase_res} to {self.model_config.resolution}')
                    self.regridder = xe.Regridder(self.dbase,dset_out,method='conservative_normed',periodic=True)

    def readdate(self,indate : Union[datetime.datetime, np.datetime64]):
        # Reads the cache (if extant) and database for an entry corresponding to inndate and builds an
        # XArray dataset at the supplied model resolution, containing all input_data fields that cannot
        # be derived.  Returns (datset, write_handle), the latter being a Dask handle for data output
        # if the cache can be updated.

        # Convert the input time to an numpy datetime64[ns] object
        dtime_ns = np.datetime64(indate).astype('datetime64[ns]')
        out_dataset = xr.Dataset(coords={'time':[dtime_ns],
                                'level':list(self.task_config.pressure_levels),
                                'latitude':self.latitude,
                                'longitude':self.longitude}) 

        if (self.cache is not None and \
            (cache_ds := self.cache.readdate(dtime_ns)) is not None):
            # print(dtime_ns,cache_ds)
            out_dataset = update_from_xarray(out_dataset,dtime_ns,self.task_config,
                                             self.latitude,self.longitude,cache_ds,None)

        # Get the list necessary variables missing from the input dataset
        missing_variables = set.difference(set(self.task_config['input_variables']),set(out_dataset.data_vars))
        missing_variables = set.difference(missing_variables,set(DERIVED_VARIABLES))
        if (len(missing_variables) == 0):
            # We have all required variables, return
            return out_dataset
                
        # Otherwise, we need to read the missing variables from the database

        timestamp = np.datetime64(indate).astype('datetime64[us]').astype(datetime.datetime).strftime('%Y%m%dT%HZ')
        
        if (self.dbase is None):
            raise(ValueError(f'Field {timestamp} is not present in the cache, and no database is supplied'))
        
        if (dtime_ns not in self.dbase.time):
            raise(ValueError(f'Field {timestamp} is not present in the database'))
        
        # Read the date from the database, with delayed processing via dask
        if (self.verbose): print(f'Reading {timestamp} from database')
        out_dataset = update_from_xarray(out_dataset,indate, self.task_config, self.latitude, self.longitude, self.dbase, self.regridder)
            
        # Check again for missing variables
        missing_variables = set.difference(set(self.task_config['input_variables']),set(out_dataset.data_vars))
        missing_variables = set.difference(missing_variables,set(DERIVED_VARIABLES))
        if (len(missing_variables) > 0):
            raise ValueError(f'Variables f{sorted(list(missing_variables))} were not found in the input dataset')

        if (self.cache is not None):
            self.cache.update(out_dataset)

        return(out_dataset)
        





