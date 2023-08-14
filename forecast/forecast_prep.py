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

import numpy as np
import xarray as xr
import datetime
import dask
import dask.diagnostics

from forecast.forecast_variables import ATMO_VARIABLES, FIXED_VARIABLES

def add_progress(dataset):
    from graphcast import data_utils
    timestamps = dataset['time'].data.astype('datetime64[s]').astype('int') # Seconds since epoch
    lon = dataset['longitude'].data
    year_progress = np.array(data_utils.get_year_progress(timestamps))
    year_progress_sin = np.sin(2*np.pi*year_progress).astype(np.float32)
    year_progress_cos = np.cos(2*np.pi*year_progress).astype(np.float32)
    
    day_progress = data_utils.get_day_progress(timestamps,lon)
    day_progress_cos = np.cos(2*np.pi*day_progress).astype(np.float32)
    day_progress_sin = np.sin(2*np.pi*day_progress).astype(np.float32)
    

    dataset['year_progress_sin'] = xr.DataArray(year_progress_sin,dims=['time'])
    dataset['year_progress_cos'] = xr.DataArray(year_progress_cos,dims=['time'])
    dataset['day_progress_sin'] = xr.DataArray(day_progress_sin,dims=['time','longitude'])
    dataset['day_progress_cos'] = xr.DataArray(day_progress_cos,dims=['time','longitude'])

    return(dataset)

def toa_radiation_deprecated(dataset):
    '''GraphCast requires a forcing variable for "one-hour top of atmosphere radiation,"
    and through trial and error this appears to be the accumulated radiation over the
    one hour period just prior to the valid time.  This function uses pysolar to calculate
    the top-of-atmosphere solar angle plus a modified calculation for extraterrestrial irradiance
    to approximate this field within a fraction of a percent.  In particular, ECMWF appears to
    model the solar cycle, while this calculation does not.'''
    from pysolar import solar, numeric, radiation
    
    lat = dataset.coords['latitude'].data
    lon = dataset.coords['longitude'].data
    # Convert NumPy time objects back to datetime objects, used internally by pysolar
    # for calculations
    times = [datetime.datetime.fromtimestamp(t * 1e-9,tz=datetime.timezone.utc) 
                                             for t in dataset['time'].data.astype('int')]
    # print(times)

    # Get coordinate shapes
    if (lat.ndim == 2): # lat/lon specified as point cloud
        rad = np.zeros((len(times),) + lat.shape)
    elif (lat.ndim == 1): # lat/lon specified as 1D arrays
        rad = np.zeros((len(times),) + lat.shape + lon.shape)
        # (lat,lon) = np.meshgrid(lat,lon,indexing='ij')
        lat = lat.reshape((-1,1))
        lon = lon.reshape((1,-1))
        
    Nx = lat.shape[0]
    Ny = lon.shape[1]

    # Initialize radiation array
    rad = np.zeros((dataset.coords['time'].size,Nx,Ny))

    # Integrate prior-hour radiation over num_times substeps
    num_times = 60
    trange = datetime.timedelta(hours=1)
    dt = trange/num_times
    
    # Iterate over each 'time' coordinate in the dataset
    for field_idx in range(dataset.coords['time'].size):
        for tidx in range(num_times+1):
            # print(tidx)
            tnow = times[field_idx] - dt*tidx
            # altitude_deg = solar.get_altitude(lat,lon,tnow)
            topocentric_sun_declination, topocentric_local_hour_angle = \
                    solar.get_topocentric_position(lat, lon, tnow, 0)
            topocentric_elevation_angle = solar.get_topocentric_elevation_angle(lat, 
                                                                                topocentric_sun_declination, 
                                                                                topocentric_local_hour_angle)
            
            # rad_now = 1367 * np.maximum(0,np.sin(topocentric_elevation_angle*np.pi/180))
            # see https://pvpmc.sandia.gov/modeling-steps/1-weather-design-inputs/irradiance-insolation/extraterrestrial-radiation/
            # rad_now = radiation.get_apparent_extraterrestrial_flux(numeric.tm_yday(tnow)) * np.maximum(0,np.sin(topocentric_elevation_angle*np.pi/180))
            day = numeric.tm_yday(tnow)*2*np.pi/365
            rad_now = 1363*(1+0.00011*0 + 0.034221*np.cos(day) + 0.00128*np.sin(day) + \
                            1*0.000719*np.cos(2*day) + 1*0.000077*np.sin(2*day)) * np.maximum(0,np.sin(topocentric_elevation_angle*np.pi/180))
            if (tidx == 0 or tidx == num_times):
                rad[field_idx,:,:] += rad_now*0.5*(dt.seconds + dt.microseconds*1e-6)
            else:
                rad[field_idx,:,:] += rad_now*(dt.seconds + dt.microseconds*1e-6)
    #rad /= num_times
    return(rad)

def make_forcing_target(outtime,task_config,latitude,longitude):
    '''Create forcing and target datasets corresponding to a particular task_config
    and lat/lon grid at a specific output time'''
    from forecast.toa_radiation import toa_radiation
    target_variables = task_config['target_variables'] # things produced by graphcast, at t6
    forcing_variables = task_config['forcing_variables'] # things privded as input, at t6
    forecast_time = np.datetime64(outtime).astype('datetime64[ns]') # Valid time of forecast
    print(f'Making forcing fields for {outtime.strftime("%Y%m%dT%H")}')

    forcing_dataset = xr.Dataset(coords={'time':[forecast_time],
                                         'latitude':latitude,
                                         'longitude':longitude,
                                         'level':list(task_config.pressure_levels)})
    target_dataset = xr.Dataset(coords={'time':[forecast_time],
                                        'latitude':latitude,
                                        'longitude':longitude,
                                        'level':list(task_config.pressure_levels)})

    # Add day/year progress and TOA radiation to forcing
    add_progress(forcing_dataset)
    forcing_dataset['toa_incident_solar_radiation'] = toa_radiation(forcing_dataset)
    # # On occasion, pysolar tries to take an arcsin outside the -1/1 range thanks to roundoff
    # # error, giving nan values near the terminator.  This radiation can and should be replaced with 0.
    # forcing_dataset['toa_incident_solar_radiation'] = forcing_dataset['toa_incident_solar_radiation'].fillna(0)

    # Create NANs for the target variables -- things GraphCast will predict
    nan_4d = np.full((1,
                      len(task_config.pressure_levels),
                      latitude.size,
                      longitude.size),np.nan,dtype=np.float32)
    nan_3d = np.full((1,
                      latitude.size,
                      longitude.size),np.nan,dtype=np.float32)
    for var in target_variables:
        if var in ATMO_VARIABLES:
            target_dataset[var] = xr.DataArray(nan_4d,dims=['time','level','latitude','longitude'])
        else:
            target_dataset[var] = xr.DataArray(nan_3d,dims=['time','latitude','longitude'])

    return(forcing_dataset,target_dataset)

def forecast_setup(idate, data_reader, forecast_length):
    '''Generate (input, forcing, target) fields required to execute a GraphCast forecast,
    given a forecast initialization date, data_reader object, and forecast length (6h steps)'''
    from forecast.toa_radiation import toa_radiation

    task_config = data_reader.task_config
    latitude = data_reader.latitude
    longitude = data_reader.longitude

    dt = datetime.timedelta(hours=6)
    atimes = [idate - dt, idate]
    ftimes = [idate + i*dt for i in range(1,forecast_length+1)]

    analyses = []
    for atime in atimes:
        analysis = data_reader.readdate(atime)
        # Add day progress and radiation fields, if not present
        if ('day_progress_cos' not in analysis.variables):
            add_progress(analysis)
        if ('toa_incident_solar_radiation' not in analysis.variables):
            analysis['toa_incident_solar_radiation'] = toa_radiation(analysis)

        analyses.append(analysis)

    (forcings, targets) = zip(*[make_forcing_target(ftime,task_config,latitude,longitude) for ftime in ftimes])

    # Construct conslidated datasets
    input_dataset = xr.concat(analyses,dim='time')
    forcing_dataset = xr.concat(forcings,dim='time')
    target_dataset = xr.concat(targets,dim='time')

    # Graphcast uses the 'time' dimension for relative time; preserve the original analysis/validity time as
    # a 'datetime' coordinate.
    i_times = input_dataset.time.data
    i_reltime = i_times - np.datetime64(idate)
    input_dataset['time'] = i_reltime
    input_dataset.coords['datetime'] = ('time',i_times)

    f_times = forcing_dataset.time.data
    f_reltime = f_times - np.datetime64(idate)
    forcing_dataset['time'] = f_reltime
    target_dataset['time'] = f_reltime
    forcing_dataset.coords['datetime'] = ('time',f_times)
    target_dataset.coords['datetime'] = ('time',f_times)

    # Prepend a 'batch' dimension and rename 'latitude' and 'longitude' to 'lat' and 'lon'
    renames = {'latitude': 'lat', 'longitude' : 'lon'}
    input_dataset = input_dataset.expand_dims(dim='batch').rename(renames)
    forcing_dataset = forcing_dataset.expand_dims(dim='batch').rename(renames)
    target_dataset = target_dataset.expand_dims(dim='batch').rename(renames)

    # Some input variables (land/sea mask, topography) are invariant
    # and don't change over the batch; remove the dimension from those
    # variables
    for var in input_dataset.variables:
        if var in FIXED_VARIABLES:
            input_dataset[var] = input_dataset[var].isel(batch=0,drop=True).isel(time=0,drop=True)

    # Invoke dask to complete the computation of all the arrays and write out cache entries
    # with dask.diagnostics.ProgressBar():
    (input_dataset, forcing_dataset, target_dataset) = dask.compute(*(input_dataset, forcing_dataset, target_dataset))

    return(input_dataset, forcing_dataset, target_dataset)

