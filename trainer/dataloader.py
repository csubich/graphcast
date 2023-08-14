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


debug_prints = False
import datetime

import threading
compute_lock = threading.Lock()

# Cache to split up a database into dask-manageable chunks on the fly.  The structure here is:
# {dbase_key(dbase), {bucket1 : optimized_subset1, bucket2 : optimized_subset2, etc}}
# Use collections.defaultdict to set a default value of an empty dictionary
import collections
dbase_cache = collections.defaultdict(dict)

def dbase_key(dbase):
    '''XArrays are mutable and unhashable, but we don't really mutate the databases after
    loading them.  Thus, define dbase_key to create a reasonably unique key to represent a
    potentially multi-file database, based on the source filename, key dimension sizes,
    and starting time'''
    return ((dbase.encoding['source'], # source filename
             dbase.time.size, 
             dbase.latitude.size,
             dbase.longitude.size,
             dbase.time[0].data.item()))

def dbase_load(dbase,input_variables,time_idx,level_idx,lat_idx=None,lon_idx=None):
    # Load a specified set of input_variables from a database, selecting a scalar time_idx
    # and subsetting by level_idx, lat_idx, and lon_idx (if latter two are provided and
    # not None).  This uses the global dbase_cache, and lazily creates dask-optimized
    # database subsets as necessary
    import dask
    import xarray as xr

    global dbase_cache

    # Find out what time bucket time_idx lies in
    TIME_BUCKET_SIZE = 1000
    time_bucket = time_idx // TIME_BUCKET_SIZE

    # Get the set of database subsets from the cache corresponding to the provided
    # database.  The result is a mutable dictionary, so changes (additions) to
    # dbase_subsets should be immediately reflected inside dbase_cache
    dbase_subsets = dbase_cache[dbase_key(dbase)]

    # Search for the proper bucket inside dbase_subsets.
    if (time_bucket not in dbase_subsets):
        # If not present, slice dbase along the time dimension
        subset = dbase.isel(time=slice(TIME_BUCKET_SIZE*time_bucket,TIME_BUCKET_SIZE*(time_bucket+1)))
        # dask.optimize, to keep the data unloaded but remove references ot out-of-scope times
        subset = dask.optimize(subset)[0]
        # Write the now-optimized subset back to the cache
        dbase_subsets[time_bucket] = subset
    else:
        # Just load the subset from the cache
        subset = dbase_subsets[time_bucket]

    # Get time_idx with respect to the beginning of this subset
    subset_time_idx = time_idx % TIME_BUCKET_SIZE

    # Check to see if we have conforming latitude/longitudes, using all indices in the dataset in-order
    if (lat_idx is None): assert(lon_idx is None)
    conforming_latlon = (lat_idx is None)

    # Load input_variables from the selected subset

    # Create a skeletal output dataset
    if (conforming_latlon):
        inputs = xr.Dataset(coords={'time':subset.time[[subset_time_idx]].data,
                            'level':subset.level[level_idx],
                            'latitude':subset.latitude,
                            'longitude':subset.longitude}) 
    else:
        inputs = xr.Dataset(coords={'time':subset.time[[subset_time_idx]].data,
                            'level':subset.level[level_idx],
                            'latitude':subset.latitude[lat_idx],
                            'longitude':subset.longitude[lon_idx]}) 

    # Assign loaded variables to the input dataset
    for var in input_variables:
        # print(var)
        if 'time' in subset[var].dims:
            if 'level' in subset[var].dims:
                # XArray "doesn't support yet fancy nd indexing", so we need to slice the data one dimension at a time
                if (conforming_latlon == True):
                    inputs[var] = (subset[var].dims, subset[var].data[[subset_time_idx],...][:,level_idx,:,:])
                else:
                    inputs[var] = (subset[var].dims, subset[var].data[[subset_time_idx],...][:,level_idx,...][...,lat_idx,:][...,lon_idx])
            else:
                if (conforming_latlon == True):
                    inputs[var] = (subset[var].dims, subset[var].data[[subset_time_idx],...])
                else:
                    inputs[var] = (subset[var].dims, subset[var].data[[subset_time_idx],...][...,lat_idx,:][...,lon_idx])
        else:
            # No support yet for static 3D variables
            assert ('level' not in subset[var].dims)
            if (conforming_latlon == True):
                inputs[var] = subset[var]
            else:
                inputs[var] = (subset[var].dims, subset[var].data[...,lat_idx,:][...,lon_idx])
                
    return inputs
    
    

# Flags to mark whether we've warned for noncomforming lat/lon coordinates
warned_v_conforming_latlon = False
warned_a_conforming_latlon = False

def build_forecast(idate,forecast_length,task_config,
                   model_latitude, model_longitude, input_variables, target_variables,
                   a_dbase, v_dbase, do_compute = False):
    # from forecast import 
    import forecast.forecast_variables
    from forecast.toa_radiation import toa_radiation
    from forecast import forecast_prep
    from forecast import encabulator
    import datetime
    import numpy as np
    import xarray as xr
    import dask
    import warnings
    dt = datetime.timedelta(hours=6)

    global warned_a_conforming_latlon
    global warned_v_conforming_latlon

    # Work on copies of the databases
    # compute_lock.acquire()
    # a_dbase = a_dbase.copy()
    # v_dbase = v_dbase.copy()
    # compute_lock.release()

    atimes = [idate - dt, idate]
    ftimes = [idate + i*dt for i in range(1,forecast_length+1)]

    atimes_ns = np.array(atimes,dtype='datetime64[ns]')
    ftimes_ns = np.array(ftimes,dtype='datetime64[ns]')
    # print(atimes_ns)

    input_levels = sorted(list(task_config.pressure_levels))

    # Manually convert levels and times to integer indices in the
    # databases, to use .isel rather than .sel for selection.
    try:
        a_level_idx = np.searchsorted(a_dbase.level.data,input_levels)
        assert(all(a_dbase.level.data[a_level_idx] == input_levels))
        v_level_idx = np.searchsorted(v_dbase.level.data,input_levels)    
        assert(all(v_dbase.level.data[v_level_idx] == input_levels))

        a_time_idx = np.searchsorted(a_dbase.time.data,atimes_ns)
        assert(all(a_dbase.time.data[a_time_idx] == atimes_ns))
        v_time_idx = np.searchsorted(v_dbase.time.data,ftimes_ns)
        assert(all(v_dbase.time.data[v_time_idx] == ftimes_ns))
    except Exception as e:
        print(f'Error (database availability) when loading forecast of length {forecast_length} for {idate}')
        raise

    # # Create skeletal analysis
    # inputs = xr.Dataset(coords={'time':atimes_ns,
    #                     'level':input_levels,
    #                     'latitude':model_latitude,
    #                     'longitude':model_longitude}) 
    
    # assert(latitude[0] == a_dbase.latitude[0])
    # assert(longitude[0] == a_dbase.longitude[0])

    # Test to see if the model lat/lon exactly match the dataset
    if (model_latitude.size == a_dbase.latitude.size and \
        model_longitude.size == a_dbase.longitude.size and \
        np.all(model_latitude.data == a_dbase.latitude.data) and \
        np.all(model_longitude.data == a_dbase.longitude.data)):
        a_lat_idx = None
        a_lon_idx = None
        a_conforming_latlon = True
    else:
        # Assert that lat/lon of the model are a subset of the dataset
        assert(np.all(np.isin(model_latitude.data,a_dbase.latitude.data)))
        assert(np.all(np.isin(model_longitude.data,a_dbase.longitude.data)))
        # Assert that the database latitude/longitude are in increasing order
        assert(np.all(np.diff(a_dbase.latitude.data) > 0))
        assert(np.all(np.diff(a_dbase.longitude.data) > 0))
        if (not warned_a_conforming_latlon):
            warnings.warn('Analysis/IC dataset is higher resolution than the model; using indexed loading')
            warned_a_conforming_latlon = True
        a_conforming_latlon = False
        a_lat_idx = np.searchsorted(a_dbase.latitude.data,model_latitude.data)
        a_lon_idx = np.searchsorted(a_dbase.longitude.data,model_longitude.data)

    # Test to see if the model lat/lon exactly match the dataset
    if (model_latitude.size == v_dbase.latitude.size and \
        model_longitude.size == v_dbase.longitude.size and \
        np.all(model_latitude.data == v_dbase.latitude.data) and \
        np.all(model_longitude.data == v_dbase.longitude.data)):
        v_lat_idx = None
        v_lon_idx = None
        v_conforming_latlon = True
    else:
        # Assert that lat/lon of the model are a subset of the dataset
        assert(np.all(np.isin(model_latitude.data,v_dbase.latitude.data)))
        assert(np.all(np.isin(model_longitude.data,v_dbase.longitude.data)))
        # Assert that the database latitude/longitude are in increasing order
        assert(np.all(np.diff(v_dbase.latitude.data) > 0))
        assert(np.all(np.diff(v_dbase.longitude.data) > 0))
        if (not warned_v_conforming_latlon):
            warnings.warn('Validation/target dataset is higher resolution than the model; using indexed loading')
            warned_v_conforming_latlon = True
        v_conforming_latlon = False
        v_lat_idx = np.searchsorted(v_dbase.latitude.data,model_latitude.data)
        v_lon_idx = np.searchsorted(v_dbase.longitude.data,model_longitude.data)

    avars_present = list(set.intersection(set(a_dbase.data_vars),input_variables))
    # inputs[avars_present] = a_dbase[avars_present].isel(time=a_time_idx,level=a_level_idx).compute()
    # for var in avars_present:
    #     # print(var)
    #     if 'time' in a_dbase[var].dims:
    #         if 'level' in a_dbase[var].dims:
    #             # XArray "doesn't support yet fancy nd indexing", so we need to slice the data one dimension at a time
    #             if (a_conforming_latlon == True):
    #                 inputs[var] = (a_dbase[var].dims, a_dbase[var].data[a_time_idx,...][:,a_level_idx,:,:])
    #             else:
    #                 inputs[var] = (a_dbase[var].dims, a_dbase[var].data[a_time_idx,...][:,a_level_idx,...][...,a_lat_idx,:][...,a_lon_idx])
    #         else:
    #             if (a_conforming_latlon == True):
    #                 inputs[var] = (a_dbase[var].dims, a_dbase[var].data[a_time_idx,...])
    #             else:
    #                 inputs[var] = (a_dbase[var].dims, a_dbase[var].data[a_time_idx,...][...,a_lat_idx,:][...,a_lon_idx])

    inputs = xr.concat( [dbase_load(a_dbase,avars_present,tt,a_level_idx,a_lat_idx,a_lon_idx) for tt in a_time_idx],
                       coords='minimal',compat='override',dim='time')

    if ('toa_incident_solar_radiation') not in avars_present:
        # Since toa_radiation is being called based on a copy of the input coordinates, we don't
        # have to worry below that time becomes a relative value; contrast the special handling
        # for forcings
        inputs['toa_incident_solar_radiation'] = (('time','latitude','longitude'),
                                                    dask.array.from_delayed(dask.delayed(toa_radiation)(xr.Dataset(inputs.coords.copy())),
                                                                            shape=(inputs.time.size,inputs.latitude.size,inputs.longitude.size),
                                                                            dtype=np.float32))
        
    if ('day_progress_cos' not in avars_present):
        forecast_prep.add_progress(inputs)

        
    forcings = xr.Dataset(coords={'time':ftimes_ns,
                        'level':input_levels,
                        'latitude':model_latitude,
                        'longitude':model_longitude}) 
    # targets = xr.Dataset(coords={'time':ftimes_ns,
    #                     'level':input_levels,
    #                     'latitude':model_latitude,
    #                     'longitude':model_longitude}) 
    tvars_present = list(set.intersection(set(v_dbase.data_vars),target_variables))

    # print(ftimes_ns)

    targets = xr.concat( [dbase_load(v_dbase,tvars_present,tt,v_level_idx,v_lat_idx,v_lon_idx) for tt in v_time_idx],
                        coords='minimal',compat='override',dim='time')
    
    # for var in tvars_present:
    #     if 'time' in v_dbase[var].dims:
    #         if 'level' in v_dbase[var].dims:
    #             if (v_conforming_latlon == True):
    #                 targets[var] = (v_dbase[var].dims, v_dbase[var].data[v_time_idx,...][:,v_level_idx,...])
    #             else:
    #                 targets[var] = (v_dbase[var].dims, v_dbase[var].data[v_time_idx,...][:,v_level_idx,...][...,v_lat_idx,:][...,v_lon_idx])
    #         else:
    #             if (v_conforming_latlon == True):
    #                 targets[var] = (v_dbase[var].dims, v_dbase[var].data[v_time_idx,...])
    #             else:
    #                 targets[var] = (v_dbase[var].dims, v_dbase[var].data[v_time_idx,...][...,v_lat_idx,:][...,v_lon_idx])
    # targets[tvars_present] = v_dbase[tvars_present].isel(time=v_time_idx,level=v_level_idx).compute()

    # As with radiation for the inputs, call toa_radiation on a skeletal dataset made from coordinate copies;
    # this avoids problems when the coordinates are modified subsequently but before(/during?) radiation calculation
    forcings['toa_incident_solar_radiation'] = (('time','latitude','longitude'),
                                                dask.array.from_delayed(dask.delayed(toa_radiation)(xr.Dataset(forcings.coords.copy())),
                                                                        shape=(forcings.time.size,forcings.latitude.size,forcings.longitude.size),
                                                                        dtype=np.float32))
    forecast_prep.add_progress(forcings)

    # Graphcast uses the 'time' dimension for relative time; preserve the original analysis/validity time as
    # a 'datetime' coordinate.
    i_reltime = atimes_ns - atimes_ns[-1]
    inputs['time'] = i_reltime
    inputs.coords['datetime'] = ('time',atimes_ns)

    f_reltime = ftimes_ns - atimes_ns[-1]
    forcings['time'] = f_reltime
    targets['time'] = f_reltime
    forcings.coords['datetime'] = ('time',ftimes_ns)
    targets.coords['datetime'] = ('time',ftimes_ns)

    # Prepend a 'batch' dimension and rename 'latitude' and 'longitude' to 'lat' and 'lon'
    renames = {'latitude': 'lat', 'longitude' : 'lon'}
    inputs = inputs.expand_dims(dim='batch').rename(renames)
    forcings = forcings.expand_dims(dim='batch').rename(renames)
    targets = targets.expand_dims(dim='batch').rename(renames)
    
    fixed_input_vars = set.intersection(set(forecast.forecast_variables.FIXED_VARIABLES),set(input_variables))
    inputs[fixed_input_vars] = inputs[fixed_input_vars].isel(batch=0,time=0,drop=True)

    # inputs, targets, forcings = dask.optimize(inputs,targets,forcings)

    if (do_compute):
        (inputs, forcings, targets) = dask.compute(*(inputs,forcings,targets))
    
    return(inputs, forcings, targets)


## PrioritizedItem wrapper for the output priority queue
from dataclasses import dataclass, field
from typing import Any

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)

def open_one_database(path_glob):
    import xarray as xr
    import glob
    import numpy as np
    from forecast import encabulator
    import dask
    import warnings

    # Suppress ECCodes warning
    with warnings.catch_warnings(action="ignore"):
        dbase_paths = sorted(list(glob.glob(f'{path_glob}/*/*')))

        # * Open each month separately
        # * Drop toa_incident_solar_radiation if present (it's not consistently present and causes bad chunking on merge)
        # * set a chunk size of 1 in time
        dbase_by_month = [xr.open_dataset(path,cache=False,engine='zarr').chunk(time=1) for path in dbase_paths]
        dbase_by_month_norad = [ d[[v for v in d.data_vars if v != 'toa_incident_solar_radiation']] for d in dbase_by_month ]

    # Concatenate each moth together. compat='override' takes the first-encountered value for variables that are time-independent
    # like geopotential_at_surface.  join='override' was once present but is erroneous; it breaks if there's a different coordinate
    # ordering between months -- this happens with ERA5 data downloaded from weatherbench versus cds
    dbase = xr.concat(dbase_by_month_norad, dim='time', coords='minimal', data_vars='minimal', compat='override')

    # Drop any time duplicates
    dbase = dbase.drop_duplicates(dim='time')

    # Finally, dask-optimize the combined dataset, saving a bit of time later on when picking out individual elements
    dbase = dask.optimize(dbase)[0]
    return dbase

def open_databases(a_dbase_path, v_dbase_path):
    import xarray as xr
    import glob
    import numpy as np
    # Ensure loading of compression codec
    from forecast import encabulator
    import dask # For graph optimization
    import warnings

    a_dbase = open_one_database(a_dbase_path)

    # Assert that levels and times are sorted
    assert(all(np.sort(a_dbase.level.data) == a_dbase.level.data))
    assert(all(np.sort(a_dbase.time.data) == a_dbase.time.data))

    # ## Preload time-independent fields
    # for var in a_dbase.data_vars:
    #     if 'time' not in a_dbase[var].dims:
    #         a_dbase[var] = a_dbase[var].compute()

    if (a_dbase_path == v_dbase_path):
        v_dbase = a_dbase
    elif (v_dbase_path is None):
        v_dbase = None
    else:
        v_dbase = open_one_database(v_dbase_path)
        assert(all(np.sort(v_dbase.level.data) == v_dbase.level.data))
        assert(all(np.sort(v_dbase.time.data) == v_dbase.time.data))

    return (a_dbase, v_dbase)


def mp_consumer(job_queue, output_queue, forecast_length, params_path,
                a_dbase, v_dbase,proc_number):

    # all_procs = sorted(list(os.sched_getaffinity(0)))
    # if (len(all_procs) > proc_number+1):
    #     target_proc = all_procs[proc_number+1]
    #     os.sched_setaffinity(0,[target_proc])
    #     print(f'Proc {proc_number} pinned to processor {target_proc}')

    # os.environ['OMP_NUM_THREADS']='1'
    # os.environ['MKL_NUM_THREADS']='1'
    import xarray as xr
    from forecast import generate_model
    from forecast.models import models, models_dict
    import numpy as np
    
    import glob
    import dask

    # (a_dbase, v_dbase) = open_databases(a_dbase_path,v_dbase_path)

    (_, task_config, _) = generate_model.load_model(params_path)
    input_variables = set(task_config['input_variables'])
    target_variables = set(task_config['target_variables']) # things produced by graphcast, at t6
    # forcing_variables = set(task_config['forcing_variables']) # things privded as input, at t6

    latitude = a_dbase.latitude.compute()
    longitude = a_dbase.longitude.compute()

    while True:
        pcommand = job_queue.get()
        command = pcommand.item
        try:
            match command[0]:
                case 'terminate':
                    # print(f'Thread {proc_number} terminating')
                    return
                case 'build_example':
                    idate = command[1]
                    if (debug_prints):
                        print(f'Thread {proc_number} building {idate.strftime("%Y%m%d+%HZ")}')
                    tic = datetime.datetime.now()
                    # compute_lock.acquire()
                    (ids,fds,tds) = build_forecast(idate,forecast_length,task_config,
                                                latitude,longitude,
                                                input_variables,target_variables,
                                                a_dbase,v_dbase,False)
                    # compute_lock.release()
                    # datestamp = idate.strftime('%Y%m%d%H')
                    # print(f'Proc {proc_number} writing {idate.strftime("%Y%m%d+%HZ")}')
                    # dask.compute(ids,fds,tds)
                    toc = datetime.datetime.now()
                    tdelta = (toc-tic).total_seconds()
                    if (debug_prints):
                        print(f'Thread {proc_number} submitting {idate.strftime("%Y%m%d+%HZ")} tdelta {tdelta:.3f}s')
                    # output_queue.put((ifname,ffname,tfname,tdelta))
                    # Enqueue the example to the output queue, wrapped as a PrioritizedItem
                    out_item = PrioritizedItem(priority=0,
                                            item=('example',{'input' : ids,
                                                'forcing' : fds,
                                                'target' : tds,
                                                'timedelta' : tdelta}))
                    output_queue.put(out_item)
                    del out_item, ids, fds, tds
                    # output_queue.put((ids,fds,tds,tdelta))
                    # if (debug_prints):
                    #     print(f'Thread {proc_number} written {idate.strftime("%Y%m%d+%HZ")}')
                    # output_lock.release()
                case 'build_minibatch':
                    # Directly construct a minibatch, given a sequence of input dates
                    idate_seq = command[1]
                    inputs = []; targets = []; forcings = []
                    tic = datetime.datetime.now()
                    if (debug_prints):
                        print(f'Thread {proc_number} computing minibatch {" ".join(idate.strftime("%Y%m%d+%HZ") for idate in idate_seq)}')
                    for idate in idate_seq:
                        (ids,fds,tds) = build_forecast(idate,forecast_length,task_config,
                                                    latitude,longitude,
                                                    input_variables,target_variables,
                                                    a_dbase,v_dbase,False)
                        inputs.append(ids)
                        forcings.append(fds)
                        targets.append(tds)
                        del ids, fds, tds
                    input_ds = xr.concat(inputs,dim='minibatch',coords=['datetime'])
                    forcing_ds = xr.concat(forcings,dim='minibatch',coords=['datetime'])
                    target_ds = xr.concat(targets,dim='minibatch',coords=['datetime'])
                    del inputs, forcings, targets
                    # Dask.compute sometimes fails with an internal error; this remains not fully solved.
                    # Retry the computation up to 5 times
                    for retry in range(5):
                        try:
                            (input_ds,forcing_ds,target_ds) = dask.compute(*(input_ds,forcing_ds,target_ds))
                            break
                        except Exception as e:
                            import time
                            if (retry == 0):
                                print(f'Warning: thread {proc_number} received exception {e=} in dask.compute, retrying')
                                time.sleep(0.1)
                            elif (retry < 4):
                                print(f'Warning: thread {proc_number} received exception again, retry {retry+1}')
                                time.sleep(0.1)
                            else: # (retry == 4):
                                # import traceback
                                # traceback.print_exception(type(e), e, e.__traceback__)
                                raise(e) # Reraise exception after 5 attempts

                    toc = datetime.datetime.now()
                    tdelta = (toc-tic).total_seconds()

                    if (debug_prints):
                        print(f'Thread {proc_number} submitting minibatch {" ".join(idate.strftime("%Y%m%d+%HZ") for idate in idate_seq)}')

                    out_item = PrioritizedItem(priority=1,item=('minibatch',{'inputs' : input_ds,
                                                'forcings' : forcing_ds,
                                                'targets' : target_ds,
                                                'timedelta' : tdelta}))
                    output_queue.put(out_item)
                    del out_item, input_ds, forcing_ds, target_ds           
                case 'assemble_minibatch':

                    tic = datetime.datetime.now()
                    inputs = command[1].get('inputs',None)
                    targets = command[1].get('targets',None)
                    forcings = command[1].get('forcings',None)

                    # input_minibatch = xr.concat(input_minibatch_list,dim='minibatch',coords=['datetime'])
                    # forcing_minibatch = xr.concat(forcing_minibatch_list,dim='minibatch',coords=['datetime'])
                    # target_minibatch = xr.concat(target_minibatch_list,dim='minibatch',coords=['datetime'])


                    # idate_seq = np.array(input_ds.datetime[...,-1].data,dtype='datetime64[s]').astype(datetime.datetime).reshape((-1,))
                    if (inputs is not None and inputs[0] is not None):
                        idate_seq = [ ind.datetime.data.astype('datetime64[s]').astype(datetime.datetime)[-1] for ind in inputs ]
                        if (debug_prints):
                            print(f'Thread {proc_number} assembling minibatch {" ".join(idate.strftime("%Y%m%d+%HZ") for idate in idate_seq)}')

                        input_ds = xr.concat(inputs,dim='minibatch',coords=['datetime'])
                    else:
                        input_ds = None

                    if forcings is not None and forcings[0] is not None:
                        forcing_ds = xr.concat(forcings,dim='minibatch',coords=['datetime'])
                    else:
                        forcing_ds = None

                    if targets is not None and targets[0] is not None:
                        target_ds = xr.concat(targets,dim='minibatch',coords=['datetime'])
                    else:
                        target_ds = None

                    # compute_lock.acquire()
                    if (debug_prints):
                        print(f'Thread {proc_number} assembling minibatch {" ".join(idate.strftime("%Y%m%d+%HZ") for idate in idate_seq)}')
                    for retry in range(5):
                        try:
                            (input_ds, forcing_ds, target_ds) = dask.compute(*(input_ds,forcing_ds,target_ds))
                            break
                        except Exception as e:
                            print(f'Thread {proc_number} received exception {e=} computing minibatch {" ".join(idate.strftime("%Y%m%d+%HZ") for idate in idate_seq)}, retry {retry}')
                            if (retry == 4):
                                raise(e)
                            
                    # compute_lock.release()
                    del inputs, forcings, targets


                    toc = datetime.datetime.now()
                    tdelta = (toc-tic).total_seconds()

                    if (debug_prints):
                        print(f'Thread {proc_number} submitting minibatch {" ".join(idate.strftime("%Y%m%d+%HZ") for idate in idate_seq)} tdelta {tdelta:.3f}s')


                    out_item = PrioritizedItem(priority=1,item=('minibatch',{'inputs' : input_ds,
                                                'forcings' : forcing_ds,
                                                'targets' : target_ds,
                                                'timedelta' : tdelta}))
                    output_queue.put(out_item)
                    del out_item, input_ds, forcing_ds, target_ds

                case _:
                    print(f'Thread {proc_number} unknown {command=}')
                    raise(NotImplementedError())
                
        except Exception as e:
            print(f'Thread {proc_number} received exception {e=}')
            import traceback
            traceback.print_exception(type(e), e, e.__traceback__)
            output_queue.put(PrioritizedItem(priority=-999,item=('exception',None)))
            raise(e)
