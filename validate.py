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

import os
# Ideally, these environment variables should be set at the command-line, before
# even launching the script.  Results seem inconsistent when the environment variables
# are set via os.
if ('TF_FORCE_UNIFIED_MEMORY' not in os.environ.keys()):
    os.environ['TF_FORCE_UNIFIED_MEMORY']='1'
if ('XLA_CLIENT_MEM_FRACTION' not in os.environ.keys()):
    os.environ['XLA_CLIENT_MEM_FRACTION'] = '2.0'
# import multiprocessing as mp

def print_memory_stats(gc_tracked = [0]):
    import jax
    
    print('Memory stats:')
    nowbytes = 0
    peakbytes = 0
    for dev in range(len(jax.local_devices())):
        mstat = jax.local_devices()[dev].memory_stats()
        if (mstat is not None): # Returns none on CPU
            nowbytes += mstat['bytes_in_use']
            peakbytes += mstat['peak_bytes_in_use']
        # print(f'Device {dev}: {jax.local_devices()[dev].memory_stats()}')
    asizes = [a.nbytes for a in jax.live_arrays()]
    import gc
    print(f'   {nowbytes / 1024 / 1024 / 1024 :.2f}GiB GPU memory in use, {peakbytes / 1024 / 1024 / 1024 :.2f}GiB peak',
            f'{len(asizes)} live Jax arrays of total size {sum(asizes)/1024/1024/1024:.2f}GiB',
            flush=True,end='')
    gc_tracked_now = len(gc.get_objects())
    print(f' {gc_tracked_now} Python objects, ({gc_tracked_now - gc_tracked[0]:+d})',flush=True,end='')
    print('',flush=True)
    gc_tracked[0] = gc_tracked_now

# Testing: locking JIT to a single thread may hurt performance with >2 GPUs
# import threading
# jit_lock = threading.Lock()
# grad_jitted = False

def wrap_dataset(ds,device):
    import graphcast.xarray_jax
    import jax
    return (graphcast.xarray_jax.Dataset(coords=ds.coords,
                                        data_vars = {k : (ds[k].dims,jax.device_put(graphcast.xarray_jax.unwrap_data(ds[k]),device=device)) for k in ds.data_vars}))
# import jax
# @jax.jit
def slice_ds(in_ds,idx):
    import jax.numpy as jnp
    
    import graphcast.xarray_jax
    coords = dict(in_ds.coords)
    coords['time'] = coords['time'][:1]
    data_vars = {}
    for v in in_ds.data_vars:
        if 'time' in in_ds[v].dims:
            data_vars[v] = (in_ds[v].dims,in_ds[v].data.jax_array[:,[idx,],...])
        else:
            data_vars[v] = (in_ds[v].dims,in_ds[v].data.jax_array)
    return graphcast.xarray_jax.Dataset(coords=coords,data_vars=data_vars)

# @jax.jit
def stack_inputs(old_input,pred,forcings):
    import xarray as xr
    inputs_next = pred[input_from_target]
    inputs_next[input_from_forcing] = forcings[input_from_forcing]
    for v in inputs_next.data_vars:
        inputs_next[v] = inputs_next[v].transpose(*old_input[v].dims)
    outputs = xr.concat((old_input.isel(time=[1,]),inputs_next),dim='time',coords='minimal',data_vars='minimal')
    outputs['time'] = old_input['time']
    return outputs

def loss_by_slice(inputs,targets,forcings,params,predictor,field_loss_fn):
    import jax.numpy as jnp
    global use_cpu_targets
    total_lead = targets.time.size
    loss = jnp.float32(0)
    my_device = list(inputs.geopotential.data.jax_array.devices())[0]
    for lead in range(total_lead):
        if (use_cpu_targets):
            targets_now = wrap_dataset(slice_ds(targets,lead),my_device)
            forcings_now = wrap_dataset(slice_ds(forcings,lead),my_device)
        else:
            targets_now = slice_ds(targets,lead)
            forcings_now = slice_ds(forcings,lead)
        pred = predictor(inputs=inputs,forcings=forcings_now,targets=targets_now,params=params)
        loss = loss + field_loss_fn(pred,targets_now)[0].data.jax_array
        del targets_now
        if (lead < total_lead-1):
            inputs = stack_inputs(inputs,pred,forcings_now)
        del forcings_now
    return loss/total_lead

def get_loss(tags,params,predictor,loss_fn):
    # In parallel, get and return the loss for a given configuration
    import datetime
    global gpu_queue
    # Get forecast input arrays from the GPU queue; these datasets have already
    # been pushed to the GPU device for forecast generation
    tic = datetime.datetime.now()
    (inputs,forcings,targets) = gpu_queue.get()
    # Push the target array to the device to signal the GPU for computation
    # (loss, diags) = loss_fn(inputs=inputs,forcings=forcings,targets=targets,params=params)
    loss = loss_by_slice(inputs=inputs,targets=targets,forcings=forcings,params=params,predictor=predictor,field_loss_fn=loss_fn)
    loss = float(loss)
    gpu_queue.put((inputs,forcings,targets))
    toc = datetime.datetime.now()
    return(tags,float(loss),(toc-tic).total_seconds())


def split_futures(futures):
    # Utility function to split a set of (dask) Futures into a done and not-done set
    done = []
    not_done = []
    for f in futures:
        if (f.done()):
            done.append(f)
        else:
            not_done.append(f)
    return(done,not_done)
    
def stamp(idate):
    # Helper function to return a YYYY-MM-DDTHH datetamp given
    # a datetime object
    return(idate.strftime('%Y-%m-%dT%H'))

def scatter_inputs(inputs,forcings,targets):
    # Scatter forecast inputs to each GPU device, posting the datasets
    # to the device queue
    global gpu_queue
    global use_cpu_targets
    import jax
    cpu_device = jax.devices('cpu')[0]
    for device in jax.devices('gpu'):
        inputs_gpu = wrap_dataset(inputs,device)
        if (use_cpu_targets):
            targets_cpu = wrap_dataset(targets,cpu_device)
            forcings_cpu = wrap_dataset(forcings,cpu_device)
            gpu_queue.put((inputs_gpu,forcings_cpu,targets_cpu))
        else:
            targets_gpu = wrap_dataset(targets,device)
            forcings_gpu = wrap_dataset(forcings,device)
            gpu_queue.put ((inputs_gpu, forcings_gpu, targets_gpu))

if __name__ == '__main__':
    import argparse

    ## Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpath',type=str,dest='dpath',default='../gdata_025_wb',help='Location of analysis data')
    parser.add_argument('--start-date',type=str,dest='start_date',default='1 Jan 2020 00:00',help='Starting date/time')
    parser.add_argument('--end-date',type=str,dest='end_date',default='31 Dec 2021 18:00',help='Ending date/time (inclusive)')
    parser.add_argument('--forecast-length',type=int,dest='forecast_length',default=1)
    parser.add_argument('--to-csv',type=str,dest='csvpath',default=None,help='(optional) CSV file for scores')
    parser.add_argument('--batch-size',type=int,dest='batch_size',default=32,help='Number of forecast dates used for validation')
    parser.add_argument('--model-checkpoints',type=str,dest='model_checkpoints',default=None,
                        help='File pattern (glob) for model checkpoints to evaluate')
    parser.add_argument('--error-weights',type=str,dest='error_weight_file',default=None,
                        help='File containing non-default variable and level weights')
    parser.add_argument('--norm-factors',type=str,dest='norm_path',default=None,
                        help='Path to the directory containing Graphcast normalization factors')
    parser.add_argument('--limit',type=int,default=-1,help='Evaluate only <limit> checkpoints from the specified glob (alphabetically sorted)')    
    parser.add_argument('--skip',type=int,default=-1,help='Skip the first <skip> checkpoints of the specified glob (alphabetically sorted)')


    args = parser.parse_args()

    # Forecast options: forecast length and dataset paths
    forecast_length = args.forecast_length
    apath = args.dpath

    # CSV output path
    csvpath = args.csvpath

    batch_size = args.batch_size

    import glob
    checkpoint_files = sorted(glob.glob(args.model_checkpoints))

    print(f'Found {len(checkpoint_files)} model checkpoints')
    if (len(checkpoint_files) < 1):
        import sys
        sys.exit(0)

    if args.skip > 0:
        assert(args.skip < len(checkpoint_files))
        checkpoint_files = checkpoint_files[args.skip:]
    
    if (args.limit > 0 and args.limit < len(checkpoint_files)):
        checkpoint_files = checkpoint_files[:args.limit]

    if (args.limit > 0 or args.skip > 0):
        print(f'   ... evaluating {len(checkpoint_files)} checkpoints starting with {max(0,args.skip)}')
    else:
        print(f'   ... evaluating all checkpoints')

    import xarray as xr
    import numpy as np
    import numcodecs
    import trainer.dataloader
    import forecast.encabulator
    import dateparser
    import jax
    import datetime
    import dask
    import dask.distributed
    import time

    slice_ds = jax.jit(slice_ds)
    stack_inputs = jax.jit(stack_inputs)

    # Disable threading inside blosc
    numcodecs.blosc.use_threads = False

    start_date = dateparser.parse(args.start_date,
                                  ['%Y%m%d%H',  # Also parse YYYYMMDDHH (ISO 8601-2004)
                                   '%Y%m%d%HZ', # ... with UTC marker
                                   '%Y%m%dT%H', # and YYYYMMDDTHH (ISO 8601-2019)
                                   '%Y%m%dT%HZ',# ... with UTC marker
                                  ])
    end_date = dateparser.parse(args.end_date,
                                ['%Y%m%d%H',  # Also parse YYYYMMDDHH (ISO 8601-2004)
                                 '%Y%m%d%HZ', # ... with UTC marker
                                 '%Y%m%dT%H', # and YYYYMMDDTHH (ISO 8601-2019)
                                 '%Y%m%dT%HZ',# ... with UTC marker
                                ])

    # File for user-specified level/variable error weights
    error_weight_file = args.error_weight_file

    # Model parameters and checkpoint schema
    from forecast import generate_model
    from forecast.models import models_dict
    param_set = {}
    for params_file in checkpoint_files:
        try:
            (model_config, task_config, params) = generate_model.load_model(params_file)
        except Exception as e:
            print(f'Error loading {params_file}')
            raise(e)
        param_set[params_file] = params

    # Open database
    dbase,_ = trainer.dataloader.open_databases(apath,None) # Note no need for a separate verification dbase
    # Generate latitude and longitude for the model, based on its resolution
    model_latitude = xr.DataArray(np.linspace(-90,90,int(1+180/model_config['resolution']),dtype=np.float32),dims='latitude')
    model_latitude = model_latitude.assign_coords({'latitude' : model_latitude})
    model_longitude = xr.DataArray(np.linspace(0,360-model_config['resolution'],int(360/model_config['resolution']),dtype=np.float32),
                                dims='longitude')
    model_longitude = model_longitude.assign_coords({'longitude' : model_longitude})

    input_variables = list(task_config['input_variables'])
    target_variables = list(task_config['target_variables'])    
    forcing_variables = list(task_config['forcing_variables'])    
    # Define variable sources for future-IC generation
    input_only_vars = [v for v in input_variables if v not in target_variables]
    global input_from_target
    global input_from_forcing
    input_from_target = [v for v in input_variables if v in target_variables]
    input_from_forcing = [v for v in input_only_vars if v in forcing_variables]

    norm_path = args.norm_path

    if (norm_path is not None):
        print(f'Using normalization factors in {norm_path}')
        # Load provided normalization factors
        diffs_stddev_by_level = xr.load_dataset(f"{norm_path}/diffs_stddev_by_level.nc").compute()
        mean_by_level = xr.load_dataset(f"{norm_path}/mean_by_level.nc").compute()
        stddev_by_level = xr.load_dataset(f"{norm_path}/stddev_by_level.nc").compute()
    else:
        # Otherwise do not load normalization factors, and default to the loading inside the predictor-generator
        diffs_stddev_by_level = None
        mean_by_level = None
        stddev_by_level = None
    # If using custom error weightings, build the appropriate error function
    if (error_weight_file is None):
        error_weight_file='error_weights/deepmind.pickle'

    if (error_weight_file is not None):
        print(f'Using error weight file {error_weight_file}')
        with open(error_weight_file,'rb') as weightfile:
            import graphcast.losses
            import trainer.loss_utils
            import pickle
            
            (per_variable_weights, level_weights) = pickle.load(weightfile)

            # Re-normalize level weights to have sum of 1; this accounts for loading
            # 37-level weights with a 13-level version of the model
            level_weights = level_weights.sel(level=list(task_config['pressure_levels']))
            level_weights = level_weights / level_weights.sum()

            # The builtin Graphcast loss function operates in the normalized forecast increment space.
            # That means that predicted variables that are also input variables are expressed as
            # (prediction - input)/Δstd, and predicted variables that are not input variables are
            # expressed as (prediction - mean)/std.  We don't care about the mean-subtraction because
            # it applies to both the prediction and the target, but we do need to know whether to divide
            # by the standard deviation of the field or its 6h increment.
            if (diffs_stddev_by_level is None):
                diffs_stddev_by_level = xr.load_dataset("stats/diffs_stddev_by_level.nc").compute()
            if (stddev_by_level is None):
                stddev_by_level = xr.load_dataset('stats/stddev_by_level.nc').compute()
            
            norms_by_level = xr.merge( [ diffs_stddev_by_level[v] if v in input_variables else stddev_by_level[v] \
                                            for v in target_variables ])
            
            latitude_weights = graphcast.losses.normalized_latitude_weights(model_latitude.rename(latitude='lat'))
            latitude_weights = latitude_weights / latitude_weights.mean()

            custom_loss = trainer.loss_utils.make_loss(norms_by_level,per_variable_weights,level_weights,latitude_weights)
    else:
        assert(False)
        custom_loss = None

    # Build predictor using full float32 precision
    predictor = generate_model.build_predictor_params(model_config, task_config, use_float16=False,
                                                      diffs_stddev_by_level = diffs_stddev_by_level, 
                                                      mean_by_level = mean_by_level,
                                                      stddev_by_level = stddev_by_level)

    # # Build loss operator (discard gradient component), using full float32 precision
    # loss_fn, _ = generate_model.build_loss_and_grad(model_config, task_config, use_float16=False,custom_loss_fn=custom_loss,
    #                                                   diffs_stddev_by_level = diffs_stddev_by_level, 
    #                                                   mean_by_level = mean_by_level,
    #                                                   stddev_by_level = stddev_by_level)


    import queue
    # Set up a GPU queue to hold on-device parameters and gradient accumulation arrays, allowing a grad-calculator
    # to run on an available GPU by popping from the queue
    gpu_queue = queue.Queue()
    gpu_device_0 = jax.devices('gpu')[0]
    cpu_device = jax.devices('cpu')[0]
    num_gpus = len(jax.devices('gpu'))

    print(f'Calculating validation loss for {len(param_set)} checkpoints over {batch_size} forecast dates (length {forecast_length*6}h) between {stamp(start_date)} and {stamp(end_date)}')
    print(f'Using {num_gpus} GPUs')

    global use_cpu_targets
    if (forecast_length < 9):
        # If the forecast length is short, put the full target array on-GPU.  In experimentation, the crossover point
        # is somewhere around an 8-step forecast length.
        use_cpu_targets = False
        print('Putting forecst targets on-GPU for validation')
    else:
        # Otherwise, keep it on-CPU and move it over slice by slice, to conserve GPU memory
        use_cpu_targets = True
        print('Keeping forecast targets on-CPU for validation')


    # Import faulthanlder, which will act as a watchdog to dump a stacktrace in the event that things hang
    import faulthandler
    # Set the traceback to dump after 10 minutes, which should be long enough to accommodate any jax compilations
    faulthandler.dump_traceback_later(600,exit=True)

    # Use a with-block for Dask, the threaded gradient executor, and (optionally) CSV writing
    import concurrent.futures
    import contextlib
    import os

    dead_tic = datetime.datetime.now()
    dead_toc = datetime.datetime.now()
    dead_time = True

    with (dask.distributed.Client(processes=False,threads_per_worker=len(os.sched_getaffinity(0))) as dask_client, 
          concurrent.futures.ThreadPoolExecutor(max_workers = num_gpus) as gpu_executor, 
          (open(csvpath,'w') if csvpath else contextlib.nullcontext()) as csvfile):

        # Prepopulate the list of training samples
        # possible_times contains the set of times which are present in the database and can be
        # used to initialize the forecast.  The full database itself will need times before and
        # after this (for the -6h IC and verification targets), but we shouldn't use those as T=0
        # initial conditions.
        possible_times = dbase.time.sel(time=slice(start_date,end_date)).data
        # Use a freshly seeded random number genrator for reproducible selection
        rng = np.random.Generator(np.random.PCG64((possible_times.size,batch_size)))
        samples = list(rng.choice(possible_times,batch_size).astype('datetime64[s]').astype(datetime.datetime))
            
        # Write the CSV file header
        if (csvfile):
            csvfile.write('Date, Params, Loss\n')
            csvfile.flush()

        # Initilaize working variables before the training loop begins
        processed = 0 # How many training examples have been processed so far
        batch_processed = 0 # How many training examples in the paramter set have been processed for this date
        queued_inputs = [] # List of pending Futures (dask) for input loading
        queued_losses = [] # List of pending Futures (threadpool) for grad generation
        max_queued_inputs = 2
        ready_forecasts = [] # List of ready (date,inputs,forcings,targets) tuples

        batch_tic = datetime.datetime.now() # Time the current forecast set for information
        while processed < batch_size:
            productive = False # Whether something has happened this iteration

            # If we don't have enough samples loading, load a new one
            if (len(samples) > 0 and len(queued_inputs) + len(ready_forecasts) < max_queued_inputs):
                idate = samples.pop()
                (inputs, forcings, targets) = trainer.dataloader.build_forecast(idate, forecast_length, task_config,
                                                                                model_latitude, model_longitude, input_variables, target_variables,
                                                                                dbase, dbase)
                (inputs, forcings, targets) = dask.optimize(inputs, forcings, targets)
                # print(f'Queueing input for {stamp(idate)}')
                queued_inputs.append(dask_client.submit(tuple,dask_client.compute((inputs,forcings,targets))))
                productive = True
                # Clear references to free memory
                del inputs, targets, forcings

            # If there are no losses being computed and we have one ready, enqueue it for computation
            if (len(ready_forecasts) > 0 and len(queued_losses) == 0):
                if (dead_time == True):
                    dead_time = False
                    dead_toc = datetime.datetime.now()
                    print(f'Forecast data ready after {(dead_toc-dead_tic).total_seconds():.2f}s of loading time')
                (inputs,forcings,targets) = ready_forecasts.pop()
                # Infer the analysis date
                idate = np.datetime64(inputs.datetime.data[-1],'s').astype(datetime.datetime)
                # print(f'Preparing forecast for {stamp(idate)}')
                # Drop the datetime coordinate
                inputs = inputs.drop_vars('datetime')
                forcings = forcings.drop_vars('datetime')
                targets = targets.drop_vars('datetime')
                
                # Clear existing data from GPU queue
                if (processed > 0):
                    for idx in range(num_gpus):
                        # Use a nonzero timeout to detect a deadlock.  If we reach this point,
                        # then all computations should have finished and the GPU queue should
                        # contain all the devices.  If we can't pop once per device, then
                        # there's some deadlock.
                        gpu_queue.get(timeout=0.1)
                # Scatter data to GPU
                tic = datetime.datetime.now()
                scatter_inputs(inputs,forcings,targets)
                toc = datetime.datetime.now()
                print(f'Inputs scattered to GPU in {(toc-tic).total_seconds():.2f}s')

                # Start one loss computation per paramter set:
                for (param_path, params) in param_set.items():
                    queued_losses.append(gpu_executor.submit(get_loss,(idate,param_path),params,predictor,custom_loss))

                productive = True
                del inputs, targets, forcings

            # Test whether any queued inputs are ready
            (queued_inputs_ready, queued_inputs_unready) = split_futures(queued_inputs)
            for future in queued_inputs_ready:
                (inputs,forcings,targets) = future.result()
                ready_forecasts.append((inputs,forcings,targets))
                del inputs, targets, forcings
                productive = True
            queued_inputs = queued_inputs_unready

            if (len(ready_forecasts) == 0 and len(queued_losses) == 0 and dead_time == False):
                # Start the clock for dead time if there are no ready forecasts and no losses being computed
                dead_time = True
                dead_tic = datetime.datetime.now()

            # Now, process losses as they come in
            batch_processed = 0
            if (len(queued_losses) > 0):
                for future in concurrent.futures.as_completed(queued_losses):
                    ((idate, param_path), loss, dt) = future.result()
                    batch_processed += 1
                    print(f'Forecast {batch_processed}/{len(param_set)} for {stamp(idate)} ({processed+1}/{batch_size}), params {param_path}, loss {loss:.3f}, {dt=:.3f}')
                    if (csvfile):
                        csvfile.write(f'{stamp(idate)}, "{param_path}", {loss:.5e}\n')
                    productive = True
                    # Feed the watchdog
                    import sys
                    faulthandler.dump_traceback_later(600,exit=True)
                    sys.stdout.flush()
                
                if (batch_processed):
                    # Upon processing a paramter-batch of forecasts, flush the output csv and note that we've finished
                    # a date
                    queued_losses = []    
                    if (csvfile is not None): csvfile.flush()
                    processed += 1
                    batch_toc = datetime.datetime.now()
                    dt = (batch_toc-batch_tic).total_seconds()
                    print(f'Forecast set {processed}/{batch_size} completed in {dt:.2f}s ({dt/len(param_set):.3f}s per)')
                    batch_tic = batch_toc

            if (productive):
                # Feed the watchdog
                import sys
                faulthandler.dump_traceback_later(600,exit=True)
                sys.stdout.flush()
            else:
                # Otherwise sleep to allow threads to work
                import time
                time.sleep(0.01)

