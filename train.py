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
    os.environ['XLA_CLIENT_MEM_FRACTION'] = '5.0'
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

    
def wrap_dataset(ds,device):
    import graphcast.xarray_jax
    import jax
    return (graphcast.xarray_jax.Dataset(coords=ds.coords,
                                        data_vars = {k : (ds[k].dims,jax.device_put(graphcast.xarray_jax.unwrap_data(ds[k]),device=device)) for k in ds.data_vars}))


# Testing: locking JIT to a single thread may hurt performance with >2 GPUs
import threading
jit_lock = threading.Lock()
import collections
grad_jitted = collections.defaultdict(lambda : False)

def split_grad(idate,inputs,forcings_list,targets_list,grad_fn,grad_weight,predictor,dask_client):
    # Compute a combined gradient based on split but sequential inputs
    # forcings_list and targets_list are lists of xarrays.  The first entry in these lists
    # must be realized, but the subesequent entires need not be.  This function will use
    # dask_client to compute (load) the data during the gradient and IC calculation.
    import datetime
    import jax
    import datetime
    import contextlib
    global gpu_queue
    global grad_jitted
    global debug_prints
    (params_gpu, grad_accum_gpu, device) = gpu_queue.get()
    tic = datetime.datetime.now()
    
    total_lead_time = sum(t.time.size for t in targets_list)
    loss_accum = 0
    
    inputs_now = wrap_dataset(inputs,device)

    assert(len(targets_list) == len(forcings_list))
    assert(len(targets_list) > 0)

    # Get forcings and targets from the queue, and move them to the GPU
    targets_now = wrap_dataset(targets_list.pop(0),device)
    forcings_now = wrap_dataset(forcings_list.pop(0),device)

    grad_count = 1
    
    while True:
        this_lead = targets_now.time.size
        if (debug_prints):
            dtic1 = datetime.datetime.now()
            print(f'Computing gradient of {stamp(idate)} on {device}, stage {grad_count} for {this_lead} steps')
        if (len(targets_list) > 0):
            # If there are targets left, we'll be continuing the computation
            continue_computation = True
            # In the background, commence loading the next bunch of target/forcing data
            (targets_future,forcings_future) = dask_client.compute((targets_list.pop(0),forcings_list.pop(0)),sync=False)
        else:
            continue_computation = False

        # Compute the gradient over this segment
        with (jit_lock if not grad_jitted[(device,this_lead)] else contextlib.nullcontext()) as _:
            if (debug_prints):
                dtic2 = datetime.datetime.now()
                if ((dtic2-dtic1).total_seconds() > 1):
                    print(f'Commencing gradient computation on {device}, waited {(dtic2-dtic1).total_seconds():.2f}s for compilation lock')
                else: dtic2=dtic1
            (loss_now, _, grad_now) = grad_fn(inputs=inputs_now,forcings=forcings_now,targets = targets_now,params=params_gpu)
            grad_jitted[(device,this_lead)] = True
        # print(f'size {targets_now.time.size} in {(toc-tic).total_seconds():.2f}s')

        # Add the loss and gradient to their accumulators
        
        if (debug_prints):
            dtic3 = datetime.datetime.now()
            print(f'Accumulating gradient on {device} (+{(dtic3-dtic2).total_seconds():.2f}s)')
        loss_accum += loss_now*(targets_now.time.size/total_lead_time)
        grad_accum_gpu = grad_accumulate_jit(grad_now,grad_accum_gpu,grad_weight*this_lead/total_lead_time)        
        if (debug_prints):
            dtic4 = datetime.datetime.now()
            print(f'Gradient accumulated on {device} (+{(dtic4-dtic3).total_seconds():.2f}s)')

        if (continue_computation):
            grad_count += 1
            # Generate the next set of ICs
            inputs_now = future_ics(predictor=predictor,inputs=inputs_now,forcings=forcings_now,targets=targets_now,
                                    params=params_gpu,lead=targets_now.time.size)
            if (debug_prints):
                dtic5 = datetime.datetime.now()
                print(f'New ICs generated on {device} (+{(dtic5-dtic4).total_seconds():.2f}s)')
            # And realize the loading of the next target/forcing data.
            targets_now = wrap_dataset(targets_future.result(),device)
            forcings_now = wrap_dataset(forcings_future.result(),device)
            if (debug_prints):
                dtic6 = datetime.datetime.now()
                print(f'Next targets loaded on {device} (+{(dtic6-dtic5).total_seconds():.2f}s)')
                dtic_last = dtic6
        else:
            if (debug_prints):
                dtic_last=dtic4
            break

        
    gpu_queue.put( (params_gpu, grad_accum_gpu, device) )
    # Uncomment to block this call until the gradient accumulation is completed.
    # With this call commented out, timings may be inaccurate because Jax might return speculatively
    # while commputation is still happening on the GPU; the measured timings might be a better reflection
    # of how long it took to compute the _last_ gradient rather than the current one. 

    # With the call uncommented, timings will be more accurate, but the blocking will eliminate real
    # opportunities to overlap computation with other work, potentially causing a small slowdown.
    
    # next(iter(next(iter(grad_accum_gpu.values())).values())).block_until_ready()
    toc = datetime.datetime.now()
    if (debug_prints):
        print(f'Finished on {device}, {(toc-tic).total_seconds():.2f}s (+{(toc-dtic_last).total_seconds():.2f}s)')
    return (idate, loss_accum, (toc-tic).total_seconds())

# Old gradient update function, not supporting split-horizon gradient computation
# def grad_update(idate,inputs,forcings,targets,grad_fn,grad_weight):
#     # In parallel, execute a prediction and accumulate the gradient to the on-GPU accumulator
#     import datetime
#     import contextlib
#     global gpu_queue
#     global grad_jitted
#     global debug_prints
#     (params_gpu, grad_accum_gpu, device) = gpu_queue.get()
#     tic = datetime.datetime.now()
#     # # Jax's JIT of GraphCast is very spammy thanks to the long compilation times.  We don't
#     # # really want to repeat these messages per GPU.  If the gradient function has not yet
#     # # been compiled, acquire jit_lock to serialize the compilation; everyone who waits on
#     # # this lock should see a faster first-compile through reuse.
#     with (jit_lock if not grad_jitted[device] else contextlib.nullcontext()) as _:
#         if (debug_prints):
#             tic2 = datetime.datetime.now()
#             print(f'Computing gradient on {device=}, {(tic2-tic).total_seconds():.2f}s')
#         new_grad = grad_fn(inputs=inputs, forcings=forcings, targets=targets, params=params_gpu)
#         grad_jitted[device] = True
#     del inputs, forcings, targets
#     if (debug_prints):
#         tic3 = datetime.datetime.now()
#         print(f'Accumulating gradient on {device=}, {(tic3-tic).total_seconds():.2f}s (+{(tic3-tic2).total_seconds():.2f}s)')
#     new_grad_accum_gpu = grad_accumulate_jit(new_grad[2],grad_accum_gpu,grad_weight)
#     # grad_jitted = True
#     err = np.array(new_grad[0])
#     del new_grad
#     gpu_queue.put( (params_gpu, new_grad_accum_gpu, device) )
#     toc = datetime.datetime.now()
#     if (debug_prints):
#         print(f'Finished on {device=}, {(toc-tic).total_seconds():.2f}s (+{(toc-tic3).total_seconds():.2f}s)')
#     return (idate, err, (toc-tic).total_seconds())

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

def zero_grad_like(grad):
    '''Compute a tree structure of all zeros, matching the composition of an input
    structure – intended to initialize gradient updates given a sample gradient'''
    import tree
    return tree.map_structure(lambda gr: 0*gr, grad)

def grad_accumulate(grad,accum,weight):
    '''Return accum + weight*grad, intended to accumulate gradients over several independent
    examples of a batch'''
    import tree
    # Accumulate the gradient
    if (accum is None):
        if (weight == 1.0):
            return grad
        else:
            return tree.map_structure(lambda gr: gr*weight,grad)
    return tree.map_structure(lambda gr, acc : acc + gr*weight, grad, accum)


def consolidate_grad():
    # Consolidate accumulated gradients between GPU devices
    import jax
    global gpu_queue
    global params_device
    global grad_accumulate_jit
    accum_grad = None
    for idx in range(len(jax.devices('gpu'))):
        (params, grad, device) = gpu_queue.get(timeout=0.1)
        grad = jax.device_put(grad,params_device)
        if (accum_grad is None):
            accum_grad = grad
        else:
            accum_grad = grad_accumulate_jit(grad,accum_grad,1.0)
    assert(gpu_queue.empty())
    return accum_grad

def scatter_params(params):
    # Scatter the parameters to each GPU device, posting the paramters
    # and an initialized (zero) gradient accumulator to the device queue
    global gpu_queue
    import jax
    import trainer.grad_utils
    for device in jax.devices('gpu'):
        # print(f'Scattering parameters to {device=}')
        params_gpu = jax.device_put(params,device)
        accum_gpu = trainer.grad_utils.zero_grad_like(params_gpu)
        gpu_queue.put( (params_gpu, accum_gpu, device) )

def params_update(optimizer,accum_grad,opt_state,params):
    # Use a provided optax updater to update paramters, returning
    # the new paramters and the updated optimizer state
    import optax
    # print('Applying optimizer update')
    updates, opt_state = optimizer.update(accum_grad,opt_state,params)
    params = optax.apply_updates(params,updates)
    # print('... done')
    return(params,opt_state)

def write_checkpoint(path_schema,batch_number,params,model_config,task_config):
    checkpoint_filename = path_schema.format(batchnum = batch_number)
    with open(checkpoint_filename,'wb') as cfile:
        from graphcast import checkpoint
        import graphcast
        checkpoint.dump(cfile,graphcast.graphcast.CheckPoint(params=params,
                                                            model_config=model_config,
                                                            task_config=task_config,
                                                            description=f'Model checkpoint batch {batch_number}',license=""))
        
def write_opt_checkpoint(path_schema,batch_number,opt_state):
    opt_checkpoint_filename = path_schema.format(batchnum = batch_number) + '.opt'
    with open(opt_checkpoint_filename,'wb') as cfile:
        import pickle
        pickle.dump(opt_state,cfile,-1)

device_target_template = collections.defaultdict(lambda : None)
import jax
@jax.jit
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

@jax.jit
def stack_inputs(old_input,pred,forcings):
    global input_from_target
    global input_from_forcing
    inputs_next = pred[input_from_target]
    inputs_next[input_from_forcing] = forcings[input_from_forcing]
    for v in inputs_next.data_vars:
        inputs_next[v] = inputs_next[v].transpose(*old_input[v].dims)
    outputs = xr.concat((old_input.isel(time=[1,]),inputs_next),dim='time',coords='minimal',data_vars='minimal')
    outputs['time'] = old_input['time']
    return outputs

def future_ics(predictor,inputs,forcings,targets,params,lead):
    '''Given a predictor function, generate a set of Graphcast-compatible initial conditions
    by taking a basic input and integrating it over a given period, determined by the conventional
    forcings and targets arguments'''
    import xarray as xr
    import datetime
    global debug_prints
    params_device = list(list(list(params.values())[0].values())[0].devices())[0]
    # print(f'Computing ICs {params_device}')
    tic = datetime.datetime.now()
    # Advancing one step at a time, construct initial conditions valid at +lead*6h
    if (device_target_template[params_device] is None):
        device_target_template[params_device] = targets.isel(time=[0,]).copy(deep=True)
        # print(f'Creating target template for {params_device}, loaded on {device_target_template[params_device].geopotential.data.jax_array.device()}')
    targets_template = device_target_template[params_device]
    toc = datetime.datetime.now()
    # print(f'IC setup device {params_device} target template in {(toc-tic).total_seconds():.2f}s')
    tic=toc
    for it in range(lead):
        # forcings_now = forcings.isel(time=[it,])
        # forcings_now['time'] = targets_template['time']
        forcings_now = slice_ds(forcings,it)
        toc = datetime.datetime.now()
        # print(f'IC iter {it} device {params_device} forcings {(toc-tic).total_seconds():.2f}')
        tic=toc
        pred = predictor(inputs=inputs,forcings=forcings_now,targets=targets_template,params=params)
        pred.geopotential.data.jax_array.block_until_ready()
        toc = datetime.datetime.now()
        # print(f'IC iter {it} device {params_device} prediction {(toc-tic).total_seconds():.2f}')
        tic=toc

        inputs = stack_inputs(inputs,pred,forcings_now)
        toc=datetime.datetime.now()
        # print(f'IC iter {it} device {params_device} inputs_next {(toc-tic).total_seconds():.2f}')
        tic=toc
    return inputs

def data_split(targets,forcings,sizes):
    '''Given targets and forcings variables, split them into a disjoint set specified by
    the sizes parameter.  Rewrite the 'time' variable of each such that the resulting variables
    all begin at +6h.'''
    import numpy as np
    out_targets = []
    out_forcings = []
    assert(sum(sizes) == targets.time.size)
    assert(all(s >= 0 for s in sizes))
    for s in sizes:
        if (s == 0): continue
        t = targets.isel(time=slice(0,s))
        f = forcings.isel(time=slice(0,s))
        t['time'] = t['time'] - t['time'][0] + np.timedelta64(6,'h')
        f['time'] = f['time'] - f['time'][0] + np.timedelta64(6,'h')
        out_targets.append(t)
        out_forcings.append(f)

        targets = targets.isel(time=slice(s,None))
        forcings = forcings.isel(time=slice(s,None))
    return (out_targets,out_forcings)


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
    print(f'   {nowbytes / 1024 / 1024 / 1024 :.2f}GiB GPU memory in use, {peakbytes / 1024 / 1024 / 1024 :.2f}GiB peak')
    print(f'   {len(asizes)} live Jax arrays of total size {sum(asizes)/1024/1024/1024:.2f}GiB')
    gc_tracked_now = len(gc.get_objects())
    print(f'   {gc_tracked_now} Python objects ({gc_tracked_now - gc_tracked[0]:+d})',flush=True)
    # print('',flush=True)
    gc_tracked[0] = gc_tracked_now

if __name__ == '__main__':
    import argparse
    global debug_prints
    debug_prints = False

    ## Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--apath',type=str,dest='apath',default='../gdata_025_wb',help='Location of analysis (initial condition) data')
    parser.add_argument('--start-date',type=str,dest='start_date',default='1 Jan 2020 00:00',help='Starting date/time')
    parser.add_argument('--end-date',type=str,dest='end_date',default='31 Dec 2021 18:00',help='Ending date/time (inclusive)')
    parser.add_argument('--forecast-length',type=str,dest='forecast_length',default="1",
                        help='Length of forecast used for training.  Use "+" to separate training periods, like "4+8"')
    parser.add_argument('--to-csv',type=str,dest='csvpath',default=None,help='(optional) CSV file for scores')
    parser.add_argument('--batch-size',type=int,dest='batch_size',default=32,help='Batch size used in training')
    parser.add_argument('--batch-number',type=int,dest='train_batches',default=None,help='Number of batches to train over')
    parser.add_argument('--model-checkpoint',type=str,dest='model_checkpoint',default=None,help='Model checkpoint to load')
    parser.add_argument('--checkpoint-every',type=int,dest='checkpoint_interval',default=10,help='How often to write a new model checkpoint')
    parser.add_argument('--learning-rate',type=float,dest='learning_rate',default=1e-6,help='Learning rate for adamw')
    parser.add_argument('--debug',action='store_true',dest='debug',default=False,help='Debug printouts')
    parser.add_argument('--debug-memory',action='store_true',dest='debug_memory',default=False,help='Debug printouts (GPU memory use only)')
    parser.add_argument('--dry-run',action='store_true',dest='dry_run',default=False,help="Read and assemble data, but don't run the model")
    parser.add_argument('--log-jax-compiles',action='store_true',dest='jaxlog',default=False,help='Log jax compilations')
    parser.add_argument('--num-preload',type=int,dest='num_preload',default=None,
                        help='Maximum number of training set examples to load while waiting for forecast generation')
    parser.add_argument('--opt-checkpoint-every',type=int,dest='opt_checkpoint_interval',default=None,
                        help='How often to checkpoint the optimizer state')
    parser.add_argument('--cosine-anneal',nargs=2,type=int,dest='cosine_anneal_epochs',metavar=('warmup','total'),
                        help='Warm-up and total batches for cosine annealing')
    parser.add_argument('--cosine-anneal-end-rate',type=float,dest='cosine_anneal_end_rate',default=None,
                        help='Endpoint learning rate for cosine annealing')
    parser.add_argument('--error-weights',type=str,dest='error_weight_file',default=None,
                        help='File containing non-default variable and level weights')
    parser.add_argument('--norm-factors',type=str,dest='norm_path',default=None,
                        help='Path to the directory containing Graphcast normalization factors')


    args = parser.parse_args()

    import xarray as xr
    import numpy as np
    import numcodecs
    import trainer.dataloader
    import forecast.encabulator
    import dateparser
    import jax
    import trainer.grad_utils
    import datetime
    import dask
    import dask.distributed
    import time
    import sys

    # Disable threading inside blosc
    numcodecs.blosc.use_threads = False

    # Forecast options: forecast length and dataset paths
    forecast_lengths = [int(f) for f in args.forecast_length.split('+')]
    assert(all([f > 0 for f in forecast_lengths]))
    total_forecast_length = sum(forecast_lengths)
    apath = args.apath

    # CSV output path
    csvpath = args.csvpath

    num_preload = args.num_preload

    params_path = args.model_checkpoint
    batch_size = args.batch_size
    train_batches = args.train_batches

    numtrain = batch_size*train_batches
    learning_rate = args.learning_rate

    use_cosine_annealing = (args.cosine_anneal_epochs is not None)
    if (use_cosine_annealing):
        cosine_warmup = args.cosine_anneal_epochs[0]
        cosine_total = args.cosine_anneal_epochs[1]
        if (args.cosine_anneal_end_rate is not None):
            cosine_end_lr = args.cosine_anneal_end_rate
        else:
            cosine_end_lr = args.learning_rate / 100

    debug_prints = args.debug
    dry_run = args.dry_run
    trainer.dataloader.debug_prints = debug_prints
    debug_print_memory = args.debug_memory
    debug_print_memory_last = datetime.datetime(1970,1,1)

    checkpoint_interval = args.checkpoint_interval
    opt_checkpoint_interval = args.opt_checkpoint_interval

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

    param_path_components = params_path.split('.')
    initial_batch_number = int(param_path_components[-2])
    np.random.seed(initial_batch_number)
    param_path_components[-2] = '{batchnum:06d}'
    checkpoint_path_schema = '.'.join(param_path_components)
    print(f'Using param checkpoint schema {checkpoint_path_schema=}, {initial_batch_number=}')

    # Check that we're not trying to train for more batches than a cosine annealing period covers
    if (use_cosine_annealing):
        print(f'Using cosine annealing: {cosine_warmup} warmup batches, {cosine_total} total training batches')
        assert(train_batches + initial_batch_number <= cosine_total)

    from forecast import generate_model
    from forecast.models import models_dict
    (model_config, task_config, params) = generate_model.load_model(params_path)


    # Open database
    print(f'Using analysis database contained in {apath}')
    dbase,_ = trainer.dataloader.open_databases(apath,None) # Note no need for a separate verification dbase
    # latitude = dbase.latitude
    # longitude = dbase.longitude

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
        print(f'Using default normalization factors')
        # Otherwise do not load normalization factors, and default to the loading inside the predictor-generator
        diffs_stddev_by_level = None
        mean_by_level = None
        stddev_by_level = None

    # If using custom error weightings, build the appropriate error function
    if (error_weight_file is not None):
        print(f'Using custom error weight file {error_weight_file}')
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
        print('Using default error weights')
        custom_loss = None


    # Build operators for prediction (forecast generation), GraphCast-style loss computation (builtin,
    # averaging losses over lead times), and gradients
    predictor = generate_model.build_predictor_params(model_config,task_config,use_float16=False,
                                                      diffs_stddev_by_level = diffs_stddev_by_level, 
                                                      mean_by_level = mean_by_level,
                                                      stddev_by_level = stddev_by_level)
    # But keep float32 precision when computing losses alone
    loss_fn, _ = generate_model.build_loss_and_grad(model_config, task_config, use_float16=False,custom_loss_fn=custom_loss,
                                                      diffs_stddev_by_level = diffs_stddev_by_level, 
                                                      mean_by_level = mean_by_level,
                                                      stddev_by_level = stddev_by_level)
    # Use float16 when computing gradients
    _, grad_fn = generate_model.build_loss_and_grad(model_config, task_config, use_float16=True,custom_loss_fn=custom_loss,
                                                      diffs_stddev_by_level = diffs_stddev_by_level, 
                                                      mean_by_level = mean_by_level,
                                                      stddev_by_level = stddev_by_level)

    # Jittted function to accumulate gradients
    grad_accumulate_jit = jax.jit(grad_accumulate,static_argnums=(2,))

    dt = datetime.timedelta(hours=6)
    startdate = start_date 
    enddate = end_date 
    ndates = (enddate-startdate)//dt+1
    idx = 0
    processed = 0
    tic = datetime.datetime.now()

    # Initialize loss, gradient, and optimizer
    import optax
    import queue

    # Set up a GPU queue to hold on-device parameters and gradient accumulation arrays, allowing a grad-calculator
    # to run on an available GPU by popping from the queue
    gpu_queue = queue.Queue()
    gpu_device_0 = jax.devices('gpu')[0]
    cpu_device = jax.devices('cpu')[0]
    params_device = cpu_device # gpu_device_0
    num_gpus = len(jax.devices('gpu'))
    print(f'Running with {num_gpus} GPUs')

    scatter_params(params)
    params = jax.device_put(params,params_device)

    if (use_cosine_annealing):
        # Create the optimizer with a cosine-annealing schedule for the learning rate
        print(f'Optimizing with cosine annealing, learning rate {cosine_end_lr:.2e} - {learning_rate:.2e}, {cosine_warmup} warmup batches, and {cosine_total} total batches')
        cosine_schedule = optax.warmup_cosine_decay_schedule(cosine_end_lr, learning_rate, cosine_warmup, cosine_total, end_value=cosine_end_lr, exponent=1.0)
        optimizer = optax.adamw(learning_rate=cosine_schedule,b1=0.9,b2=0.95,weight_decay=0.1,mask=trainer.grad_utils.weight_mask(params))
    else:
        # Create the optimizer with a fixed learning rate
        print(f'Optimizing with fixed learning rate {learning_rate:.2e}')
        optimizer = optax.adamw(learning_rate=learning_rate,b1=0.9,b2=0.95,weight_decay=0.1,mask=trainer.grad_utils.weight_mask(params))

    # Check to see if an optimizer checkpoint file exists
    opt_checkpoint_file = params_path+'.opt'
    opt_checkpoint_loaded = False
    if (os.path.exists(opt_checkpoint_file)):

        print(f'Attempting to load optimizer state checkpoint {opt_checkpoint_file}')
        import pickle
        try:
            # Load the adamw momentum statistics from the pickled optimizer state file.
            # Everything else (currently the weight mask and cosine schedule counter)
            # can be re-generated easily.
            with open(opt_checkpoint_file,'rb') as ofile:
                opt_state_adamw_loaded = pickle.load(ofile)[0]
                opt_checkpoint_loaded = True
        except Exception as e:
            print(f'Optimizer state load failed, exception {e=}')

    opt_state = optimizer.init(params)
    if (opt_checkpoint_loaded):
        # Attach the adamw state to the initialized optimizer state, replacing the first
        # field of the tuple
        opt_state = (opt_state_adamw_loaded,) + opt_state[1:]
    if (use_cosine_annealing):
        # We want opt_state[2] to reflect the current batch number
        # if (not isinstance(opt_state[2],optax.ScaleByScheduleState) or opt_state[2].count != initial_batch_number):
        print(f'Reseting optimizer epoch count to {initial_batch_number} for cosine annealing')
        import jax.numpy as jnp
        opt_state = opt_state[:2] + (optax.ScaleByScheduleState(count = jnp.array([initial_batch_number,],dtype=np.int32)),)
    elif (not isinstance(opt_state[2],optax.EmptyState)):
        # Otherwise, we're using a constant learning rate, and opt_state[2] should be an EmptyState.
        # This code will probably not be executed, since we're only applying the adamw information when
        # loading from disk.
        print('Clearing optimizer learning rate state for fixed LR training')
        opt_state = opt_state[:2] + (optax.EmptyState(),)

    # Store the optimizer state on the same device as the canonical parameter copy
    opt_state = jax.device_put(opt_state,params_device)
        
    if (len(forecast_lengths)>1):
        print(f'Evaluating {numtrain} forecasts of total length {total_forecast_length*6}h',
              f'(split as {"/".join(str(6*h) for h in forecast_lengths)}h)',
              f'between {stamp(startdate)} and {stamp(enddate)}')
    else:
        print(f'Evaluating {numtrain} forecasts of total length {total_forecast_length*6}h',
              f' between {stamp(startdate)} and {stamp(enddate)}')

    # Import faulthanlder, which will act as a watchdog to dump a stacktrace in the event that things hang
    import faulthandler
    # Set faulthandler to exit the program with a traceback after 15 minutes.  The previous value of 10 minutes was not
    # long enough to accommodate the multi-stage JAX compilations for split-horizon gradient calculations.  An initial
    # 15-minutes will be re-set to 10 minutes after the first forecast has been processed.
    faulthandler.dump_traceback_later(900,exit=True)

    # Use a with-block for Dask, the threaded gradient executor, and (optionally) CSV writing
    import concurrent.futures
    import contextlib
    import logging

    import dask.config

    dask.config.set(
        {'distributed.worker.memory.target':False,
        'distributed.worker.memory.spill':False,
        #'distributed.worker.memory.pause':0.95,
        'distributed.worker.memory.terminate':0.95,}
    )

    with (dask.distributed.Client(processes=False,
                                  silence_logs=logging.ERROR
                                  ) as dask_client, 
          concurrent.futures.ThreadPoolExecutor(max_workers = num_gpus) as gpu_executor, 
          (open(csvpath,'w') if csvpath else contextlib.nullcontext()) as csvfile):

        # Prepopulate the list of training samples
        samples = []
        # possible_times contains the set of times which are present in the database and can be
        # used to initialize the forecast.  The full database itself will need times before and
        # after this (for the -6h IC and verification targets), but we shouldn't use those as T=0
        # initial conditions.
        possible_times = dbase.time.sel(time=slice(start_date,end_date)).data
        print('Populating training sample list...')
        tic = datetime.datetime.now()
        for idx in range(numtrain):
            # Use a freshly seeded random number genrator for reproducible selection
            rng = np.random.Generator(np.random.PCG64((possible_times.size,total_forecast_length,idx + batch_size*initial_batch_number)))
            samples.append(rng.choice(possible_times,1).astype('datetime64[s]').astype(datetime.datetime)[0])
        toc = datetime.datetime.now()
        print(f'... done in {(toc-tic).total_seconds():.3f}s')
            
        # Write the CSV file header
        if (csvfile):
            csvfile.write('Batch, Number, Loss\n')
            csvfile.flush()

        # Initilaize working variables before the training loop begins
        processed = 0 # How many training examples have been processed so far
        queued_inputs = {} # Dictionary of pending Futures (dask) for input loading; defined
                           # as a dictionary to include ancillary information (subsequent
                           # unrealized target/forcing data) that is not immediately computed
        #queued_inputs = [] # List of pending Futures (dask) for input loading
        queued_grads = [] # List of pending Futures (threadpool) for grad generation
        max_queued_inputs = (num_preload if num_preload is not None else num_gpus + 1)
        ready_forecasts = [] # List of ready (date,inputs,forcings,targets) tuples
        batch_processed = 0 # Number of examples processed within the current batch

        # Performance-diagnostic variables, to measure the time that the GPU is stalled
        # for lack of data
        dead_tic = datetime.datetime.now() # Timer counting non-GPU-using time
        dead_state = True # True if no GPU grad computation is in use or pending
        dead_time = 0 # Amount of non-GPU time since the last printout

        batch_tic = datetime.datetime.now() # Timer for the current batch

        while processed < numtrain:
            productive = False # Flag whether this loop iteration completed useful work

            # First, check to see if any gradient computations have finished
            (grads_done, queued_grads) = concurrent.futures.wait(queued_grads,timeout=0)
            queued_grads = list(queued_grads)
            for future in grads_done:
                processed += 1
                batchnum = initial_batch_number + processed//batch_size 
                batch_ex = processed % batch_size
                (idate, err, tictoc) = future.result()
                # Write a message about it to standard output
                print(f'Received {stamp(idate)} {err=:.3f} in {tictoc:.3f}s',
                    f', {len(queued_grads)} queued',
                    f', {len(ready_forecasts)} ready',
                    f', {len(queued_inputs)} loading', 
                    f', {dead_time:.2f}s waiting time' if dead_time > 0 else '', sep='')
                sys.stdout.flush()
                
                # Write the sample error to the CSV file
                if (csvfile is not None):
                    csvfile.write(f'{initial_batch_number + (processed-1)//batch_size}, ' + \
                                  f'{1 + ((processed-1) % batch_size)}, ' + \
                                  f'{err:.6e}\n')
                productive = True
                dead_time = 0

                # We've completed a batch
                if (processed % batch_size == 0):
                    tic = datetime.datetime.now()
                    # Consolidate the on-gpu gradient accumulators
                    accum_grad = consolidate_grad()
                    # Use them to update the paramters
                    params, opt_state = params_update(optimizer,accum_grad,opt_state,params)
                    # Scatter the updated parameters back to the GPUs
                    scatter_params(params)
                    del accum_grad
                    toc = datetime.datetime.now()
                    
                    print(f'Updated optimizer parameters in {(toc-tic).total_seconds():.2f}s (batch {batchnum}) [batch time {(toc-batch_tic).total_seconds():.2f}s]')
                    sys.stdout.flush()
                    batch_tic = toc
                    batch_processed = 0

                    if (checkpoint_interval and batchnum % checkpoint_interval == 0):
                        # Write out a new model checkpoint
                        write_checkpoint(checkpoint_path_schema,batchnum,params,model_config,task_config)
                        # Also flush the output csv file, if used
                        if (csvfile is not None):
                            csvfile.flush()
                    if (opt_checkpoint_interval and batchnum % opt_checkpoint_interval == 0):
                        # Write an optimizer checkpoint
                        write_opt_checkpoint(checkpoint_path_schema,batchnum,opt_state)

            # Next, check to see if any input loads have finished
            (inputs_done, inputs_not_done) = split_futures(queued_inputs.keys())

            # if (debug_prints and len(inputs_done) > 0):
            #     print(f'Now {len(queued_inputs)} queued inputs')
                # print(queued_inputs)

            for future in inputs_done:
                # Get data from the future object
                (inputs, forcings_first, targets_first) = future.result()
                (idate, forcings_rem, targets_rem) = queued_inputs[future]
                del queued_inputs[future]
                if (debug_prints):
                    print(f'Preparing forecast for {stamp(idate)}')
                ready_forecasts.append((idate,inputs,[forcings_first] + forcings_rem, [targets_first] + targets_rem))
                # if (debug_prints):
                #     print([stamp(r[0]) for r in ready_forecasts])
                #     print(f'{batch_processed=}')
                productive = True
                del inputs, forcings_first, forcings_rem, targets_first, targets_rem

            while (batch_processed < batch_size and len(ready_forecasts) > 0):
                # Submit a new forecast for execution, up to the batch size
                new_forecast = ready_forecasts.pop(0)
                if (debug_prints):
                    print(f'Submitting forecast for {stamp(new_forecast[0])}')
                if (not dry_run):
                    queued_grads.append(gpu_executor.submit(split_grad,*new_forecast,grad_fn,1/batch_size,predictor,dask_client))
                batch_processed += 1
                if (dry_run):
                    # If dry run (no prediction), pretend that the processing happens instantly
                    processed += 1
                    #print(f'{processed=}, {batch_processed=}, {processed % batch_size=}, {processed % batch_size == 0=}')
                    if (processed % batch_size == 0): # Also pretend the batch is done
                        batch_processed = 0
                productive = True
                del new_forecast
                if (dead_state):
                    # print('Predictions now in queue')
                    dead_state = False
                    dead_time += (datetime.datetime.now() - dead_tic).total_seconds()

            # Check to see if at least one GPU is stalled while we have samples left to process
            if (dead_state == False and \
                (len(queued_grads) + len(ready_forecasts) < num_gpus) and \
                (len(samples) + len(queued_inputs) > 0)):
                dead_state = True  
                dead_tic = datetime.datetime.now()

            # If we have samples left to process, and if we don't have more than
            # the maximum number of samples loading+ready+processing, then queue
            # the loading of more samples from disk
            if (len(samples) > 0 and len(queued_inputs) + len(queued_grads) + len(ready_forecasts) < num_gpus + max_queued_inputs):
                itic = datetime.datetime.now()
                idate = samples.pop(0)
                (inputs, forcings, targets) = trainer.dataloader.build_forecast(idate, total_forecast_length, task_config,
                                                                                model_latitude, model_longitude, input_variables, target_variables,
                                                                                dbase, dbase)
                inputs = inputs.drop_vars('datetime')
                forcings = forcings.drop_vars('datetime')
                targets = targets.drop_vars('datetime')
                (inputs, forcings, targets) = dask.optimize(inputs, forcings, targets) # Optimize before split
                (targets_split, forcings_split) = data_split(targets,forcings,forecast_lengths)
                forcings_first = forcings_split[0]
                forcings_rem = forcings_split[1:]
                targets_first = targets_split[0]
                targets_rem = targets_split[1:]
                # queued_inputs.append(dask_client.submit(tuple,dask_client.compute((inputs,forcings,targets))))
                first_data = dask_client.compute(dask.delayed(tuple)((inputs,forcings_first,targets_first)),sync=False)
                queued_inputs[first_data] = (idate,forcings_rem,targets_rem)
                itoc = datetime.datetime.now()
                if (debug_prints):
                    print(f'Queueing input for {stamp(idate)} ({(itoc-itic).total_seconds():.2f}s)',
                          f'{len(samples)} samples remain, {len(queued_inputs)} queued inputs, {len(queued_grads)} queued grads,',
                          f'{len(ready_forecasts)} ready for computation')
                    # print(queued_inputs)
                productive = True
                del inputs, forcings, targets

            # If we're printing memory stats and it's been more than five minutes
            # since the last printout, print it
            if (debug_print_memory and datetime.datetime.now() - debug_print_memory_last > datetime.timedelta(seconds=300)):
                print_memory_stats()
                sys.stdout.flush()
                debug_print_memory_last = datetime.datetime.now()

            # If nothing useful happened, sleep and allow other threads to work
            if not productive:
                time.sleep(0.01)  
            else:
                # Feed the watchdog
                if (processed > 0):
                    faulthandler.dump_traceback_later(600,exit=True)
                else:
                    faulthandler.dump_traceback_later(900,exit=True)

        # Loop finish, write out final checkpoints
        batchnum = initial_batch_number + processed // batch_size
        write_checkpoint(checkpoint_path_schema,batchnum,params,model_config,task_config)
        write_opt_checkpoint(checkpoint_path_schema,batchnum,opt_state)


    print('exiting')
