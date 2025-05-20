
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

# Testing: locking JIT to a single thread may hurt performance with >2 GPUs
# import threading
# jit_lock = threading.Lock()
# grad_jitted = False

def nancheck(pytree):
    # Check a returned gradient for any nan values
    import jax.tree
    import jax.numpy as jnp
    # Algorithm: apply "is anything a nan?" to each tree leaf, flatten it
    # to a list, convert that list to an array, then use jnp.sum to find
    # out if any leaf contained nans.  This works around trouble with 
    # jax.tree.all, which didn't seem to like compilation.
    return jnp.sum(jnp.array(jax.tree.flatten(jax.tree.map(lambda x: jnp.any(jnp.isnan(x)),
                                                                              pytree))[0]))

def grad_update(idate,inputs,forcings,targets,grad_fn,grad_weight):
    # In parallel, execute a prediction and accumulate the gradient to the on-GPU accumulator
    import datetime
    import contextlib
    global gpu_queue
    global grad_jitted
    global nancheck_jit
    (params_gpu, grad_accum_gpu, device) = gpu_queue.get()
    tic = datetime.datetime.now()
    # # Jax's JIT of GraphCast is very spammy thanks to the long compilation times.  We don't
    # # really want to repeat these messages per GPU.  If the gradient function has not yet
    # # been compiled, acquire jit_lock to serialize the compilation; everyone who waits on
    # # this lock should see a faster first-compile through reuse.
    # with (jit_lock if not grad_jitted else contextlib.nullcontext()) as _:
    new_grad = grad_fn(inputs=inputs, forcings=forcings, targets=targets, params=params_gpu)
    del inputs, forcings, targets
    # Check for and suppress any nan result, with warning
    if (nancheck_jit(new_grad[2])):
        import sys
        print(f'WARNING: nan detected in gradient for {stamp(idate)}',
              file=sys.stderr)
        # The "new" accumulated gradient is just the old one again, with no change
        new_grad_accum_gpu = grad_accum_gpu
    else:
        new_grad_accum_gpu = grad_accumulate_jit(new_grad[2],grad_accum_gpu,grad_weight)
    # grad_jitted = True
    err = np.array(new_grad[0])
    del new_grad
    gpu_queue.put( (params_gpu, new_grad_accum_gpu, device) )
    toc = datetime.datetime.now()
    return (idate, err, (toc-tic).total_seconds())

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
    return tree.map_structure(lambda gr: np.zeros(gr.shape,dtype=gr.dtype), grad)

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
    global my_gpus # Use global variable for GPU list in case of MPI processing
    global myprefix
    accum_grad = None
    gradlist = []
    newgradlist = []
    for idx in range(len(my_gpus)):
        (_,grad, device) = gpu_queue.get(timeout=0.1)
        gradlist.append((grad,device))
    assert(gpu_queue.empty())

    while len(gradlist) > 1:
        while len(gradlist) > 1:
            newgradlist = []
            (grad1,dev1) = gradlist.pop()
            (grad2,dev2) = gradlist.pop()
            grad2 = jax.device_put(grad2,dev1)
            accum_grad = grad_accumulate_jit(grad1,grad2,1.0)
            newgradlist.append((accum_grad,dev1))
        if len(gradlist) > 0:
            (grad_odd,dev_odd) = gradlist[0]
            newgradlist.append((grad_odd,dev_odd))
        gradlist = newgradlist
    (accum_grad,_) = gradlist[0]
    accum_grad = jax.device_put(accum_grad,params_device)

    # for idx in range(len(jax.devices('gpu'))):
    #     (params, grad) = gpu_queue.get(timeout=0.1)
    #     grad = jax.device_put(grad,params_device)
    #     if (accum_grad is None):
    #         accum_grad = grad
    #     else:
    #         accum_grad = grad_accumulate_jit(accum_grad,grad,1.0)
    return accum_grad

def scatter_params(params):
    # Scatter the parameters to each GPU device, posting the paramters
    # and an initialized (zero) gradient accumulator to the device queue
    global gpu_queue
    global zero_grad_like_jit
    import jax
    global my_gpus # Use global variable for GPU list in case of MPI processing
    for device in my_gpus:
        params_gpu = jax.device_put(params,device)
        accum_gpu = zero_grad_like_jit(params_gpu)
        gpu_queue.put( (params_gpu, accum_gpu, device) )

def params_update(optimizer,accum_grad,opt_state,params):
    # Use a provided optax updater to update paramters, returning
    # the new paramters and the updated optimizer state
    import optax
    updates, opt_state = optimizer.update(accum_grad,opt_state,params)
    params = optax.apply_updates(params,updates)
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

## Functions to flatten and unflatten parameter-like arrays, for MPI communication

def params_to_array(flat_params):
    # Given a flattened set of parameters (with dictionary nesting removed), concatenate
    # all the matrices into a flat, 1D array
    import jax.numpy as jnp
    import jax.tree_util
    return jnp.concatenate(jax.tree_util.tree_map(lambda x: x.ravel(),flat_params))

def get_param_structure(params):
    # Take a parameter set and return structure information necessary to reconstruct
    # a parameter matrix from a flattened, concatenated version
    # import jax.numpy as jnp
    import numpy as np
    import jax.tree_util
    flat_params, flat_treedef = jax.tree_util.tree_flatten(params)
    flat_sizes = jax.tree_util.tree_map(lambda x: x.size,flat_params)
    flat_shapes = jax.tree_util.tree_map(lambda x: x.shape, flat_params)

    param_offsets = tuple(int(p) for p in np.cumsum(flat_sizes)[:-1])
    # params_as_array = params_to_array(flat_params)
    return(param_offsets,tuple(flat_shapes),flat_treedef)

def rebuild_params(params_as_array,param_offsets,param_shapes,param_treedef):
    # Given a 1D concatenation of parameter values and structure information,
    # rebuild the nested model parameters
    import jax.numpy as jnp
    import jax.tree_util
    
    # import jax.tree
    # Split the unified array into segments
    segments = jnp.split(params_as_array,param_offsets)
    # Reconstitute the flat segments into shaped parameters
    matrices = jax.tree_util.tree_map(lambda p, sh : p.reshape(sh), segments, list(param_shapes))
    # Rebuild the nested tree from matrices
    return jax.tree_util.tree_unflatten(param_treedef,matrices)

if __name__ == '__main__':
    import argparse

    ## Command-line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--mpi',action='store_true',dest='use_mpi',default=False,
                        help='Parallelize training via MPI')

    data_group = parser.add_argument_group('Data options')
    data_group.add_argument('--apath',type=str,dest='apath',default='../gdata_025_wb',help='Location of analysis (initial condition) data')
    data_group.add_argument('--start-date',type=str,dest='start_date',default='1 Jan 2020 00:00',help='Starting date/time')
    data_group.add_argument('--end-date',type=str,dest='end_date',default='31 Dec 2021 18:00',help='Ending date/time (inclusive)')
    data_group.add_argument('--num-preload',type=int,dest='num_preload',default=None,
                        help='Maximum number of training set examples to load while waiting for forecast generation')

    forecast_group = parser.add_argument_group('Forecast options')
    forecast_group.add_argument('--forecast-length',type=int,dest='forecast_length',default=1,help='Forecast length for training, in # of steps')
    forecast_group.add_argument('--batch-size',type=int,dest='batch_size',default=32,help='Batch size used in training')
    forecast_group.add_argument('--batch-number',type=int,dest='train_batches',default=None,help='Number of batches to train over')
    forecast_group.add_argument('--model-checkpoint',type=str,dest='model_checkpoint',default=None,help='Model checkpoint to load')
    forecast_group.add_argument('--norm-factors',type=str,dest='norm_path',default=None,
                        help='Path to the directory containing Graphcast normalization factors')
    forecast_group.add_argument('--seed',type=int,dest='seed',default=0,
                        help='Random seed factor')

    learning_group = parser.add_argument_group('Learning options')
    learning_group.add_argument('--learning-rate',type=float,dest='learning_rate',default=1e-6,help='Default/starting learning rate')
    learning_group.add_argument('--cosine-anneal',nargs=2,type=int,dest='cosine_anneal_epochs',metavar=('warmup','total'),
                        help='Warm-up and total batches for cosine annealing')
    learning_group.add_argument('--cosine-anneal-end-rate',type=float,dest='cosine_anneal_end_rate',default=None,
                        help='Endpoint learning rate for cosine annealing')

    output_group = parser.add_argument_group('Output options')
    output_group.add_argument('--to-csv',type=str,dest='csvpath',default=None,help='(optional) CSV file for scores')
    output_group.add_argument('--checkpoint-every',type=int,dest='checkpoint_interval',default=10,help='How often to write a new model checkpoint')
    output_group.add_argument('--opt-checkpoint-every',type=int,dest='opt_checkpoint_interval',default=None,
                        help='How often to checkpoint the optimizer state')
    
    error_group = parser.add_argument_group('Error options')
    import trainer.loss_utils

    trainer.loss_utils.add_error_args(error_group)
    
    
    # error_group.add_argument('--error-weights',type=str,dest='error_weight_file',default=None,
    #                     help='File containing non-default variable and level weights')
    # error_group.add_argument('--wind-speed',action='store_true',dest='wind_speed',default=False,
    #                     help='Add wind speed variable to loss function')
    # error_group.add_argument('--time-bias',action='store_true',dest='time_bias',default=False,
    #                     help='Add time-averaged term to loss function')
    # error_group.add_argument('--mean-bias',action='store_true',dest='mean_bias',default=False,
    #                     help='Add global mean bias term to loss function')
    
    
    debug_group = parser.add_argument_group('Debug options')
    # debug_group.add_argument('--log-jax-compiles',action='store_true',dest='jaxlog',default=False,help='Log jax compilations')
    debug_group.add_argument('--debug',action='store_true',dest='debug',default=False,help='Debug printouts')
    debug_group.add_argument('--debug-memory',action='store_true',dest='debug_memory',default=False,help='Debug printouts (GPU memory use only)')
    debug_group.add_argument('--dry-run',action='store_true',dest='dry_run',default=False,help="Read and assemble data, but don't run the model")
    debug_group.add_argument('--compile-params',action='store_true',dest='compile_params',default=False,help='Compile the params_update function (testing)')


    args = parser.parse_args()
    trainer.loss_utils.parse_arguments(args)

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

    # Disable threading inside blosc
    numcodecs.blosc.use_threads = False

    # Forecast options: forecast length and dataset paths
    forecast_length = args.forecast_length
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

    global compute_wind_speed
    compute_wind_speed = args.wind_speed
    compute_time_bias = args.time_bias
    compute_mean_bias = args.mean_bias

    # File for user-specified level/variable error weights
    error_weight_file = args.error_weight_file

    import sys
    use_mpi = args.use_mpi
    global myprefix # Prefix for standard output
    global my_gpus # List of locally available GPUs after MPI split
    if (use_mpi):
        # Using MPI: adjust parameters so that the full batch is split over N processes
        from mpi4py import MPI
        import jax

        # Step 1 is to detect which GPUs are to be assigned to this process.  From testing,
        # it appears that PBS will use a shared environment for processes launched on the
        # same system, which means for example that a one-host, two-process job will have
        # both processes see both GPUs.  This is not great.
        worldcomm = MPI.COMM_WORLD
        hostcomm = worldcomm.Split_type(MPI.COMM_TYPE_SHARED,worldcomm.rank)

        master_process = (worldcomm.rank == 0)
        if (master_process):
            print('Using MPI parallelism')

        if (debug_prints):
            print(f'Process {worldcomm.rank+1} is process {hostcomm.rank+1} on its host')

        all_gpus = jax.devices('gpu')
        my_gpus = all_gpus[ (len(all_gpus)*hostcomm.rank) // hostcomm.size : (len(all_gpus)*(hostcomm.rank+1))//hostcomm.size]
        if (debug_prints):
            print(f'Process {worldcomm.rank+1} using {my_gpus}')

        # Share the GPU information among all processes in order to divide up the work of a batch
        gpu_num_list = np.zeros((worldcomm.size),dtype=np.int32)
        gpu_num_send = np.array(len(my_gpus),dtype=np.int32)
        worldcomm.Allgather(gpu_num_send,gpu_num_list)
        cumulative_gpu = np.cumsum(gpu_num_list)

        # Assert all processes have at least one GPU
        assert(np.all(gpu_num_list > 0))
        
        # Find out where this process's GPUs place among all GPUs
        total_gpus = cumulative_gpu[-1]
        my_shard_begin = cumulative_gpu[worldcomm.rank] - len(my_gpus)
        my_shard_end = cumulative_gpu[worldcomm.rank]

        # Assert that the batch is large enough to be split amongst all GPUs
        assert(batch_size >= total_gpus)

        local_batch_start = (batch_size * my_shard_begin) // total_gpus
        local_batch_end = (batch_size * my_shard_end) // total_gpus # noninclusive
        local_batch_size = local_batch_end - local_batch_start
        myrank = worldcomm.rank
        mysize = worldcomm.size
        myprefix = f'({myrank+1}/{mysize}) '
        local_numtrain = local_batch_size*train_batches

        print(f'Process {worldcomm.rank+1} to evaluate batch members {local_batch_start}-{local_batch_end-1} ({local_batch_size} of {batch_size} total)')
        # sys.exit(1)
    else:
        print('MPI not being used')
        # Set local batch parameters to match the global versions
        local_batch_size = batch_size
        local_batch_start = 0
        local_batch_end = batch_size
        local_numtrain = numtrain
        master_process = True
        my_gpus = jax.devices('gpu')
        myrank = 0
        mysize = 1
        myprefix=''
        # sys.exit(1)

    # Subsequent processing continues almost identically whether or not the job is using MPI parallelism.  We assume
    # that each process will conduct its own I/O to read the model parameters and training data

    # Model parameters and checkpoint schema
    param_path_components = params_path.split('.')
    initial_batch_number = int(param_path_components[-2])
    np.random.seed(initial_batch_number)
    param_path_components[-2] = '{batchnum:06d}'
    checkpoint_path_schema = '.'.join(param_path_components)
    if (master_process):
        print(f'Using param checkpoint schema {checkpoint_path_schema=}, {initial_batch_number=}')

    # Check that we're not trying to train for more batches than a cosine annealing period covers
    if (use_cosine_annealing):
        if (master_process):
            print(f'Using cosine annealing: {cosine_warmup} warmup batches, {cosine_total} total training batches')
        assert(train_batches + initial_batch_number <= cosine_total)

    from forecast import generate_model
    from forecast.models import models_dict
    (model_config, task_config, params) = generate_model.load_model(params_path)


    # Initialize loss, gradient, and optimizer
    import optax
    import queue

    # Set up a GPU queue to hold on-device parameters and gradient accumulation arrays, allowing a grad-calculator
    # to run on an available GPU by popping from the queue.

    # To support MPI parallelism, the list of GPU devices is taken from my_gpus rather than directly from jax.devices('gpu'),
    # which lets multiple processors on a single host play nice even if they see a shared set of GPUs.
    gpu_queue = queue.Queue()
    gpu_device_0 = my_gpus[0]
    # Explicitly set the Jax default device to one of the process's assigned GPUs.
    # Otherwise, in a multi-process configuration where several processes see the same set of GPUs
    # and should balance among themselves, the default device might not be the same as the intended
    # executiond evice.
    jax.config.update("jax_default_device",gpu_device_0)
    cpu_device = jax.devices('cpu')[0]
    params_device = cpu_device
    num_gpus = len(my_gpus) # len(jax.devices('gpu'))
    # print(f'Running with {num_gpus} GPUs')

    if (use_mpi):
        # With MPI parallelism, gradients will be accumulated onto the master process, parameter
        # updates computed there, and revised parametesr will be broadcast back to the child
        # processes.  This requires information on the structure of the parameter array
        (param_offsets, param_shapes, param_treedef) = get_param_structure(params)
        params_to_array_jit = jax.jit(params_to_array)
        rebuild_params_jit = jax.jit(lambda x : rebuild_params(x,param_offsets,param_shapes,param_treedef))

    # Open database
    if (master_process):
        print(f'Using analysis database contained in {apath}')
    tic = datetime.datetime.now()
    dbase,_ = trainer.dataloader.open_databases(apath,None) # Note no need for a separate verification dbase
    toc = datetime.datetime.now()
    if (master_process):
        print(f'   Database opened in {(toc-tic).total_seconds():.2f}s')
    # latitude = dbase.latitude
    # longitude = dbase.longitude

    if (use_mpi):
        sys.stdout.flush()
        worldcomm.Barrier()


    # Generate latitude and longitude for the model, based on its resolution
    model_latitude, model_longitude = generate_model.get_model_coords(model_config)

    input_variables = list(task_config['input_variables'])
    target_variables = list(task_config['target_variables'])

    norm_path = args.norm_path

    if (norm_path is not None):
        if (master_process):
            print(f'Using normalization factors in {norm_path}')
        # Load provided normalization factors
        diffs_stddev_by_level = xr.load_dataset(f"{norm_path}/diffs_stddev_by_level.nc").compute()
        mean_by_level = xr.load_dataset(f"{norm_path}/mean_by_level.nc").compute()
        stddev_by_level = xr.load_dataset(f"{norm_path}/stddev_by_level.nc").compute()
    else:
        if (master_process):
            print(f'Using default normalization factors')
        # Otherwise do not load normalization factors, and default to the loading inside the predictor-generator
        diffs_stddev_by_level = None
        mean_by_level = None
        stddev_by_level = None

    custom_loss = trainer.loss_utils.make_loss_new(model_config,task_config,
                                     diffs_stddev_by_level,mean_by_level,stddev_by_level,
                                     not master_process)

    # Build operators for prediction (forecast generation), GraphCast-style loss computation (builtin,
    # averaging losses over lead times), and gradients
    predictor = generate_model.build_predictor_params(model_config,task_config,use_float16=True,
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
    
    # If dry run, build trivial predictor functions
    if (dry_run):
        def predictor(inputs,forcings,targets,params):
            return targets
        def loss_fn(inputs,forcings,targets,params):
            return (0.0,None)
        import jax
        grad_loss = jax.value_and_grad(loss_fn,argnums=3,has_aux=True)
        def grad_fn(inputs,forcings,targets,params):
            import jax
            ((val,aux), grad) = grad_loss(inputs,forcings,targets,params)
            return (val, aux, grad)
        grad_fn = jax.jit(grad_fn)

    # Jittted function to accumulate gradients
    grad_accumulate_jit = jax.jit(grad_accumulate,static_argnums=(2,))
    zero_grad_like_jit =jax.jit(zero_grad_like)
    nancheck_jit = jax.jit(nancheck)

    if (nancheck_jit(params)):
        raise ValueError('Parameters contain a nan!')

    if (args.compile_params):
        params_update = jax.jit(params_update,static_argnums=(0,))

    dt = datetime.timedelta(hours=6)
    startdate = start_date 
    enddate = end_date 
    ndates = (enddate-startdate)//dt+1
    idx = 0
    processed = 0
    tic = datetime.datetime.now()

    scatter_params(params)
    params = jax.device_put(params,params_device)

    if (use_cosine_annealing):
        # Create the optimizer with a cosine-annealing schedule for the learning rate
        if (master_process):
            print(f'Optimizing with cosine annealing, learning rate {cosine_end_lr:.2e} - {learning_rate:.2e}, {cosine_warmup} warmup batches, and {cosine_total} total batches')
        cosine_schedule = optax.warmup_cosine_decay_schedule(cosine_end_lr, learning_rate, cosine_warmup, cosine_total, end_value=cosine_end_lr, exponent=1.0)
        optimizer = optax.adamw(learning_rate=cosine_schedule,b1=0.9,b2=0.95,weight_decay=0.1,mask=trainer.grad_utils.weight_mask(params))
    else:
        # Create the optimizer with a fixed learning rate
        if (master_process):
            print(f'Optimizing with fixed learning rate {learning_rate:.2e}')
        optimizer = optax.adamw(learning_rate=learning_rate,b1=0.9,b2=0.95,weight_decay=0.1,mask=trainer.grad_utils.weight_mask(params))

    # Check to see if an optimizer checkpoint file exists
    opt_checkpoint_file = params_path+'.opt'
    opt_checkpoint_loaded = False
    if (os.path.exists(opt_checkpoint_file)):

        if (master_process):
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
            print(f'{myprefix}Optimizer state load failed, exception {e=}')

    opt_state = optimizer.init(params)
    if (opt_checkpoint_loaded):
        # Attach the adamw state to the initialized optimizer state, replacing the first
        # field of the tuple
        opt_state = (opt_state_adamw_loaded,) + opt_state[1:]
    if (use_cosine_annealing):
        # We want opt_state[2] to reflect the current batch number
        # if (not isinstance(opt_state[2],optax.ScaleByScheduleState) or opt_state[2].count != initial_batch_number):
        if (master_process):
            print(f'Reseting optimizer epoch count to {initial_batch_number} for cosine annealing')
        import jax.numpy as jnp
        opt_state = opt_state[:2] + (optax.ScaleByScheduleState(count = jnp.array([initial_batch_number,],dtype=np.int32)),)
    elif (not isinstance(opt_state[2],optax.EmptyState)):
        # Otherwise, we're using a constant learning rate, and opt_state[2] should be an EmptyState.
        # This code will probably not be executed, since we're only applying the adamw information when
        # loading from disk.
        if (master_process):
            print('Clearing optimizer learning rate state for fixed LR training')
        opt_state = opt_state[:2] + (optax.EmptyState(),)

    # Store the optimizer state on the same device as the canonical parameter copy
    opt_state = jax.device_put(opt_state,params_device)
        
    if (master_process):
        print(f'Evaluating {numtrain} forecasts of length {forecast_length*6}h between {startdate.strftime("%Y-%m-%d %Hz")} and {enddate.strftime("%Y-%m-%d %Hz")}')

    # Import faulthanlder, which will act as a watchdog to dump a stacktrace in the event that things hang
    import faulthandler
    # Set the traceback to dump after 10 minutes, which should be long enough to accommodate any jax compilations
    # Update: spectral amse seems to make the first compilation take quite a bit longer, so extend this timeout to 15 minutes
    faulthandler.dump_traceback_later(900,exit=True)

    # Use a with-block for Dask, the threaded gradient executor, and (optionally) CSV writing
    import concurrent.futures
    import contextlib
    import logging

    import dask.config

    dask.config.set(
        {'distributed.worker.memory.target':False,
        'distributed.worker.memory.spill':False,
        'distributed.worker.memory.pause':0.95,
        'distributed.worker.memory.terminate':0.95,}
    )
    if (use_mpi):
        sys.stdout.flush()
        worldcomm.Barrier()
    
    if (use_mpi and csvpath):
        # Rewrite the CSV filename for multi-process execution, to create one file
        # per process
        csvparts = csvpath.split('.')
        csv_new_parts = csvparts[:-1] + [f'{myrank+1}'] + csvparts[-1:]
        csvpath = '.'.join(csv_new_parts)

    with (dask.distributed.Client(processes=False,
                                  silence_logs=logging.ERROR
                                  ) as dask_client, 
          concurrent.futures.ThreadPoolExecutor(max_workers = num_gpus) as gpu_executor, 
          (open(csvpath,'w') if csvpath else contextlib.nullcontext()) as csvfile):

        # Prepopulate the list of training samples
        # samples = []
        # possible_times contains the set of times which are present in the database and can be
        # used to initialize the forecast.  The full database itself will need times before and
        # after this (for the -6h IC and verification targets), but we shouldn't use those as T=0
        # initial conditions.
        possible_times = dbase.time.sel(time=slice(start_date,end_date)).data
        if (master_process):
            print('Populating training sample list...')

        # Compute the maximum sample index that may be asked for, as the maximum of
        # either (initial sample + to_train_here) and (cosine_total*batch).  This will
        # allow us to generate all training samples up front and then select the ones
        # to be used for this invocation, which in turn gives us repeatable sample selection
        # between jobs. 
        maximum_sample = numtrain+initial_batch_number*batch_size
        if (use_cosine_annealing):
            maximum_sample = max(maximum_sample,cosine_total*batch_size)
        initial_sample = initial_batch_number*batch_size
        tic = datetime.datetime.now()

        # Use a freshly seeded random number genrator for reproducible selection
        rng = np.random.Generator(np.random.PCG64((possible_times.size,forecast_length,args.seed)))
        samples_arr = rng.choice(possible_times,maximum_sample) # Generate all samples
        samples_arr = samples_arr[initial_sample:(initial_sample+numtrain)] # Subset to the ones used here)

        # Subset the samples according to the local batch distribution
        samples = samples_arr.reshape((-1,batch_size))
        samples = samples[:,local_batch_start:local_batch_end]
        samples = samples.reshape((-1,))        
        samples = list(samples.astype('datetime64[s]').astype(datetime.datetime)) # Make a list
        del samples_arr

        toc = datetime.datetime.now()
        print(f'{myprefix}{len(samples)} samples for processing')
        assert(local_numtrain == len(samples))

        if (use_mpi):
            sys.stdout.flush()
            worldcomm.Barrier()
        if (master_process):
            print(f'... done in {(toc-tic).total_seconds():.3f}s')
            
        # Write the CSV file header
        if (csvfile):
            csvfile.write('Batch, Date, Number, Loss\n')
            csvfile.flush()

        # Initilaize working variables before the training loop begins
        processed = 0 # How many training examples have been processed so far
        queued_inputs = [] # List of pending Futures (dask) for input loading
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

        while processed < local_numtrain:
            productive = False # Flag whether this loop iteration completed useful work

            # First, check to see if any gradient computations have finished
            (grads_done, queued_grads) = concurrent.futures.wait(queued_grads,timeout=0)
            queued_grads = list(queued_grads)
            for future in grads_done:
                processed += 1
                batchnum = initial_batch_number + processed//local_batch_size 
                batch_ex = processed % local_batch_size
                (idate, err, tictoc) = future.result()
                # Write a message about it to standard output
                print(f'{myprefix}Received {stamp(idate)} {err=:.3f} in {tictoc:.3f}s',
                    f', {len(queued_grads)} queued',
                    f', {len(ready_forecasts)} ready',
                    f', {len(queued_inputs)} loading', 
                    f', {dead_time:.2f}s waiting time' if dead_time > 0 else '', sep='',flush=True)
                
                # Write the sample error to the CSV file
                if (csvfile is not None):
                    csvfile.write(f'{initial_batch_number + (processed-1)//local_batch_size}, ' + \
                                  f'{stamp(idate)}, ' + \
                                  f'{1 + (local_batch_start + ((processed-1) % local_batch_size))}, ' + \
                                  f'{err:.6e}\n')
                productive = True
                dead_time = 0

                # We've completed a batch
                if (processed % local_batch_size == 0):
                    tic = datetime.datetime.now()
                    # Consolidate the on-gpu gradient accumulators
                    accum_grad = consolidate_grad()

                    toc1 = datetime.datetime.now()
                    if (debug_prints):
                        print(f'{myprefix}  consolidate_grad in {(toc1-tic).total_seconds():.2f}s')

                    if (use_mpi):
                        import jax.tree_util
                        # Flatten the gradient to remove its nested dictionary structure.  The gradient shares
                        # its structure with the parameters
                        flat_grad, _ = jax.tree_util.tree_flatten(accum_grad)
                        grad_vals = np.array(params_to_array_jit(flat_grad),dtype=np.float32)

                        # Use MPI reduce to sum the gradient values
                        if (master_process):
                            received_grad = np.empty_like(grad_vals)
                            worldcomm.Reduce(grad_vals,received_grad,op=MPI.SUM,root=0)
                            # Broadcast the summed parameters back out to the full structure
                            with (jax.default_device(params_device)):
                                accum_grad = rebuild_params_jit(received_grad)
                        else:
                            worldcomm.Reduce(grad_vals,None,op=MPI.SUM,root=0)
                    
                    if (master_process):
                        # Use them to update the paramters
                        params, opt_state = params_update(optimizer,accum_grad,opt_state,params)
                        toc2 = datetime.datetime.now()
                        if (debug_prints):
                            print(f'{myprefix}  params_update in {(toc2-toc1).total_seconds():.2f}s')
                    else:
                        # Adjust debug timer, since no parameter update happens on child processes
                        toc2 = toc1
                    
                    if (use_mpi):
                        # With parameters computed, broadcast them from the master process
                        if (master_process):
                            param_vals = np.array(params_to_array_jit(jax.tree_util.tree_flatten(params)[0]),dtype=np.float32)
                            worldcomm.Bcast(param_vals,0)
                        else:
                            # Receive new parameters from the master process
                            param_vals = np.empty_like(grad_vals)
                            worldcomm.Bcast(param_vals,0)
                            # Reconstruct them into the nested dictionary form
                            with (jax.default_device(params_device)):
                                params = rebuild_params_jit(param_vals)
                        
                        # Delete the concatenated arrays used for communication
                        del param_vals, grad_vals

                    # Scatter the updated parameters back to the GPUs
                    scatter_params(params)
                    del accum_grad
                    toc = datetime.datetime.now()
                    if (debug_prints):
                        print(f'{myprefix}  scatter_params in {(toc-toc2).total_seconds():.2f}s')

                    if (use_mpi):
                        sys.stdout.flush()
                        worldcomm.Barrier()
                    
                    if (master_process):
                        print(f'{myprefix}Updated optimizer parameters in {(toc-tic).total_seconds():.2f}s (batch {batchnum}) [batch time {(toc-batch_tic).total_seconds():.2f}s]')
                    batch_tic = toc
                    batch_processed = 0

                    if (checkpoint_interval and batchnum % checkpoint_interval == 0):
                        # Write out a new model checkpoint
                        if (master_process):
                            write_checkpoint(checkpoint_path_schema,batchnum,params,model_config,task_config)
                        # Also flush the output csv file, if used
                        if (csvfile is not None):
                            csvfile.flush()
                    if (master_process and opt_checkpoint_interval and batchnum % opt_checkpoint_interval == 0):
                        # Write an optimizer checkpoint
                        write_opt_checkpoint(checkpoint_path_schema,batchnum,opt_state)

            # Next, check to see if any input loads have finished
            (inputs_done, queued_inputs) = split_futures(queued_inputs)
            queued_inputs = list(queued_inputs)

            for future in inputs_done:
                # Get data from the future object
                (inputs, forcings, targets) = future.result()
                # Infer the analysis date
                idate = np.datetime64(inputs.datetime.data[-1],'s').astype(datetime.datetime)
                # print(f'Preparing forecast for {stamp(idate)}')
                inputs = inputs.drop_vars('datetime')
                forcings = forcings.drop_vars('datetime')
                targets = targets.drop_vars('datetime')
                ready_forecasts.append((idate,inputs,forcings,targets))
                productive = True
                del inputs, forcings, targets

            while (batch_processed < local_batch_size and len(ready_forecasts) > 0):
                # Submit a new forecast for execution, up to the batch size
                new_forecast = ready_forecasts.pop(0)
                # print(f'Submitting forecast for {stamp(new_forecast[0])}')
                queued_grads.append(gpu_executor.submit(grad_update,*new_forecast,grad_fn,1/batch_size))
                productive = True
                del new_forecast
                batch_processed += 1
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
            while (len(samples) > 0 and len(queued_inputs) + len(queued_grads) + len(ready_forecasts) < num_gpus + max_queued_inputs):
                itic = datetime.datetime.now()
                idate = samples.pop(0)
                (inputs, forcings, targets) = trainer.dataloader.build_forecast(idate, forecast_length, task_config,
                                                                                model_latitude, model_longitude, input_variables, target_variables,
                                                                                dbase, dbase)
                (inputs, forcings, targets) = dask.optimize(inputs, forcings, targets)
                # if (not dry_run):
                queued_inputs.append(dask_client.submit(tuple,dask_client.compute((inputs,forcings,targets))))
                itoc = datetime.datetime.now()
                # print(f'Queueing input for {stamp(idate)} ({(itoc-itic).total_seconds():.2f}s)')
                productive = True
                del inputs, forcings, targets


            # If nothing useful happened, sleep and allow other threads to work
            if not productive:
                time.sleep(0.01)  
            else:
                # Feed the watchdog
                if (processed == 0):
                    # If we haven't completed any steps yet, set a long timeout because
                    # we might still be compiling
                    faulthandler.dump_traceback_later(600,exit=True)
                else:
                    # Otherwise, set a shorter timeout to catch GPU hangs with a minimum
                    # delay
                    faulthandler.dump_traceback_later(180,exit=True)

        # Loop finish, write out final checkpoints
        if (use_mpi):
            print(f'{myprefix}Finished')
            worldcomm.Barrier()
        if (master_process):
            batchnum = initial_batch_number + processed // local_batch_size
            write_checkpoint(checkpoint_path_schema,batchnum,params,model_config,task_config)
            write_opt_checkpoint(checkpoint_path_schema,batchnum,opt_state)
        print(f'{myprefix}Exiting')
        if (csvfile is not None):
            csvfile.flush()


    print('exiting')
