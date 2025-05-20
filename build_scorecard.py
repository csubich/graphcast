# "Scorecard" style bulk forecaster

## Helper class to store a forecast sequence, noting its initialization date and the current
## set of inputs (t-6h, t0), used to initialize a new prediction
import dataclasses
from typing import Optional
import numpy as np
import xarray as xr


def unwrap_ds(in_ds):
    '''"Unwrap" a dataset, taking its data out of GPU memory.  This acts by creating a new dataset, copying the coordinates
    while redefining the data variables; the old on-GPU dataset can now fall out of scope.'''
    import xarray as xr
    from graphcast import xarray_jax

    return xr.Dataset( {var : (in_ds[var].dims, xarray_jax.unwrap_data(in_ds[var])) for var in in_ds}, coords=in_ds.coords)

def wrap_dataset(ds,device):
    import graphcast.xarray_jax
    import jax
    return (graphcast.xarray_jax.Dataset(coords=ds.coords,
                                        data_vars = {k : (ds[k].dims,jax.device_put(graphcast.xarray_jax.unwrap_data(ds[k]),device=device)) for k in ds.data_vars}))

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

def new_inputs(targets_last,targets_now,forcings_last,forcings_now,static_vars):
    import xarray as xr
    import numpy as np
    global input_from_target
    global input_from_forcing

    foo_last = xr.merge((targets_last[input_from_target],forcings_last[input_from_forcing],static_vars))
    foo_last = foo_last.assign_coords(time=[np.timedelta64(-6,'h').astype('timedelta64[ns]')])
    foo_now =  xr.merge((targets_now[input_from_target],forcings_now[input_from_forcing]))
    foo_now = foo_now.assign_coords(time=[np.timedelta64(0,'h').astype('timedelta64[ns]')])
    foo = xr.concat((foo_last,foo_now),dim='time',coords='minimal',data_vars='minimal')

    return foo

def load_climato(targets,climato_dbase,target_variables):
    global model_latitude, model_longitude
    target_datetime = targets.datetime[-1].data
    target_dayofyear = 1+(target_datetime - np.datetime64(target_datetime,'Y'))//np.timedelta64(1,'D')
    target_hour = (target_datetime - np.datetime64(target_datetime,'D'))//np.timedelta64(1,'h')
    climato_now = climato_dbase.sel(dayofyear=target_dayofyear,hour=target_hour,drop=True)[target_variables]
    # If model lat/lon don't conform to the climato variable, take the correct subset
    if (model_latitude.size != climato_now.latitude.size or \
        model_longitude.size != climato_now.longitude.size):
        climato_now = climato_now.sel(latitude=model_latitude, longitude=model_longitude)
    climato_now = climato_now.rename(latitude='lat',longitude='lon')
    return climato_now

gpu_time = 0
def advance_forecast(idate,device_queue,inputs_cpu,targets_cpu,tspec_cpu,forcings_cpu,climato_cpu,predictor,climato_level_idx,mask_hurr_reg):
    # Get paramters from device_data dictionary 
    import datetime
    global gpu_time
    device_data = device_queue.get()
    tic = datetime.datetime.now()
    gpu_device = device_data['device'] # GPU device being targeted
    params = device_data['params'] # Parameters (broadcast to GPU)
    targets_gpu = device_data['targets'] # Targets if on GPU, else None
    forcings_gpu = device_data['forcings'] # Forcings if on GPU, else None
    climato_gpu = device_data['climato'] # Climato array if on GPU, else None
    tspec_gpu = device_data['tspec'] # Spherical harmonic of targets if on-GPU, else None

    # Push data to GPU if not already present
    inputs_gpu = wrap_dataset(inputs_cpu,gpu_device)
    if targets_gpu is None: targets_gpu = wrap_dataset(targets_cpu,gpu_device)
    if forcings_gpu is None: forcings_gpu = wrap_dataset(forcings_cpu,gpu_device)
    if climato_gpu is None: climato_gpu = wrap_dataset(climato_cpu,gpu_device)
    if tspec_gpu is None and tspec_cpu is not None: tspec_gpu = wrap_dataset(tspec_cpu,gpu_device)

    # Perform one-step prediction
    assert(forcings_gpu.time.size == 1)
    prediction_gpu = predictor(inputs=inputs_gpu,forcings=forcings_gpu,targets=targets_gpu,params=params)

    # Stack the current inputs and the prediction to give next-step inputs
    inputs_next = stack_inputs(inputs_gpu,prediction_gpu,forcings_gpu)

    # add speed variables
    from trainer.loss_utils import derived_variables
    prediction_gpu2 = derived_variables(prediction_gpu, compute_wind_speed = True )
    targets_gpu2    = derived_variables(targets_gpu,    compute_wind_speed = True )
    climato_gpu2    = derived_variables(climato_gpu,    compute_wind_speed = True )

    # If provided target spectrum information, compute power spectral density and coherence of
    # prediction versus target
    if (tspec_gpu is not None):
        this_speccard = speccard_jit(prediction_gpu2.isel(time=0,batch=0,drop=True),
                                     tspec_gpu)
        pass
    else:
        this_speccard = None

    del prediction_gpu

    # Compute the scorecard for the current prediction
    scorecard = scorecard_jit(prediction_gpu2.isel(time=0,batch=0,drop=True), mask_hurr_reg,
                              targets_gpu2.isel(time=0,batch=0,drop=True),
                              climato_gpu2,
                              tuple(climato_level_idx))

    del prediction_gpu2, targets_gpu2, climato_gpu2
    
    # Repopulate the device_data dictionary, including the on-device targets/forcings/climato
    out_device_data = {'device' : gpu_device, 'params' : params,
                       'targets' : targets_gpu, 'tspec' : tspec_gpu,
                       'forcings' : forcings_gpu,
                       'climato' : climato_gpu}
    
    # Push the device data back to the GPU queue
    device_queue.put(out_device_data)
    toc = datetime.datetime.now()

    # Move next-input and scorecard data off-GPU
    inputs_next = unwrap_ds(inputs_next)
    scorecard = {key : unwrap_ds(scorecard[key]) for key in scorecard.keys()}
    if (tspec_gpu is not None):
        this_speccard = {key : unwrap_ds(this_speccard[key]) for key in this_speccard.keys()}

    gpu_time += (toc-tic).total_seconds()


    return (idate, inputs_next, scorecard, this_speccard)

# Helper data movement functions
def slice_target(target_future):
    # From a target (analysis) dataset to be returned from a 'future' object, select only the levels appropriate to 
    # the climatology.  Also wrap it as a set of on-cpu Jax arrays, for later computation via JIT
    return wrap_dataset(target_future.result().isel(level=climato_level_idx,time=-1,batch=-1),cpu_device)

def cpuwrap_future(ds):
    # Wrap a generic future-returned dataset as on-cpu Jax arrays (used for climatology)
    return wrap_dataset(ds.result(),cpu_device)

def sht_ds(ds_in,sht_forward):
    # Given an input prediction (or analysis) as an XArray dataset and the forward transform function, 
    # compute the spherical harmonic transform and return a new dataset.  Suitable for JIT compilation.

    from graphcast import xarray_jax
    from jax import tree_util

    exploded = xarray_jax.unwrap_vars(ds_in)
    spec_dict = tree_util.tree_map(sht_forward,exploded)
    rewrapped = xarray_jax.Dataset( {v : (ds_in[v].dims[:-2] + ('zonal_wavenumber','total_wavenumber'), spec_dict[v]) for v in spec_dict.keys()},
                                   coords=ds_in[set(ds_in.coords) - {'latitude','longitude','lat','lon'}].coords)
    rewrapped = rewrapped.assign_coords(total_wavenumber = ('total_wavenumber', np.arange(ds_in.lat.size)))
    return rewrapped

def power_spectra(spec):
    # Given an XArray dataset of spherical power spectra, compute the power spectrum
    # by total wavenumber
    from graphcast import xarray_jax
    from jax import tree_util
    from trainer.spectrum import power_spectral_density

    # The PSD computation function does not work with the fully wrapped dataset, so use unwrap_vars
    # to get a tree structure of (key, array) pairings
    exploded = xarray_jax.unwrap_vars(spec)

    # Compute the power spectrum of this dictionary by mapping over the 'pytree'
    power_spec_dict = tree_util.tree_map(power_spectral_density,exploded)

    # Rebuild the xarray dataset, copying coordinates from the supplied arrays
    power = xarray_jax.Dataset( {v : (spec[v].dims[:-2] + ('total_wavenumber',), power_spec_dict[v]) for v in power_spec_dict.keys()},
                               coords = spec.coords)
    return power

def cross_spectra(pspec, tsoec):
    # Given spherical spectra for a prediction and analysis, compute the cross spectrum 
    #  against the analysis, by total wavenumber
    # Suitable for compilation with jax
    from graphcast import xarray_jax
    from jax import tree_util
    from trainer.spectrum import cross_spectral_density

    # Get a tree representation of the prediction and analysis spectra, "exploding" the
    # XArray dataset
    pexplode = xarray_jax.unwrap_vars(pspec)
    aexplode = xarray_jax.unwrap_vars(tsoec)

    # Compute the power spectrum and cross spectrum
    cross_spec_dict = tree_util.tree_map(cross_spectral_density,pexplode,aexplode)

    # Rebuild the XArrays for return
    cross = xarray_jax.Dataset( {v : (pspec[v].dims[:-2] + ('total_wavenumber',), cross_spec_dict[v]) for v in cross_spec_dict.keys()},
                               coords = pspec.coords)
    return (cross)

def speccard(prediction, tspec, sht_func):
    # Given a prediction field, target spectrum, and tansformation function, compute the "spectral scorecard"
    # containing power sepctral density of the prediction and the cross spectral density between the prediction
    # and target (analysis), by variable and level

    pspec = sht_func(prediction)
    ppower = power_spectra(pspec)
    # tpower = power_spectra(tspec) # Not needed for cross spectral density
    cross = cross_spectra(pspec,tspec)
    return {'psd' : ppower, 'csd' : cross}


def scorecard(prediction,mask_hurr_reg, target,climato,climato_level_idx):
    # Compute the scorecard between predictions, analysis (target), and climatology, for all variables
    # and levels present in the climatology
    import graphcast.losses
    climato_level_idx = list(climato_level_idx)

    latitude_weight = graphcast.losses._weight_for_latitude_vector_with_poles(target.lat)
    longitude_weight = 1/target.lon.size

    # Apply masking to latitude/longitude weight, creating a number of 2D fields.  Note that the total sum
    # of the masked fields should be 1, so that mean and standard deviation values are comparable between
    # regions
    mask_unweighted = (latitude_weight*longitude_weight*mask_hurr_reg)
    mask_weighted = mask_unweighted / mask_unweighted.sum(dim=('lat','lon'))
    
    # prediction - analysis bias
    pa_bias = ((prediction - target)*mask_weighted).sum(dim=('lat','lon'))
    # prediction - analysis standard deviation (central)
    pa_std = (((prediction - target - pa_bias)**2 * mask_weighted).sum(dim=('lat','lon')))**0.5
    
    # Predictions vs climatology, see 
    # https://confluence.ecmwf.int/display/FUG/Section+12.A+Statistical+Concepts+-+Deterministic+Data#Section12.AStatisticalConceptsDeterministicData-TheDecompositionofMSE
    
    # Activity of prediction and analysis
    pc_act = ((prediction.isel(level=climato_level_idx) - climato)**2*mask_weighted).sum(dim=('lat','lon'))**0.5
    ac_act = ((target.isel(level=climato_level_idx) - climato)**2*mask_weighted).sum(dim=('lat','lon'))**0.5
    
    # Anomaly correlation coefficient
    acc = ((prediction.isel(level=climato_level_idx)-climato)*\
           (target.isel(level=climato_level_idx)-climato)*\
           mask_weighted).sum(dim=('lat','lon')) / (pc_act*ac_act)

    return({'bias' : pa_bias,
            'std' : pa_std,
            'p_act' : pc_act,
            'a_act' : ac_act,
            'acc' : acc})

short_names = {'10m_u_component_of_wind' : '10u',
               '10m_v_component_of_wind' : '10v',
               '2m_temperature' : '2t',
               'mean_sea_level_pressure' : 'msl',
               'total_precipitation_6hr' : 'tp',
               'temperature' : 't',
               'geopotential' : 'z',
               'specific_humidity' : 'q',
               'u_component_of_wind' : 'u',
               'v_component_of_wind' : 'v',
               'vertical_velocity' : 'w'}

def process_speccard(icard,idate,date_now):
    # Merge "spectral scorecards" from independent forecasts into a single aggregate.
    # Unlike process_scorecard, this function will use the valid date as its canonical
    # date variable.  From the power and cross spectral densities, amplitude ratio and
    # coherence are found by scaling against the analysis PSD, and that requires every
    # field be valid at the same time.
    import xarray as xr
    import numpy as np
    lead_time = np.timedelta64(date_now-idate,'ns')
    cards = [ icard[key].rename(short_names).assign_coords({'stat' : [key,],
                                                            'lead_time' : [lead_time,],
                                                            'valid_date' : [np.datetime64(date_now.replace(tzinfo=None),'ns'),],
                                                           }) for key in icard.keys()]
    return xr.concat(cards,dim='stat')                                                            

def process_scorecard(icard,idate,date_now):
    import xarray as xr
    import numpy as np
    lead_time = np.timedelta64(date_now-idate,'ns')
    cards = [ icard[key].rename(short_names).assign_coords({'stat' : [key,], 
                                                            'lead_time' : [lead_time,], 
                                                            'idate' : [np.datetime64(idate.replace(tzinfo=None),'ns'),] # Strip any timezone info
                                                            }) for key in icard.keys()]
    return xr.concat(cards,dim='stat')

def stamp(idate):
    # Helper function to return a YYYY-MM-DDTHH datetamp given
    # a datetime object
    return(idate.strftime('%Y-%m-%dT%H'))

if __name__ == '__main__':
    import datetime
    import dateparser
    import faulthandler
    import sys

    # Enable faulthandler to get tracebacks on signals like sigsegv
    faulthandler.enable()

    import argparse
    ## Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--apath',type=str,dest='apath',default='../gdata_025_wb',help='Location of analysis data')
    parser.add_argument('--cpath',type=str,dest='cpath',default='../era5_climatology.zarr',help='Location of climatology')
    parser.add_argument('--start-date',type=str,dest='start_date',default='1 Jan 2023 00:00',help='Starting date/time')
    parser.add_argument('--end-date',type=str,dest='end_date',default='31 Dec 2023 18:00',help='Ending date/time (inclusive)')
    parser.add_argument('--forecast-length',type=int,dest='forecast_length',default=6,help='Maximum forecast length (hours)')
    parser.add_argument('--to-path',type=str,dest='outpath',help='Output file for scores')
    parser.add_argument('--model-checkpoint',type=str,dest='model_checkpoint',help='Model checkpoint to load')
    parser.add_argument('--init-interval',type=int,dest='init_interval',default=6,help='How often to initialize a new forecast (h)')
    parser.add_argument('--norm-factors',type=str,dest='norm_path',default=None,
                        help='Path to the directory containing Graphcast normalization factors')
    parser.add_argument('--spectrum-leads',type=int,nargs='*',dest='spectrum_leads',help='Lead times in hours for spectrum computation (leave blank for no spectrum)')
    parser.add_argument('--spectrum-output',type=str,dest='specpath',help='Output file for spectrum',default=None)

    ## Runtime parameters, to be set from the command line
    args = parser.parse_args()

    import jax
    import xarray as xr
    import numpy as np
    import forecast.generate_model
    import trainer.dataloader
    import forecast.encabulator
    import dask
    import dask.distributed
    
    # Set Dask configuration to suppress warnings about memory use
    dask.config.set({'distributed.scheduled.active-memory-manager.measure' : 'managed',
                    'distributed.worker.memory.rebalance.measure' : 'managed',
                    'distributed.worker.memory.spill' : False,
                    'distributed.worker.memory.pause' : False,
                    'distributed.worker.memory.terminate' : False})
    
    # Get the available number of CPUs and GPUs
    import os
    num_procs = len(os.sched_getaffinity(0))
    try:
        num_gpus = len(jax.devices(backend='gpu'))
    except RuntimeError: # no GPU
        num_gpus = 0


    forecast_length = args.forecast_length # 40 # Forecast length for the scorecard, 40=10d
    init_interval = args.init_interval
    analysis_path = args.apath # '../gdata_025_wb' # Path of verifying analysis and initial conditions
    climato_path = args.cpath # '../era5_climatology.zarr' # Path to climatology
    # Path to graphcast parameters
    params_path = args.model_checkpoint # 'params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz'
    outpath = args.outpath
    specpath = args.specpath
    spectrum_leads = [float(f) for f in args.spectrum_leads]
    if len(spectrum_leads) > 0:
        compute_spectrum=True
        assert(specpath is not None)
    else:
        compute_spectrum=False


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



    cpu_device = jax.devices(backend='cpu')[0]

    # Open databases
    dbase,_ = trainer.dataloader.open_databases(analysis_path,None) # Note no need for a separate verification dbase
    climato_dbase = xr.open_zarr(climato_path)

    # Load model
    (model_config, task_config, params) = forecast.generate_model.load_model(params_path)
    levels = np.array(sorted(list(task_config['pressure_levels'])),dtype=np.int64)

    # Generate latitude and longitude for the model, based on its resolution
    global model_latitude, model_longitude
    model_latitude = xr.DataArray(np.linspace(-90,90,int(1+180/model_config['resolution']),dtype=np.float32),dims='latitude')
    model_latitude = model_latitude.assign_coords({'latitude' : model_latitude})
    model_longitude = xr.DataArray(np.linspace(0,360-model_config['resolution'],int(360/model_config['resolution']),dtype=np.float32),
                                dims='longitude')
    model_longitude = model_longitude.assign_coords({'longitude' : model_longitude})

    input_variables = list(task_config['input_variables'])
    target_variables = list(task_config['target_variables'])
    forcing_variables = list(task_config['forcing_variables'])

    import graphcast.graphcast
    input_only_vars = [v for v in input_variables if v not in target_variables]
    global input_from_target
    global input_from_forcing
    input_from_target = [v for v in input_variables if v in target_variables]
    input_from_forcing = [v for v in input_only_vars if v in forcing_variables]
    input_static_vars = list(graphcast.graphcast.STATIC_VARS)

    # Get static variables that do not change in time
    static_vars = dbase[input_static_vars]
    static_vars = static_vars.sel(latitude=model_latitude,longitude=model_longitude)
    static_vars = static_vars.rename(latitude='lat',longitude='lon')
    for var in static_vars.data_vars:
        # If the database does have a time dimension for any of these variables, drop it.
        if 'time' in static_vars[var].dims:
            static_vars[var] = static_vars[var].isel(time=0,drop=True)
    if 'time' in static_vars.coords:
        del static_vars.coords['time']
    static_vars = static_vars.compute()

    # static_vars = dbase[input_static_vars].isel(time=0,drop=True).rename(latitude='lat',longitude='lon').compute()

    # Get the level indices of the output prediction that correspond to the levels available in the climatology
    # database, for scorecarding
    climato_level_idx = np.searchsorted(levels,climato_dbase.level.data)

    norm_path = args.norm_path

    if (norm_path is not None):
        # Load provided normalization factors
        diffs_stddev_by_level = xr.load_dataset(f"{norm_path}/diffs_stddev_by_level.nc").compute()
        mean_by_level = xr.load_dataset(f"{norm_path}/mean_by_level.nc").compute()
        stddev_by_level = xr.load_dataset(f"{norm_path}/stddev_by_level.nc").compute()
    else:
        # Otherwise do not load normalization factors, and default to the loading inside the predictor-generator
        diffs_stddev_by_level = None
        mean_by_level = None
        stddev_by_level = None

    # Build the predictor function; note full float32 predictions
    predictor = forecast.generate_model.build_predictor_params(model_config,task_config,use_float16=True,
                                                               diffs_stddev_by_level = diffs_stddev_by_level, 
                                                               mean_by_level = mean_by_level,
                                                               stddev_by_level = stddev_by_level)
    # generate hurricane mask
    from hurr_scorecard_mask.mask_hurr_generate import mask_hurr_gen
    mask_hurr_reg = mask_hurr_gen()

    # JIT compile the scorecard calculator
    scorecard_jit = jax.jit(scorecard,static_argnums=(4,))   
    stack_inputs = jax.jit(stack_inputs) 

    import forecast.forecast_variables

    gpu_skeleton = [{'device' : dev, 'params' : jax.device_put(params,dev),
                'inputs' : None, 'targets' : None, 'tspec' : None, 'forcings' : None, 'climato' : None} for dev in jax.devices('gpu')[:num_gpus]]
    
    forecast_set = []
    output_futures = []

    dt_6h = datetime.timedelta(hours=6)
    dt_1h = datetime.timedelta(hours=1)
    
    total_forecasts = 1 + int((end_date - start_date)/(dt_1h*init_interval))
    forecast_length = forecast_length*dt_1h

    import concurrent

    print(f'Computing {total_forecasts} scorecard sets from {stamp(start_date)} to {stamp(end_date)}')
    print(f'based on forecasts initialized every {init_interval}h')
    print(f'initializing with and verifying against {analysis_path}')
    print(f'using {num_gpus} GPUs and {num_procs} CPUs')
    print(f'with model checkpoint {params_path}')
    print(f'and output written to {outpath}')
    if (compute_spectrum):
        print(f'Also computing spectrum at {", ".join(f"{d}h" for d in spectrum_leads)}')
        print(f'   and writing output to {specpath}')
    sys.stdout.flush()

    # If computing spectrum, get assoicated weights
    if (compute_spectrum):
        import trainer.spectrum
        leg_coefs = trainer.spectrum.generate_spectral_coefs(model_latitude.data)

        do_sht_array = lambda f: trainer.spectrum.sht_eval(leg_coefs,f)
        do_sht_ds = lambda f: sht_ds(f,do_sht_array)
        do_speccard = lambda pred, tspec: speccard(pred,tspec,do_sht_ds)

        sht_ds_jit = jax.jit(do_sht_ds)
        speccard_jit = jax.jit(do_speccard)
    else:
        speccard_jit = None
        sht_ds_jit = None

    # Set a 'watchdog timer' to dump a traceback if execution hangs
    # Set a long timeout at first, to account for jax compilation
    faulthandler.dump_traceback_later(600,exit=True)
    gtic = datetime.datetime.now()
    import queue
    gpu_queue = queue.Queue()

    with (concurrent.futures.ThreadPoolExecutor(max_workers=max(1,num_gpus+1)) as gpu_executor,
          dask.distributed.Client(processes=False,threads_per_worker=len(os.sched_getaffinity(0))) as Client):
        # List of current forecasts, format (idate, inputs)
        forecast_ics = []
        # Output of per-forecast scorecards, for later merging
        out_scorecards = []
        out_speccards = []

        ## Get first set of iniitial conditions

        now_date = start_date # Initialization time of current forecast
        # Get forcings and targets valid at +6h
        (ignore,forcings_p6h, targets_p6h) = trainer.dataloader.build_forecast(now_date,1,task_config,
                                                            model_latitude,model_longitude,input_variables,target_variables,
                                                            dbase,dbase)
        # Get climatology valid at +6h
        climato_p6h = load_climato(targets_p6h,climato_dbase,target_variables)
        targets_p6h = targets_p6h.drop_vars('datetime')
        forcings_p6h = forcings_p6h.drop_vars('datetime')

        # We also want to keep forcings and targets valid at 0h and -6h to create new ICs.
        (ignore,forcings_0h, targets_0h) = trainer.dataloader.build_forecast(now_date-dt_6h,1,task_config,
                                                            model_latitude,model_longitude,input_variables,target_variables,
                                                            dbase,dbase)
        targets_0h = targets_0h.drop_vars('datetime')
        forcings_0h = forcings_0h.drop_vars('datetime')
        (ignore,forcings_m6h, targets_m6h) = trainer.dataloader.build_forecast(now_date-2*dt_6h,1,task_config,
                                                            model_latitude,model_longitude,input_variables,target_variables,
                                                            dbase,dbase)
        targets_m6h = targets_m6h.drop_vars('datetime')
        forcings_m6h = forcings_m6h.drop_vars('datetime')

        # Load/compute all data
        (targets_m6h,targets_0h,targets_p6h,
        forcings_m6h,forcings_0h,forcings_p6h,
        climato_p6h, mask_hurr_reg) = Client.compute((targets_m6h,targets_0h,targets_p6h,forcings_m6h,forcings_0h,forcings_p6h,climato_p6h,mask_hurr_reg),sync=True)
        del ignore

        while (now_date < end_date):
            # Loop through all current forecasts, which have ICs valid at now_date
            # Propagate to now_date+6h, grabbing scorecards and next ICs
            forecast_futures = [] # No forecasts are currently enqueued
            
            tic = datetime.datetime.now()
            count = 0
            
            # Populate GPU queue
            for skel in gpu_skeleton:
                gpu_queue.put(skel)
                
            next_date = now_date + dt_6h
            # print(f'Processing {stamp(now_date)}')
            
            if (next_date < end_date):
                # Queue loading of data for the next date, valid at now+12h
                (ignore, forcings_next, targets_next) = trainer.dataloader.build_forecast(next_date,1,task_config,
                                                                    model_latitude,model_longitude,input_variables,target_variables,
                                                                    dbase,dbase)
                climato_next = load_climato(targets_next,climato_dbase,target_variables)
                targets_next = targets_next.drop_vars('datetime')
                forcings_next = forcings_next.drop_vars('datetime')
        
                # Load the data in the background with dask, getting Futures
                (targets_next_f,forcings_next_f,climato_next_f) = Client.compute((targets_next,forcings_next,climato_next),sync=False)
                del climato_next, forcings_next, targets_next, ignore

            if (compute_spectrum):
                # If computing spectrum, take the transform of the target field including derived variables
                import trainer.loss_utils
                expanded_analysis = trainer.loss_utils.derived_variables(targets_p6h,compute_wind_speed=True).isel(time=0,batch=0,drop=True)
                tspec = unwrap_ds(sht_ds_jit(expanded_analysis))

                # Build "speccard" for the analysis itself, 0h lead time
                analysis_speccard = speccard_jit(expanded_analysis,tspec)
                # The values are now on GPU, so unwrap them
                analysis_speccard = {key: unwrap_ds(analysis_speccard[key]) for key in analysis_speccard.keys()}
                # print(analysis_speccard)
                out_speccards.append(process_speccard(analysis_speccard,next_date,next_date))
            else:
                tspec = None

            #for (idate, inputs) in forecast_ics:
            while len(forecast_ics) > 0:
                (idate,inputs) = forecast_ics.pop()
                # print(f'  Queueing existing forecast from {stamp(idate)}')
                forecast_futures.append(gpu_executor.submit(advance_forecast,idate,gpu_queue,inputs,
                                                        targets_p6h,tspec,
                                                        forcings_p6h,climato_p6h,predictor,climato_level_idx,mask_hurr_reg))
                del inputs # Remove memory reference, allowing garbage collection

            # Check to see if we should initialize a new forecast
            if ( (now_date - start_date)/dt_1h % init_interval == 0):
                # print(f'  Initializing new input for {stamp(now_date)}')
                inputs_new = new_inputs(targets_m6h,targets_0h,forcings_m6h,forcings_0h,static_vars)
                forecast_futures.append(gpu_executor.submit(advance_forecast,now_date,gpu_queue,inputs_new,
                                                        targets_p6h,tspec,
                                                        forcings_p6h,climato_p6h,predictor,climato_level_idx,mask_hurr_reg))
                del inputs_new # Remove memory reference


            for completed_future in concurrent.futures.as_completed(forecast_futures):
                (idate,inputs_next,scorecard,this_speccard) = completed_future.result()
                count += 1
                if (compute_spectrum and (next_date - idate)/datetime.timedelta(hours=1) in spectrum_leads):
                    out_speccards.append(process_speccard(this_speccard,idate,next_date))
                out_scorecards.append(process_scorecard(scorecard,idate,next_date))
                # print(f'  Received forecast for {stamp(idate)} init, Z500: {float(scorecard["std"].geopotential.sel(level=500).data):.2f}')
                if ( (next_date - idate) < forecast_length and next_date <= end_date ):
                    # print(f'  Re-enqueueing forecast')
                    forecast_ics.append((idate,inputs_next))
                else:
                    # print('    Forecast at max length, discarding')
                    pass
                del inputs_next
                # Feed the watchdog
                faulthandler.dump_traceback_later(120,exit=True)
            # break
        
            if (next_date <= end_date):
                # Keep forcings and targets for the past two steps, to use in creating ICs
                forcings_m6h = forcings_0h
                targets_m6h = targets_0h
                forcings_0h = forcings_p6h
                targets_0h = targets_p6h
                
                # Overwrite _p6h variables with _next, waiting on the Futures
                forcings_p6h = forcings_next_f.result()
                targets_p6h = targets_next_f.result()
                climato_p6h = climato_next_f.result()
        
            # Clear GPU queue
            for idx in range(num_gpus):
                gpu_queue.get(timeout=1)
            assert(gpu_queue.empty())

            toc = datetime.datetime.now()
            print(f'Processed {count} forecasts for {stamp(now_date)} in {(toc-tic).total_seconds():.2f} seconds')
                
            now_date = next_date
            
    gtoc = datetime.datetime.now()
    gtime = (gtoc-gtic).total_seconds()
    print(f'{gpu_time:.2f}s GPU time of {gtime:.2f}s, ratio {100*gpu_time/gtime/num_gpus:.2f}%')

    # Cancel watchdog
    faulthandler.cancel_dump_traceback_later()

    out_dataset = xr.combine_by_coords(out_scorecards)
    out_dataset.coords['valid_date'] = out_dataset['idate'] + out_dataset['lead_time']

    if (compute_spectrum):
        out_spec = xr.combine_by_coords(out_speccards)
        out_spec.coords['idate'] = out_spec['valid_date'] - out_spec['lead_time']

    if (os.path.exists(outpath)):
        print(f'{outpath} exists already, removing')
        import shutil
        shutil.rmtree(outpath)


    print(f'Writing to {outpath}')
    out_dataset.to_zarr(outpath,compute=True)
    
    if (compute_spectrum):
        if(os.path.exists(specpath)):
            print(f'{specpath} exists already, removing')
            import shutil
            shutil.rmtree(specpath)
        print(f'Writing to {specpath}')
        out_spec.to_zarr(specpath,compute=True)
        

    print(f'… complete')
