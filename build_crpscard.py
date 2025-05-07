import xarray as xr
import numpy as np
import numcodecs
import trainer.dataloader
import forecast.encabulator
import jax
import jax.numpy as jnp
import datetime
import dask
import forecast.generate_model
# from forecast import generate_model
import matplotlib.pyplot as plt

# Utility functions for dataset manipulation
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

# def new_inputs(targets_last,targets_now,forcings_last,forcings_now,static_vars):
#     import xarray as xr
#     import numpy as np
#     global input_from_target
#     global input_from_forcing

#     foo_last = xr.merge((targets_last[input_from_target],forcings_last[input_from_forcing],static_vars))
#     foo_last = foo_last.assign_coords(time=[np.timedelta64(-6,'h').astype('timedelta64[ns]')])
#     foo_now =  xr.merge((targets_now[input_from_target],forcings_now[input_from_forcing]))
#     foo_now = foo_now.assign_coords(time=[np.timedelta64(0,'h').astype('timedelta64[ns]')])
#     foo = xr.concat((foo_last,foo_now),dim='time',coords='minimal',data_vars='minimal')

#     return foo

def make_target_template(target_vars,length,latitude,longitude,levels,on_device = None):
    '''Given a forcing dataset, compute a target dataset consisting of
    uninitialized (zero) Jax arrays of the correct size.'''
    import xarray as xr
    import graphcast.graphcast
    # import dask
    import numpy as np
    import graphcast.xarray_jax
    import jax.numpy as jnp

    delta_6h = np.timedelta64(6*3600*1_000_000_000,'ns')

    if (on_device is None):
        on_device = jax.devices('cpu')[0]
    
    with jax.default_device(on_device):
        # Create blank arrays, for with-levels and without-levels variables.  The
        # target dataset is read-only, so there's no need to have separate arrays
        # per variable.
        array_3d = jnp.zeros((1, length, levels.size, latitude.size, longitude.size),
                             dtype=np.float32)
        array_2d = jnp.zeros((1, length, latitude.size, longitude.size),
                             dtype=np.float32)

    data_vars = {}

    for var in target_vars:
        if var in graphcast.graphcast.TARGET_SURFACE_VARS:
            # 2D variable
            data_vars[var] = (('batch','time','lat','lon'),array_2d,)
                                          
        else:
            # 3D variable
            data_vars[var] = (('batch','time','level','lat','lon'),array_3d)

    
    target = graphcast.xarray_jax.Dataset(coords={'time' : delta_6h*(1+np.arange(length)),
                                  'level' : np.array(levels),
                                  'lat' : np.array(latitude),
                                  'lon' : np.array(longitude)},
                                  data_vars=data_vars)

    return target

@jax.jit
def stack_inputs(old_input,pred,forcings):
    import xarray as xr
    if (pred.time.size == 1):
        inputs_next = pred[input_from_target]
        inputs_next[input_from_forcing] = forcings[input_from_forcing]
        for v in inputs_next.data_vars:
            inputs_next[v] = inputs_next[v].transpose(*old_input[v].dims)
        outputs = xr.concat((old_input.isel(time=[1,]),inputs_next),dim='time',coords='minimal',data_vars='minimal')
        outputs['time'] = old_input['time']
        return outputs
    else:
        inputs_next = pred[input_from_target].isel(time=[-2,-1]).copy(deep=True)
        inputs_next[input_from_forcing] = forcings[input_from_forcing].isel(time=[-2,-1])
        for v in inputs_next.data_vars:
            inputs_next[v] = inputs_next[v].transpose(*old_input[v].dims)
        for v in old_input.data_vars:
            if 'time' not in old_input[v].dims:
                inputs_next[v] = old_input[v]
        inputs_next['time'] = old_input['time']
        return inputs_next
    
def stamp(idate):
    # Helper function to return a YYYY-MM-DDTHH datetamp given
    # a datetime object
    return(idate.strftime('%Y-%m-%dT%H'))

def date_range(start,end,interval):
    out = []
    now = start
    while now < end:
        out.append(now)
        now += interval
    return out

def varstats(target_var,ensemble_vars):
    N = len(ensemble_vars)
    middle_e = len(ensemble_vars)//2 
    det_mse = (((ensemble_vars[middle_e] - target_var)**2).mean(dim='lon')*lat_weights_da).sum(dim='lat')
    ens_mean = sum(e for e in ensemble_vars)/N
    ens_mse = (((ens_mean - target_var)**2).mean(dim='lon')*lat_weights_da).sum(dim='lat')
    spread_sq = (sum( (e - ens_mean)**2 for e in ensemble_vars).mean(dim='lon')*lat_weights_da).sum(dim='lat')/(N-1)
    local_mae = sum( np.abs(e - target_var) for e in ensemble_vars)
    outer_spread = sum(np.abs(ensemble_vars[j] - ensemble_vars[k]) for j in range(len(ensemble_vars)) for k in range(j+1,len(ensemble_vars)))

    # Unbiased eRMSE from Leutbecher 2007: https://doi.org/10.1016/j.jcp.2007.02.014 
    # or eq5 of https://journals.ametsoc.org/view/journals/mwre/150/11/MWR-D-21-0315.1.xml?tab_body=fulltext-display

    ub_emse = (((ens_mean - target_var)**2 - \
                1/(N*(N-1)) * sum((e - ens_mean)**2 for e in ensemble_vars)).mean(dim='lon')*lat_weights_da).sum(dim='lat')
    


    # No factor of 2 in fair CRPS calculation because the sum is over i, j>i
    local_fair_crps = local_mae/N - 1/(N*(N-1)) * outer_spread
    fair_crps = (local_fair_crps.mean(dim='lon')*lat_weights_da).sum(dim='lat')
    return det_mse, ens_mse, spread_sq, fair_crps, ub_emse

@jax.jit
def ensstats(targets,ensemble):
    vstat = {}
    for var in targets.data_vars:
        stat = varstats(targets[var],[e[var] for e in ensemble])
        vstat[var]=xr.concat(stat,dim='stat').assign_coords(stat=('stat',['detmse','ensmse','sqspread','crps','ub_ensmse']))
    return vstat

def allstats(targets,ensemble,vdate):
    outs = []
    for start in range(0,len(ensemble) - num_ensemble_members+1):
        # print(start)
        ens = ensemble[start:start+num_ensemble_members]
        lead = (vdate - np.median([t.astype(int) for (t,e) in ens]).astype(ens[0][0].dtype))
        lead = lead.astype('timedelta64[ns]')
        vstat = {}
        vstat = ensstats(targets,[e[1] for e in ens])
        outs.append(xr.Dataset(vstat).expand_dims(vdate=[vdate.astype('datetime64[ns]')],lead=[lead]))
    return xr.merge(outs)

if __name__ == '__main__':
    import dateparser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--params-path',type=str,dest='params_path')
    parser.add_argument('--outpath',type=str,dest='outpath')
    parser.add_argument('--start-vdate',type=str,dest='start_vdate',default='1 Jan 2023 00:00',help='Starting date/time')
    parser.add_argument('--end-vdate',type=str,dest='end_vdate',default='31 Dec 2023 18:00',help='Ending date/time (inclusive)')

    ## Runtime parameters, to be set from the command line
    args = parser.parse_args()


    # Graphcast checkpoint to be used
    # params_path = 'params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz'
    # params_path = 'params/GraphCast_operational - ERA5-HRES 1979-2021 - resolution 0.25 - pressure levels 13 - mesh 2to6 - precipitation output only.npz'
    params_path = args.params_path
    print(f'Loading paramters from {params_path}')
    # params_path = 'train_checkpoints/amse_finetune_025deg/ar12/hres.001250.ckpt'

    # Database for ICs/evaluations
    dbase_path = '/fs/site6/eccc/mrd/rpnatm/csu001/ppp6/hres_precip'
    dbase,_ = trainer.dataloader.open_databases(dbase_path,None)

    # Parameters of lagged ensemble

    start_vdate = dateparser.parse(args.start_vdate,
                                  ['%Y%m%d%H',  # Also parse YYYYMMDDHH (ISO 8601-2004)
                                   '%Y%m%d%HZ', # ... with UTC marker
                                   '%Y%m%dT%H', # and YYYYMMDDTHH (ISO 8601-2019)
                                   '%Y%m%dT%HZ',# ... with UTC marker
                                  ])
    assert(start_vdate is not None)
    end_vdate = dateparser.parse(args.end_vdate,
                                ['%Y%m%d%H',  # Also parse YYYYMMDDHH (ISO 8601-2004)
                                 '%Y%m%d%HZ', # ... with UTC marker
                                 '%Y%m%dT%H', # and YYYYMMDDTHH (ISO 8601-2019)
                                 '%Y%m%dT%HZ',# ... with UTC marker
                                ])
    assert(end_vdate is not None)
    # start_vdate = datetime.datetime(2022,1,11,0) # First date for ensemble computation
    # end_vdate = datetime.datetime(2022,1,31,12) # Last date of ensemble computation
    vdate_interval = datetime.timedelta(hours=12) # Increments of evaluation date
    eval_vdates = date_range(start_vdate,end_vdate+vdate_interval,vdate_interval)

    print(f'Evaluating ensembles from {stamp(start_vdate)} to {stamp(end_vdate)}, every {vdate_interval.total_seconds()/3600}h')
    print(f'Writing output to {args.outpath}')

    step_length = datetime.timedelta(hours=6) # Step size of graphcast
    max_forecast_length = 40*step_length # Maximum length of the forecast
    init_interval = 2*step_length # Time to wait between new initializations
    ensemble_halfspan = datetime.timedelta(hours=48) # Span around central idate to include in lagged ensemble
    num_ensemble_members = 1 + 2 * ensemble_halfspan // init_interval

    start_idate = start_vdate - max_forecast_length # First date 
    eval_idates = date_range(start_idate,end_vdate,init_interval)

    # GPU device reference
    gpu_device = jax.devices('gpu')[0]

    # Load the model
    (model_config, task_config, params) = forecast.generate_model.load_model(params_path)
    model_latitude = xr.DataArray(np.linspace(-90,90,int(1+180/model_config['resolution']),dtype=np.float32),dims='latitude')
    model_latitude = model_latitude.assign_coords({'latitude' : model_latitude})
    model_longitude = xr.DataArray(np.linspace(0,360-model_config['resolution'],int(360/model_config['resolution']),dtype=np.float32),
                                dims='longitude')
    model_longitude = model_longitude.assign_coords({'longitude' : model_longitude})
    levels = list(task_config['pressure_levels'])

    # Variables used/predicted by this checkpoint
    input_variables = list(task_config['input_variables'])
    target_variables = list(task_config['target_variables'])
    forcing_variables = list(task_config['forcing_variables'])

    # Helper arrays for data shuffling between prediction, forcing, and input
    input_from_target = [i for i in input_variables if i in target_variables]
    input_from_forcing = [i for i in input_variables if i in forcing_variables]
    static_vars = [i for i in input_variables if i not in target_variables and i not in forcing_variables]

    # Load normalizing statistics
    norm_path='stats/era5'
    diffs_stddev_by_level = xr.load_dataset(f"{norm_path}/diffs_stddev_by_level.nc").compute()
    mean_by_level = xr.load_dataset(f"{norm_path}/mean_by_level.nc").compute()
    stddev_by_level = xr.load_dataset(f"{norm_path}/stddev_by_level.nc").compute()

    # Get a predictor function
    predictor = forecast.generate_model.build_predictor_params(model_config,task_config,use_float16=False,
                                                    diffs_stddev_by_level = diffs_stddev_by_level, 
                                                    mean_by_level = mean_by_level,
                                                    stddev_by_level = stddev_by_level)
    
    # Compute quadrature weights in latitude
    lat_rad = (jnp.array(model_latitude) * jnp.pi / 180)#.astype(jnp.float32)
    # Midpoint latitudes, re-adding the poles
    lat_mid = jnp.concatenate((jnp.array([-jnp.pi/2]), 
                            0.5*(lat_rad[1:] + lat_rad[:-1]), 
                            jnp.array([jnp.pi/2])))
    # The integrated weight is a finite difference on sin(lat).  Near the equator this is essentially
    # cosine-weighting for latitude, but it also behaves nicely near the poles
    lat_weights = jnp.sin(lat_mid[1:]) - jnp.sin(lat_mid[:-1])
    # Normalize to have sum 1
    lat_weights = (lat_weights / jnp.sum(lat_weights))
    lat_weights_da = xr.DataArray(lat_weights,coords={'lat':model_latitude.rename(latitude='lat')})

    current_inits = []
    now_vdate = start_idate + step_length # the valid date corresponding to the first idate
    targets_template = make_target_template(target_variables,1,model_latitude,model_longitude,np.array(levels),gpu_device)
    stats = []
    next_vdate = eval_vdates[0]
    
    while now_vdate <= end_vdate: # Loop until we've reached the last valid date
        # print(f'Processing vdate {stamp(now_vdate)}')
        current_predictions = []
        next_inits = []
        now_idate = now_vdate - step_length
        # Get the fields required to compute a forecast valid now
        (inputs,forcings,targets) = trainer.dataloader.build_forecast(now_idate, 1, task_config,
                                                                    model_latitude, model_longitude, input_variables, target_variables,
                                                                    dbase, dbase)
    
        # Drop 'datetime' from all fields, we don't need it
        inputs = inputs.drop_vars('datetime')
        forcings = forcings.drop_vars('datetime')
        targets = targets.drop_vars('datetime')
        forcings = wrap_dataset(forcings.compute(),gpu_device)
    
        tic = datetime.datetime.now()
        count = 0
    
        if now_vdate in eval_vdates:
            next_vdate = now_vdate + vdate_interval
    
        if now_idate in eval_idates: # We need to initialize a new forecast
            # print(f'Starting new forecast for {stamp(now_idate)}')
            current_inits.append((now_idate,inputs.compute()))
    
        # Update all current forecasts to the new vdate
        for (idate,input) in current_inits:
            count += 1
            # print(f'Making prediction initialized {stamp(idate)}')
            pred = predictor(inputs=input,targets=targets_template,forcings=forcings,params=params)
    
            # If this forecast will be admissible at the next vdate, make new inputs to continue
            # the forecast
            if (next_vdate - idate <= max_forecast_length):
                next_inits.append((idate, unwrap_ds(stack_inputs(input,pred,forcings))))
    
            # Save prediction for statistics
            current_predictions.append((np.datetime64(idate),
                                        unwrap_ds(pred.isel(batch=0,time=0,drop=True))))
            del pred
    
        # Do statistics
        if now_vdate in eval_vdates:
            # Do statistics here
            targets = targets.compute()
            stats.append(unwrap_ds(allstats(targets.isel(time=0,drop=True),current_predictions,np.datetime64(now_vdate))))
            
        toc = datetime.datetime.now()
        print(f'Produced {count} forecasts for {stamp(now_vdate)} in {(toc-tic).total_seconds():.2f}s')
    
        # Reassign init array and continue
        current_inits = next_inits
        now_vdate += step_length
    
    stats = xr.merge(stats)
    # stats_out[label] = stats
    print(f'Saving to {args.outpath}')
    stats.to_zarr(args.outpath,mode='w')