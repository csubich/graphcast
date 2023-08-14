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


## variable_err_weights: compute the sensitivity of "proper" error functions at long lead times
# to small perturbations in each forecast variable, in order to determine better error weights

import os
# Ideally, these environment variables should be set at the command-line, before
# even launching the script.  Results seem inconsistent when the environment variables
# are set via os.
if ('TF_FORCE_UNIFIED_MEMORY' not in os.environ.keys()):
    os.environ['TF_FORCE_UNIFIED_MEMORY']='1'
if ('XLA_CLIENT_MEM_FRACTION' not in os.environ.keys()):
    os.environ['XLA_CLIENT_MEM_FRACTION'] = '5.0'
import forecast.encabulator # Data compressor, for cache
import argparse
import trainer.dataloader as dataloader
debug_prints = False

def make_target_template(forcing,target_vars):
    '''Given a forcing dataset, compute a target dataset consisting of
    uninitialized (zero) Jax arrays of the correct size.'''
    import xarray as xr
    import graphcast.graphcast
    # import dask
    import numpy as np
    import graphcast.xarray_jax
    import jax.numpy as jnp


    array_3d = jnp.zeros((forcing.batch.size, forcing.time.size, forcing.level.size, forcing.lat.size, forcing.lon.size),
                         dtype=np.float32)
    array_2d = jnp.zeros((forcing.batch.size, forcing.time.size, forcing.lat.size, forcing.lon.size),
                         dtype=np.float32)

    data_vars = {}

    for var in target_vars:
        if var in graphcast.graphcast.TARGET_SURFACE_VARS:
            # 2D variable
            data_vars[var] = graphcast.xarray_jax.DataArray(array_2d,
                                          dims=('batch','time','lat','lon'))
        else:
            # 3D variable
            data_vars[var] = graphcast.xarray_jax.DataArray(array_3d,
                                          dims=('batch','time','level','lat','lon'))

    
    target = graphcast.xarray_jax.Dataset(coords=forcing.coords,data_vars=data_vars)

    return target


def unwrap_ds(in_ds):
    import xarray as xr
    from graphcast import xarray_jax

    return xr.Dataset( {var : (in_ds[var].dims, xarray_jax.unwrap_data(in_ds[var])) for var in in_ds}, coords=in_ds.coords)

def forecast_by_step(predictor,task_config,params,inputs,forcings,target_lead_times):
    # Build a long forecast, one 6h forecast at a time, but only return
    # the prediction at selected lead times
    import xarray as xr

    target_time_list = list(target_lead_times)

    targets = make_target_template(forcings.isel(time=[0,]),task_config['target_variables'])

    # Starting input field is the one provided
    ids = inputs

    out_preds = []
    for dt in range(forcings.time.size):
        # Build the proper forcing field:
        # print(dt)
        fds = forcings.isel(time=[dt,]).compute()
        # Rewrite the 'time' field of forcing to contain just +6h
        fds['time'] = targets.time
        # Make predictions
        preds = predictor(inputs=ids,targets=targets,forcings=fds,params=params)

        if (dt in target_time_list):
            out_preds.append(unwrap_ds(preds))

        # Build input field from predictions: copy over all predictions that are also inputs
        pred_to_input = preds[[v for v in preds.data_vars if v in ids.data_vars]]
        # And copy any forcings that become inputs
        for v in [v for v in fds.data_vars if v in ids.data_vars]:
            pred_to_input[v] = fds[v]

        # Build new input field
        ids_new = xr.concat([ids.isel(time=[-1,]), pred_to_input],dim='time',coords='minimal',data_vars='minimal').compute()
        # Rewrite time array
        ids_new['time'] = ids.time

        del ids, preds, pred_to_input
        
        ids = ids_new.compute()

    preds = xr.concat(out_preds,dim='time')
    preds['time'] = forcings.time.isel(time=target_time_list)
    return(preds)

def graphcast_loss_by_level(pred,target,diffs_stddev_by_level,weighted=True):
    import graphcast.losses
    import xarray as xr

    if ('level' in pred.coords and weighted):
        level_weights = pred['level'] / pred['level'].sum()
    else:
        level_weights = 1
    latitude_weights = graphcast.losses._weight_for_latitude_vector_with_poles(pred.lat)
    latitude_weights = latitude_weights / latitude_weights.mean()

    # npred = trainer.loss_utils.normalize(pred,inputs.isel(time=-1),mean_by_level,stddev_by_level,diffs_stddev_by_level)
    # ntarget = trainer.loss_utils.normalize(target,inputs.isel(time=-1),mean_by_level,stddev_by_level,diffs_stddev_by_level)
    # npred = pred/diffs_stddev_by_level
    # ntarget = target/diffs_stddev_by_level

    mse = xr.Dataset()
    # print('loss_by_level',diffs_stddev_by_level)
    for v in pred.data_vars:
        if 'level' in pred[v].dims:
            mse[v] = ((target[v] - pred[v])**2 * level_weights * latitude_weights / diffs_stddev_by_level[v]**2).mean(dim=('lat','lon'))
        else:
            mse[v] = ((target[v] - pred[v])**2 * latitude_weights / diffs_stddev_by_level[v]**2).mean(dim=('lat','lon'))

    # for v in mse.data_vars:
    #     if (v in per_variable_weights):
    #         mse[v] = mse[v] * per_variable_weights[v]

    return(mse)

def graphcast_loss(pred,target,diffs_stddev_by_level,per_variable_weights,level_weighted=True):
    # print('graphcast_loss',diffs_stddev_by_level)
    mse = graphcast_loss_by_level(pred,target,diffs_stddev_by_level,level_weighted)
    
    mse = mse.sum(dim='level')
    loss = sum(mse[v] * (per_variable_weights[v] if v in per_variable_weights else 1) for v in mse.data_vars)
    return(loss,mse)

def graphcast_loss_time_avg(loss,mse):
    return (loss.mean(dim='time'),
            mse.mean(dim='time'))

def physical_mse(ds1,ds2,vars,levels):
    import graphcast.losses
    if ('level' in ds1[vars].dims):
        ds1 = ds1[vars].sel(level=levels)
        ds2 = ds2[vars].sel(level=levels)
    latitude_weights = graphcast.losses._weight_for_latitude_vector_with_poles(ds1.lat)
    latitude_weights = latitude_weights / latitude_weights.mean()

    mse = ((ds1 - ds2)**2*latitude_weights).mean(dim=('lat','lon'))
    return unwrap_ds(mse)

from dataclasses import dataclass
from typing import Tuple
@dataclass(frozen=True)
class TestGroup:
    vars: Tuple[str]
    name: str

test_groups = {
    TestGroup(vars=('10m_u_component_of_wind','10m_v_component_of_wind'),name='10u+v'),
    TestGroup(vars=('2m_temperature',),name='2t'),
    # TestGroup(vars=('temperature','geopotential'),name='z+t'),
    TestGroup(vars=('temperature',),name='t'),
    TestGroup(vars=('geopotential',),name='z'),
    TestGroup(vars=('mean_sea_level_pressure',),name='msl'),
    TestGroup(vars=('specific_humidity',),name='q'),
    TestGroup(vars=('total_precipitation_6hr',),name='tp'),
    TestGroup(vars=('u_component_of_wind','v_component_of_wind'),name='u+v'),
    TestGroup(vars=('vertical_velocity',),name='w'),
    TestGroup(vars=(),name='control'), # Control -- no modification
    }

# Short names of output variables, based on the ECMWF short names
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

def print_headings(mse, test_vars,outfile):
    print('Base Date,Test Group,Test var,Sensitivity',file=outfile)
    # print('Base date, Test Group',end='',file=outfile)
    # for var in test_vars:
    #     for t in mse[var].time:
    #         t = t.data.astype('timedelta64[h]')
    #         if ('level' in mse[var].dims):
    #             for l in mse[var].level:
    #                 print(f", {var} {l.data}hPa {t}",end='',file=outfile)
    #         else:
    #             print(f", {var} {t}",end='',file=outfile)
    # print('',file=outfile)

if __name__ == '__main__':
    import datetime
    import xarray as xr
    import numpy as np
    import dateparser
    import numcodecs
    import trainer
    # Disable threading inside blosc
    numcodecs.blosc.use_threads = False

    ## Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--apath',type=str,dest='apath',default='../gdata_025_wb',help='Location of analysis data')
    parser.add_argument('--start-date',type=str,dest='start_date',default='1 Jan 2020 00:00',help='Starting date/time')
    parser.add_argument('--end-date',type=str,dest='end_date',default='31 Dec 2021 18:00',help='Ending date/time (inclusive)')
    # parser.add_argument('--forecast-length',type=int,dest='forecast_length',default=1)
    parser.add_argument('--to-csv',type=str,dest='csvpath',default=None,help='(optional) CSV file for scores')
    parser.add_argument('--num-samples',type=int,dest='num_samples',default=32,help='Number of samples to compute')
    parser.add_argument('--start-number',type=int,dest='start_sample',default=1,help='Starting sample number (for restart/parallelism)')
    parser.add_argument('--model-checkpoint',type=str,dest='model_checkpoint',default=None,help='Model checkpoint to load')
    parser.add_argument('--norm-factors',type=str,dest='norm_path',default='stats',
                        help='Path to the directory containing Graphcast normalization factors')

    args = parser.parse_args()

    # Forecast options: forecast length and dataset paths
    apath = args.apath

    # CSV output path
    csvpath = args.csvpath

    params_path = args.model_checkpoint
    num_samples = args.num_samples
    start_sample = args.start_sample

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
    
    import jax
    import dask
    import dask.distributed
    dask_client = dask.distributed.Client(processes=False)

    from forecast import generate_model
    from forecast.models import models_dict
    (model_config, task_config, params) = generate_model.load_model(params_path)

    # # Define a custom function to compute losses over forecasts _after_ they're made
    # per_variable_weights={
    #     "2m_temperature": 1.0,
    #     "10m_u_component_of_wind": 0.1,
    #     "10m_v_component_of_wind": 0.1,
    #     "mean_sea_level_pressure": 0.1,
    #     "total_precipitation_6hr": 0.1,
    # }
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

    from graphcast import xarray_jax

    # Build prediction operator
    # NOTE: Do not use float16 for prediction.  Reduced precision greatly increases run-to-run variance
    # even without perturbed initial conditions, which makes it harder to see relatively weak effects.
    predictor = generate_model.build_predictor_params(model_config,task_config,use_float16=False,
                                                      diffs_stddev_by_level = diffs_stddev_by_level, 
                                                      mean_by_level = mean_by_level,
                                                      stddev_by_level = stddev_by_level)

    (ds,_) = trainer.dataloader.open_databases(apath,None)
    ds = ds.sel(time=slice(start_date,end_date))

    rng = np.random.default_rng(seed=(0,ds.time.size))

    per_variable_weights = {}

    import jax
    ljit = jax.jit(graphcast_loss)

    forecast_length = 20 # 40
    target_lead_times = list(range(3,forecast_length,4))
    test_vars = ['mean_sea_level_pressure',
                 '2m_temperature',
                 'total_precipitation_6hr',
                 'geopotential',
                 'temperature',
                 'specific_humidity']
    test_levels = [50,100,150,200,250,300,400,500,600,700,850,925,1000]
    TARGET_ERROR=0.01
    
    trial_times = rng.permuted(ds.time.data[4:-forecast_length])

    target_variables = set(task_config['target_variables'])
    input_variables = set(task_config['input_variables'])
    latitude = ds.latitude
    longitude = ds.longitude

    def get_base_errors(orig,pert,diffs_stddev_by_level,per_variable_weights):
        # return unwrap_ds(ljit(orig[target_variables].isel(time=-1),pert[target_variables].isel(time=-1),diffs_stddev_by_level,per_variable_weights)[1])
        # print('base_errors',diffs_stddev_by_level)
        return unwrap_ds(
            ljit(
                orig[target_variables].isel(time=-1),
                pert[target_variables].isel(time=-1),
                diffs_stddev_by_level,
                per_variable_weights)[1])

    HEADING_PRINTED=False
    if (csvpath is not None):
        csvfile = open(csvpath,'w')
    else:
        import sys
        csvfile = sys.stdout
    
    for ict in range(num_samples):
        idx = ict+start_sample-1
        init_date = trial_times[idx]
        # perturb_date = init_date - np.timedelta64(86400,'s')
        perturb_date = init_date - 6*np.timedelta64(3600,'s')
        init_date = init_date.astype('datetime64[s]').astype(datetime.datetime)
        perturb_date = perturb_date.astype('datetime64[s]').astype(datetime.datetime)
        
        print(f"Processing {init_date.strftime('%Y-%m-%d %HZ')}, {idx+1} of {num_samples+start_sample-1}")

        (control_input, control_forcing, _) = trainer.dataloader.build_forecast(init_date,forecast_length,task_config,
                                                                                        latitude,longitude,
                                                                                        input_variables,target_variables,
                                                                                        ds,ds)
        
        (perturb_input, perturb_forcing, _) = trainer.dataloader.build_forecast(perturb_date,1,task_config,
                                                                                            latitude,longitude,
                                                                                            input_variables,target_variables,
                                                                                            ds,ds)
        
        (perturb0_input, perturb0_forcing) = dask.compute(perturb_input,perturb_forcing)
        reforecast = forecast_by_step(predictor,task_config,params,
                                    inputs=perturb0_input,forcings=perturb0_forcing,target_lead_times=[0])
        # reforecast['time'] = control_input.time
        # perturb_input = xr.concat([perturb_input.isel(time=[-1]),unwrap_ds(reforecast.isel(time=[0]))],dim='time').compute()
        # perturb_input['time'] = control_input.time

        perturb_input = reforecast[[v for v in reforecast.data_vars if v in perturb0_input.data_vars]]
        # And copy any forcings that become inputs
        for v in [v for v in perturb0_forcing.data_vars if v in perturb0_input.data_vars]:
            perturb_input[v] = perturb0_forcing[v]

        # Build new input field
        perturb_input = xr.concat([perturb0_input.isel(time=[-1,]), perturb_input],dim='time',coords='minimal',data_vars='minimal').compute()
        # Rewrite time array
        perturb_input['time'] = perturb0_input.time

        # (control_input, control_forcing, perturb_input) = dask.compute(control_input, control_forcing, perturb_input)
        (control_input, control_forcing) = dask.compute(control_input, control_forcing)

        base_errors = get_base_errors(control_input,perturb_input,diffs_stddev_by_level,{})

        # print(base_errors)
        # import sys
        # sys.exit(0)

        control_forecast = forecast_by_step(predictor,task_config,params,
                                            inputs=control_input,forcings=control_forcing,target_lead_times=target_lead_times)


        sorted_groups = [v[1] for v in sorted([ (q.name, q) for q in test_groups])]
        TARGET_ERROR = 1e-1
        for group in sorted_groups:
            # break
            # print(group)
            # Compute the expected MSE if the interpolation weight were 1 for the variables in this group, using just
            # the perturbed initial conditions
            base_error_group = sum(base_errors[var].data for var in group.vars)
            # Compute the target weight; note the square root because this is a mean squared error
            interp_weight = TARGET_ERROR/base_error_group**0.5 if base_error_group > 0 else 0
            # print(group.name, interp_weight)
            # print(group.name, base_error_group, interp_weight)

            # Construct the initial condition dataset for the test
            test_input = control_input.copy()
            for var in group.vars:
                test_input[var] = (1-interp_weight)*control_input[var] + interp_weight*perturb_input[var]

            # Run the perturbed forecast
            perturb_forecast = forecast_by_step(predictor,task_config,params,
                                                inputs=test_input,forcings=control_forcing,target_lead_times=target_lead_times)

            # Compute the MSE of the selected "physical" quantities
            mse = physical_mse(control_forecast,perturb_forecast,vars=test_vars,levels=test_levels)
            mse = mse.isel(batch=0) # Remove vestigial batch dimension

            # Print the heading if we haven't done so yet; this is easiest to compute after we have a sample
            # error dataset in hand
            if (not HEADING_PRINTED):
                print_headings(mse,test_vars,csvfile)
                HEADING_PRINTED=True
                
            # sensitivity[var] = mse
            # print(f'   {mse.mean_sea_level_pressure.data[0,0] : .2e}')
            # print(f"{init_date.strftime('%Y-%m-%d %HZ')}, {group.name:10s}",
            #     end='',file=csvfile)
            # for v in test_vars:
            #     for t in mse[v].time:
            #         if ('level' in mse[v].dims):
            #             for l in mse[v].level:
            #                 print(f", {mse[v].sel(level=l,time=t):.3e}",end='',file=csvfile)
            #         else:
            #             print(f", {mse[v].sel(time=t):.3e}",end='',file=csvfile)
            # print('',file=csvfile)    
            datestamp = init_date.strftime('%Y-%m-%d %HZ')
            for v in test_vars:
                varcode = short_names[v]
                for t in mse[v].time:
                    # t = t.data.astype('timedelta64[h]')
                    tstamp = int(t.data / np.timedelta64(1,'h'))
                    if ('level' in mse[v].dims):
                        for l in mse[v].level:
                            print(f'{datestamp},{group.name},{varcode}/{l.data}/{tstamp},{mse[v].sel(level=l,time=t):.3e}',file=csvfile)
                            
                    else:
                        print(f'{datestamp},{group.name},{varcode}/surf/{tstamp},{mse[v].sel(time=t):.3e}',file=csvfile)




            # print('Base Date,Test Group,Test var,Sensitivity')
            # frame = []
        level_vars = [v for v in target_variables if 'level' in control_input[v].dims]
        base_errs_level = graphcast_loss_by_level(control_input[level_vars].isel(time=-1),
                                                  perturb_input[level_vars].isel(time=-1),
                                                  diffs_stddev_by_level,weighted=False).isel(batch=0)

        # print(base_errs_level.geopotential)
        # import sys
        # sys.exit(0)                                                

        TARGET_ERROR=1e0

        for ilevel in range(base_errs_level['level'].size+1):
            # print(group)
            # Compute the expected MSE if the interpolation weight were 1 for the variables in this group, using just
            # the perturbed initial conditions
            if (ilevel < base_errs_level.level.size):
                base_error_group = sum(base_errs_level[var][ilevel] for var in base_errs_level.data_vars)
                # Compute the target weight; note the square root because this is a mean squared error
                interp_weight = TARGET_ERROR/base_error_group**0.5
                # print(group.name, interp_weight)

                # print(ilevel, base_error_group.data, interp_weight.data)

                # Construct the initial condition dataset for the test
                test_input = control_input.copy(deep=True)
                for var in level_vars:
                    test_input[var][:,:,ilevel,:,:] = (1-interp_weight)*control_input[var][:,:,ilevel,:,:] + \
                                                    interp_weight*perturb_input[var][:,:,ilevel,:,:]
                    
                groupname = f'L{int(control_input.level[ilevel]):d}'
            else:
                # print(ilevel,'control')
                test_input = control_input.copy()
                groupname = 'L0'

            # Run the perturbed forecast
            perturb_forecast = forecast_by_step(predictor,task_config,params,
                                                inputs=test_input,forcings=control_forcing,target_lead_times=target_lead_times)

            # Compute the MSE of the selected "physical" quantities
            mse = physical_mse(control_forecast,perturb_forecast,vars=test_vars,levels=test_levels)
            mse = mse.isel(batch=0) # Remove vestigial batch dimension

            # Print the heading if we haven't done so yet; this is easiest to compute after we have a sample
            # error dataset in hand
            # if (not HEADING_PRINTED):
            #     print_headings(mse,test_vars,sys.stdout)
            #     HEADING_PRINTED=True
                
            # sensitivity[var] = mse
            # print(f'   {mse.mean_sea_level_pressure.data[0,0] : .2e}')
            datestamp = init_date.strftime('%Y-%m-%d %HZ')
            # print(f"{init_date.strftime('%Y-%m-%d %HZ')}, {group.name:10s}",
            #       end='')
            for v in test_vars:
                varcode = short_names[v]
                for t in mse[v].time:
                    # t = t.data.astype('timedelta64[h]')
                    tstamp = int(t.data / np.timedelta64(1,'h'))
                    if ('level' in mse[v].dims):
                        for l in mse[v].level:
                            print(f'{datestamp},{groupname},{varcode}/{l.data}/{tstamp},{mse[v].sel(level=l,time=t):.3e}',file=csvfile)
                            
                    else:
                        print(f'{datestamp},{groupname},{varcode}/surf/{tstamp},{mse[v].sel(time=t):.3e}',file=csvfile)
            csvfile.flush()
    if (csvpath is not None):
        csvfile.close()
