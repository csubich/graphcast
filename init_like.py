#!/usr/bin/env python3

# init_like.py -- given an existing Graphcast-style checkpoint, create a new, randomly-initialized checkpoint
# that follows the same specification and structure

if __name__ == '__main__':
    import argparse

    ## Command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dpath',type=str,dest='dpath',help='Location of template dataset',required=True)
    parser.add_argument('--date',type=str,dest='date',default='1 Jan 2020 00:00',help='Date to use for template data')
    parser.add_argument('--input-checkpoint',type=str,dest='model_checkpoint',default=None,help='Model checkpoint to load as reference',required=True)
    parser.add_argument('--output-checkpoint',type=str,dest='output_checkpoint',default='out.ckpt',help='Output file name')
    parser.add_argument('--seed',type=int,default=0,dest='seed',help='Random seed for initialization')


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

    

    start_date = dateparser.parse(args.date,
                                  ['%Y%m%d%H',  # Also parse YYYYMMDDHH (ISO 8601-2004)
                                   '%Y%m%d%HZ', # ... with UTC marker
                                   '%Y%m%dT%H', # and YYYYMMDDTHH (ISO 8601-2019)
                                   '%Y%m%dT%HZ',# ... with UTC marker
                                  ])

    print(f'Initializing new random model checkpoint, following the pattern of {args.model_checkpoint}')
    print(f'  with the data schema of {args.dpath} for date {start_date.strftime("%Y-%m-%d %H:00")}')

    from forecast import generate_model
    (model_config, task_config, params) = generate_model.load_model(args.model_checkpoint)


    dbase,_ = trainer.dataloader.open_databases(args.dpath,None) 

    model_latitude = xr.DataArray(np.linspace(-90,90,int(1+180/model_config['resolution']),dtype=np.float32),dims='latitude')
    model_latitude = model_latitude.assign_coords({'latitude' : model_latitude})
    model_longitude = xr.DataArray(np.linspace(0,360-model_config['resolution'],int(360/model_config['resolution']),dtype=np.float32),
                                dims='longitude')
    model_longitude = model_longitude.assign_coords({'longitude' : model_longitude})

    input_variables = list(task_config['input_variables'])
    target_variables = list(task_config['target_variables'])

    (inputs,forcings,targets) = trainer.dataloader.build_forecast(start_date,1,task_config,
                                                              model_latitude,model_longitude,input_variables,target_variables,
                                                              dbase,dbase)
    (inputs,forcings,targets) = dask.compute(inputs,forcings,targets)
    inputs = inputs.drop_vars('datetime')
    targets = targets.drop_vars('datetime')
    forcings = forcings.drop_vars('datetime')

    random_params = generate_model.init_params(model_config,task_config,inputs,targets,forcings,seed=args.seed)

    print(f'Random paramters generated, writing to {args.output_checkpoint}')
    with open(args.output_checkpoint,'wb') as cfile:
        from graphcast import checkpoint
        import graphcast
        checkpoint.dump(cfile,graphcast.graphcast.CheckPoint(params=random_params,
                                                            model_config=model_config,
                                                            task_config=task_config,
                                                            description=f'Graphcast-style model parameters',license="For ECCC Internal Use, from random initialization"))



