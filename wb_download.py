#!/usr/bin/env python
# Script to download ERA5 data from WeatherBench for a specified year/month, 
# storing 1/4-degree and 1-degree versions in separate directories


if __name__ == '__main__':
    import argparse

    ## Command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--year',dest='year',type=int,default=None,required=True)
    parser.add_argument('--month',dest='month',type=int,default=None,required=True)

    parser.add_argument('--dir-1deg',dest='dir_1deg',type=str,default='era5_1deg')
    parser.add_argument('--dir-025deg',dest='dir_025deg',type=str,default='era5_025deg')

    args = parser.parse_args()

    import warnings

    # Supress ECCodes warning
    import xarray as xr
    import numpy as np
    import dask
    import dask.distributed
    import forecast.encabulator
    import forecast.generate_model
    import datetime

    # Destination paths
    dest_path_1deg = args.dir_1deg
    dest_path_025deg = args.dir_025deg

    # Target year and month for download
    year = args.year
    month = args.month

    # Graphcast model to use as a template, defining important input and target variables
    gc_checkpoint = 'params/GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz'

    # Remote URL for WeatherBench data
    wb_url = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'

    # Required for xarray/zarr; 'trust_env' ensures that they will properly use the environment-
    # configured proxy server
    storage_options = {'session_kwargs' : {'trust_env' : True}} 

    # Load the Graphcast model and get required variables
    
    (model_config, task_config, params) = forecast.generate_model.load_model(gc_checkpoint)
    input_variables = list(task_config['input_variables'])
    target_variables = list(task_config['target_variables'])
    forcing_variables = list(task_config['forcing_variables'])

    # Make the target repositories, if they don't already exist
    import os
    os.system(f'mkdir -vp {dest_path_025deg}')
    os.system(f'mkdir -vp {dest_path_1deg}')

    # Create a Dask Client to handle data download/processing with parallelism and automatic memory management
    import dask.config
    dask.config.set(
        {'distributed.worker.memory.target':False,
        'distributed.worker.memory.spill':False,}
    )
    # Suppress 'HTTP port already in use' warning
    with warnings.catch_warnings(action="ignore"):
        Client = dask.distributed.Client(processes=False,threads_per_worker=20)

    # Open the WeatherBench dataset, and set variables to chunk along only the time dimension
    # Supress ECCodes warning
    with warnings.catch_warnings(action="ignore"):
        ds = xr.open_dataset(wb_url,engine='zarr',storage_options=storage_options,chunks={'time':1,'latitude':-1,'longitude':-1,'level':-1})

    # Figure out which variables we can download from the dataset, based on variables the model needs and
    # the variables the WB dataset has
    model_vars = set(input_variables).union(set(target_variables)).union(set(forcing_variables))
    download_vars = list(set.intersection(set(ds.data_vars),model_vars))

    # Define the output variables including total precipitation, and set the proper encoding to compress them
    output_vars = download_vars + ['total_precipitation_6hr']
    compressed_vars = set(output_vars) - {'geopotential_at_surface','land_sea_mask','total_precipitation_6hr'} - set(forcing_variables)
    encoding = {v : {'compressor' : forecast.encabulator.LayerQuantizer(nbits=16)} for v in compressed_vars}

    # print(f'Downloading: {year} / {month}')

    # Define the start and end periods for the download, and define some handy timedeltas
    dt_6h = datetime.timedelta(hours=6)
    dt_1h = datetime.timedelta(hours=1)
    date_start = datetime.datetime(year,month,1,0) # Start of the period
    # Define the end of the period by subtracting 6h from the beginning of the next month
    if (month == 12): # January is next month
        date_end = datetime.datetime(year+1,1,1,0) - dt_6h
    else:
        date_end = datetime.datetime(year,month+1,1,0) - dt_6h

    # Most of the output is just taken from the dataset, from the start to end every 6h.
    # To slice by 6h increments, it seems easiest to use isel; slice with a timedelta stride errors.
    output_ds = ds[download_vars].sel(time=slice(date_start,date_end)).isel(time=slice(None,None,6))

    # Precipitation is a bit special, since it must be accumulated over the 6h period ending at
    # the specified time.  Thus, we want to select hourly total_precipitation, beginning 5h before
    # date_start and ending with date_end
    ds_precip = ds['total_precipitation'].sel(time=slice(date_start-5*dt_1h,date_end))

    # Group this data together using xr.goupby_bins
    assert(ds_precip.time.size % 6 == 0)
    ds_precip_grouped = ds_precip.groupby_bins('time',ds_precip.time.size//6)
    ds_precip_sum = ds_precip_grouped.sum()

    # Rename the 'time_bins' dimension to 'time', and reassign meaningful valid-time values
    ds_precip_sum = ds_precip_sum.rename(time_bins='time')
    ds_precip_sum['time'] = ds_precip['time'][5::6]
    output_ds['total_precipitation_6hr'] = ds_precip_sum

    # The Graphcast model expects latitude and longitude to be increasing, [-90 -> +90], so 
    # it's better to perform any reordering here rather than every time the data is read
    if output_ds.latitude.data[1] - output_ds.latitude.data[0] < 0:
        output_ds = output_ds.isel(latitude=slice(None,None,-1))
    if output_ds.longitude.data[1] - output_ds.longitude.data[0] < 0:
        output_ds = output_ds.isel(longitude=slice(None,None,-1))

    # Define output directories
    out_zarr_025deg = f'{dest_path_025deg}/{year}/{month:02d}'
    out_zarr_1deg = f'{dest_path_1deg}/{year}/{month:02d}'

    # Subsample the quarter-degree output to give the 1-degree version
    output_ds_1deg = output_ds.isel(latitude=slice(None,None,4)).isel(longitude=slice(None,None,4))

    # Write the output; use compute=False to give dask the opportunity to mutually optimize the
    # dataset operations
    # print(f'Writing to {out_zarr_025deg} and {out_zarr_1deg}')
    tic = datetime.datetime.now()
    delayed_025deg = output_ds.to_zarr(out_zarr_025deg,encoding=encoding,compute=False)
    delayed_1deg = output_ds_1deg.to_zarr(out_zarr_1deg,encoding=encoding,compute=False)

    (delayed_025deg,delayed_1deg) = dask.optimize(delayed_025deg,delayed_1deg)

    # Import faulthanlder, which will act as a watchdog to dump a stacktrace in the event that things hang
    import faulthandler
    # Set the faulthandler to exit after half an hour, which should act as a failsafe to keep the downloads
    # moving if one month stalls
    faulthandler.dump_traceback_later(1800,exit=True)

    try:
        Client.compute((delayed_025deg,delayed_1deg),sync=True)
    except Exception as e:
        import sys
        print(f'Error downloading {year}-{month}, exception {e=}',file=sys.stderr)
        sys.exit(1)

    toc = datetime.datetime.now()
    print(f'{year}-{month} done in {(toc-tic).total_seconds():2}s')
