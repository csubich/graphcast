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


import argparse
import os
# import random

# os.sched_setaffinity(0,random.sample(list(os.sched_getaffinity(0)),1))

from forecast import data_reader, forecast_prep, generate_model
from forecast.models import models, models_dict, model_descriptions
from forecast.databases import dbase, dbase_dict, dbase_descriptions
from forecast.cache_manager import CacheManager
import forecast.encabulator # Data compressor, for cache
import datetime
from pathlib import Path
import dateparser

# Parse command-line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--model',choices=[m.name for m in models],default='era5_100',dest='model',
                    help='Pre-trained GraphCast model.  Available options:\n' + model_descriptions)
parser.add_argument('--dbase',choices=[d.name for d in dbase],default='wb_era5_100_13',dest='dbase',
                    help='Analysis databse (pre-configured).  Available options:\n' + dbase_descriptions)
parser.add_argument('--forecast-length',type=int,default=1,dest='flength',
                    help='Number of 6-hour steps to forecast (default 1)')
parser.add_argument('--cache-dir',type=str,default='data_cache',dest='cache_dir',
                    help='Directory to store cached analysis inputs')
parser.add_argument('--no-cache',dest='use_cache',action='store_false',
                    help='Do not use or add to cached analysis inputs (ignore cache-dir).')
parser.add_argument('--cpu-only',dest='use_gpu',action='store_false',
                    help='Use CPU only; do not try to use any GPU.')
parser.add_argument('--prep-only',dest='prep_only',action='store_true',
                    help='Download and preprocess data for the requested forecast dates, but do not run the forecast.')
parser.add_argument('--dbase-path',dest='dbase_path',default=None,
                    help='Custom database path for xarr or fstd databases')
parser.add_argument('--verbose',dest='verbose',action='store_true',
                    help='Verbose output')
parser.add_argument('idates',nargs='+',type=str,
                    help='Initialization dates of desired forecasts')

args = parser.parse_args()

forecast_path='forecast_outputs'

# Load GraphCast model
mymodel = models_dict[args.model]
print(f'Loading model at {mymodel.path}')
(model_config, task_config, params) = generate_model.load_model(mymodel.path)
if (not args.prep_only):
    predictor = generate_model.build_predictor(model_config,task_config,params,use_gpu=args.use_gpu)
else:
    predictor = None

# Ensure that cache and forecast paths exist
if (args.use_cache):
    # Ensure the cache directory exists
    cache_dir = args.cache_dir
    Path(cache_dir).mkdir(parents=True,exist_ok=True)

    # Set an encoder object to compress downloaded variables
    compressed_vars = set(task_config['input_variables']) - \
                      set(task_config['forcing_variables']) - \
                      {'geopotential_at_surface','land_sea_mask','total_precipitation_6hr'}
    encoding = {v : {'compressor' : forecast.encabulator.LayerQuantizer(nbits=16)} for v in compressed_vars}
    cache = CacheManager(cache_dir,encoders=encoding)
else:
    cache = None

Path(forecast_path).mkdir(parents=True,exist_ok=True)


# Load database
mydbase = dbase_dict[args.dbase]
if (mydbase.path is not None):
    myreader = mydbase.reader(mydbase.path,cache,task_config,model_config,verbose=args.verbose)
else:
    myreader = mydbase.reader(args.dbase_path,cache,task_config,model_config,verbose=args.verbose)
# myreader = data_reader.XArrayReader(mydbase.path,cache_dir,task_config,model_config)

for forecast_string in args.idates:
    # Use the DateParser library to parse dates from the command line
    idatetime = dateparser.parse(forecast_string,
                                 ['%Y%m%d%H',  # Also parse YYYYMMDDHH (ISO 8601-2004)
                                  '%Y%m%d%HZ', # ... with UTC marker
                                  '%Y%m%dT%H', # and YYYYMMDDTHH (ISO 8601-2019)
                                  '%Y%m%dT%HZ',# ... with UTC marker
                                 ])
    if (idatetime is None):
        print(f'Unknown date {forecast_string}, skipping')
        continue

    tic = datetime.datetime.now()
    print(f'Building forecast fields for {idatetime.strftime("%Y-%m-%dT%H:00Z")} (6–{6*args.flength}h)')

    (inputs, forcings, targets) = forecast_prep.forecast_setup(idatetime,myreader,args.flength)

    setup_toc = datetime.datetime.now()
    elapsed = setup_toc - tic
    print(f'Fields generated in {elapsed.seconds + 1e-6*elapsed.microseconds:.3f}s')
    
    if (args.prep_only):
        continue
        
    forecast = predictor(inputs=inputs, targets=targets, forcings=forcings)
    forecast_toc = datetime.datetime.now()
    elapsed = forecast_toc - setup_toc
    print(f'Forecast generated in {elapsed.seconds + 1e-6*elapsed.microseconds:.3f}s')

    # Rewrite 'time' coordinate of forcing to an absolute rather than relative basis
    forecast.coords['lead_time'] = forecast.coords['time']
    forecast.coords['time'] = forcings.coords['datetime'].data
    # Add an analysis time singleton
    forecast.coords['analysis_time'] = inputs.coords['datetime'].max()

    # Write forecast
    forecast_filename = f'{forecast_path}/{idatetime.strftime("%Y%m%dT%H")}.nc'
    if (os.path.exists(forecast_filename)):
        print(f'{forecast_filename} already exists, removing for new output')
        os.remove(forecast_filename)
    forecast.to_netcdf(forecast_filename,mode='w',format='NETCDF4',compute=True)
    output_toc = datetime.datetime.now()
    elapsed = output_toc - forecast_toc
    print(f'Forecast written in {elapsed.seconds + 1e-6*elapsed.microseconds:.3f}s')
