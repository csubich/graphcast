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
import cdsapi as cs
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--download-dir',type=str,dest='download_dir',required=True,default=None)
parser.add_argument('--precip-only',action='store_true',dest='precip_only',required=False,default=False,help='Download precipitation only')
parser.add_argument('year',type=int)
parser.add_argument('month',type=int)

args = parser.parse_args()

year = args.year
month = args.month

precip_only = args.precip_only

download_path = Path(f'{args.download_dir}/{year}/{month:02d}')

download_path.mkdir(parents=True,exist_ok=True)

# Variables to download from CDS
vars_3d = [
                'geopotential', 'specific_humidity', 'temperature',
                'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
            ]
vars_2d = [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                'geopotential', 'land_sea_mask', 'mean_sea_level_pressure',
                'toa_incident_solar_radiation',
            ]
vars_precip = ['total_precipitation',]

times_6h = [f'{hr:02d}:00' for hr in [0,6,12,18]]
times_1h = [f'{hr:02d}:00' for hr in range(0,24)]

pressure_levels = [
                '1', '2', '3', '5', '7', '10', '20', '30', '50',
                '70', '100', '125', '150', '175', '200', '225', '250', '300',
                '350', '400', '450', '500', '550', '600', '650', '700', '750',
                '775', '800', '825', '850', '875', '900', '925', '950', '975',
                '1000',
            ]

maxday = 31
if month in [9, 4, 6, 11]: maxday = 30
if month == 2 and (year % 4 != 0): maxday = 28
if month == 2 and (year % 4 == 0): maxday = 29

# days_10 = [f'{day}' for day in range(1,11)]
# days_20 = [f'{day}' for day in range(11,21)]
# days_30 = [f'{day}' for day in range(21,maxday+1)]

days_31 = [f'{day}' for day in range(1,maxday+1)]

# Download 3D data in batches, as grib, because xarray doesn't love
# deferred opens of grib files
days_3d = []
fname_3d = []
sday = 1
while sday < maxday:
    if (maxday - sday < 6):
        days = list(range(sday,maxday+1))
    else:
        days = list(range(sday,sday+4))
    if (days[0] > maxday): break
    days_3d.append(days)
    fname_3d.append(f'{str(download_path)}/era5_3d_{days[0]:02d}.grib')
    sday = max(days)+1

client = cs.Client()

download_2d = f'{str(download_path)}/era5_2d.nc'
download_precip = f'{str(download_path)}/era5_precip.nc'


# Get 3D data
import os
for (fname, daylist) in zip(fname_3d,days_3d):
    if precip_only: break
    if (os.path.exists(fname)):
        print(f'{fname} already exists, skipping')
        continue

    client.retrieve('reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'variable': vars_3d,
            'pressure_level': pressure_levels,
            'year': f'{year}',
            'month': f'{month:02d}',
            'day': daylist,
            'time': times_6h,
            'format': 'grib',
        },
        fname)

# Get 2D data
if (not precip_only):
    if (not os.path.exists(download_2d)):
        client.retrieve('reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'variable': vars_2d,
                'year': f'{year}',
                'month': f'{month:02d}',
                'day': days_31,
                'time': times_6h,
                'format': 'netcdf',
            },
            download_2d)
    else:
        print(f'{download_2d} already exists, skipping')

# # # Get precip
if (not os.path.exists(download_precip)):
    client.retrieve('reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': vars_precip,
            'year': f'{year}',
            'month': f'{month:02d}',
            'day': days_31,
            'time': times_1h,
            'format': 'netcdf',
        },
        download_precip)
else:
    print(f'{download_precip} already exists, skipping')
