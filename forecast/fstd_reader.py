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


# Module to read and process .fstd files
# from forecast import data_reader, forecast_prep, generate_model
# from forecast.models import models, models_dict, model_descriptions
# from forecast.databases import dbase, dbase_dict, dbase_descriptions
import xarray as xr
import numpy as np
import xesmf as xe
import datetime
import pathlib
from collections import namedtuple
import fstd2nc
import rpnpy
from forecast import data_reader
from forecast.forecast_variables import DERIVED_VARIABLES
from forecast.cache_manager import CacheManager

# Suppress a spammy warning message inside VGrid
rpnpy.librmn.librmn.Lib_LogLevelNo(5,9)

from dataclasses import dataclass
@dataclass(frozen=True)
class fstdvar:
    name_fstd : str # Variable name in standard file
    name_xarray : str # Variable name in the constructed xarray
    source_file : str # Source file: 'analysis_pres, analysis_hyb, geophy, forecast_0, forecast_6'
    conversion : None | tuple[float,float] =None # Unit conversion.  If present, outvar = invar*conversion[0] + conversion[1]
    surface : bool = False # Whether variable is surface (true) or 3D (false)
    vector_u : bool = False # Whether variable is a u-component vector
    vector_v : bool = False # Whether variable is a v-component vector

fstdvars = [
    fstdvar(name_fstd='GZ',name_xarray='geopotential',source_file='analysis_pres',conversion=(98.0616,0),surface=True),
    fstdvar(name_fstd='ME',name_xarray='geopotential_at_surface',source_file='geophy',conversion=(9.80616,0),surface=True),
    fstdvar(name_fstd='MG',name_xarray='land_sea_mask',source_file='geophy',surface=True),
    fstdvar(name_fstd='WW',name_xarray='vertical_velocity',source_file='forecast_pres'),
    # Convert from knots to m/s
    fstdvar(name_fstd='UU',name_xarray='10m_u_component_of_wind',source_file='analysis_hyb',surface=True,vector_u=True,conversion=(0.5144,0)),
    fstdvar(name_fstd='VV',name_xarray='10m_v_component_of_wind',source_file='analysis_hyb',surface=True,vector_v=True,conversion=(0.5144,0)),
    fstdvar(name_fstd='UU',name_xarray='u_component_of_wind',source_file='analysis_pres',vector_u=True,conversion=(0.5144,0)),
    fstdvar(name_fstd='VV',name_xarray='v_component_of_wind',source_file='analysis_pres',vector_v=True,conversion=(0.5144,0)),
    # Convert from C to K
    fstdvar(name_fstd='TT',name_xarray='2m_temperature',source_file='analysis_hyb',surface=True,conversion=(1,273.15)),
    fstdvar(name_fstd='TT',name_xarray='temperature',source_file='analysis_pres',conversion=(1,273.15)),
    fstdvar(name_fstd='HU',name_xarray='specific_humidity',source_file='analysis_pres'),
    fstdvar(name_fstd='PR',name_xarray='total_precipitation_6hr',source_file='forecast_hyb',surface=True),
    # Convert from hPa to Pa
    fstdvar(name_fstd='PN',name_xarray='mean_sea_level_pressure',source_file='analysis_pres',surface=True,conversion=(100,0))
]

regridder_cache = {}

def make_regridder(dset_out,dset_yin,dset_yang,file_yin=None,file_yang=None,load=False):
    if (load):
        # Load the regridder weights from disk; this will error if the
        # file does not exist
        regridder_yin = xe.Regridder(dset_yin,dset_out,'conservative',
                                     unmapped_to_nan=True,
                                     reuse_weights=True,
                                     weights=file_yin,filename=file_yin)
        regridder_yang = xe.Regridder(dset_yang,dset_out,'conservative',
                                      unmapped_to_nan=True,
                                      reuse_weights=True,
                                      weights=file_yang,filename=file_yang)
    else:
        # Create new regridders, using the filename to write the weights
        # to disk
        regridder_yin = xe.Regridder(dset_yin,dset_out,'conservative',
                                     unmapped_to_nan=True,
                                     filename=file_yin)
        regridder_yang = xe.Regridder(dset_yang,dset_out,'conservative',
                                      unmapped_to_nan=True,
                                      filename=file_yang)
    # Construct the weight field necessary to average over the yin/yang
    # overlap region
    yyg_weights = (np.nan_to_num(regridder_yin(np.ones(dset_yin.lat.shape))) + \
                  np.nan_to_num(regridder_yang(np.ones(dset_yin.lat.shape)))).astype(np.float32)
    return(regridder_yin,regridder_yang,yyg_weights)

def get_regridders(dset_out,yin_view,yang_view):
    import pathlib, hashlib, base64
    rot_pole = namedtuple('RotatedPole',yin_view['rotated_pole'].attrs.keys())

    # Construct a key to uniquely specify this regridder
    key = (yin_view.lat.shape,  # Input shape
           (dset_out.lat.shape[0],dset_out.lon.shape[-1]), # Output shape
           rot_pole(*yin_view['rotated_pole'].attrs.values()), # Yin grid rotation params
           rot_pole(*yang_view['rotated_pole'].attrs.values())) # Yang grid rotation params
    
    # print(f'Getting regridders for input {key[0]}, output {key[1]}')
    
    if (key) in regridder_cache:
        # print('Regridders found in cache')
        return regridder_cache[key]

    # Key does not exist.  Is it on disk?
    pathlib.Path('regrid_weights').mkdir(exist_ok=True)

    # Encode the key information into a stable hash, so that we can write the regridder
    # weights to disk without risk of loading the wrong one if the source grid changes
    keyhash = base64.b32encode(hashlib.sha1(str(key).encode('utf8')).digest()).decode('utf8')
    keyfile_yin = f'regrid_weights/from_yin_{yin_view.lat.shape[0]}x{yin_view.lat.shape[1]}_to_{dset_out.lat.shape[0]}x{dset_out.lon.shape[-1]}_hash_{keyhash}.nc'
    keyfile_yang = f'regrid_weights/from_yang_{yin_view.lat.shape[0]}x{yin_view.lat.shape[1]}_to_{dset_out.lat.shape[0]}x{dset_out.lon.shape[-1]}_hash_{keyhash}.nc'
    
    if (pathlib.Path(keyfile_yin).exists() and pathlib.Path(keyfile_yang).exists()):
        # Yes, load from disk
        # print('Regridders exist on disk')
        new_regridder = make_regridder(dset_out,yin_view,yang_view,keyfile_yin,keyfile_yang,load=True)
    else:
        # No, make new
        # print('Regridders do not exist on disk; make new')
        new_regridder = make_regridder(dset_out,yin_view,yang_view,keyfile_yin,keyfile_yang,load=False)

    # Assign regridder to the cache dictionary
    regridder_cache[key] = new_regridder
    return new_regridder

def transform_2d_winds(uvar,vvar,input_crs,target_crs):
# '''Transforms 2D wind fields (m/s) on a rotated coordinate system to the target coordinate system
# via a finite difference approximation on the coordinate transform'''
    uvar_stack = uvar.stack(coord=['rlat','rlon'])
    vvar_stack = vvar.stack(coord=['rlat','rlon'])
    
    scale_raw_speed = False
    
    # To avoid roundoff problems, compute the transformation of wind vectors
    # with a normalized speed, then scale the output back to the wind speed
    # of the source vectors.
    
    if (scale_raw_speed):
        # Scale in terms of physical speed
        source_speed = (uvar_stack**2 + vvar_stack**2)**0.5
        speed_eps = 300.0
        
        uvar_ms = uvar_stack*speed_eps/source_speed
        vvar_ms = vvar_stack*speed_eps/source_speed
    else:
        # Scale in terms of wind image (degrees/sec)
        uvar_ms = uvar_stack
        vvar_ms = vvar_stack
        
    # Convert to degrees/sec
    uvar_deg = uvar_ms / (input_crs.ellipsoid.semi_major_metre * np.cos(uvar_stack.rlat*np.pi/180)) * 180/np.pi
    vvar_deg = vvar_ms / (input_crs.ellipsoid.semi_major_metre) * 180/np.pi
    
    # Use cartopy.crs.transform_points to get wind vectors in the target CRS by finite difference.
    # cartopy.crs.transform_vectors is broken, see https://github.com/SciTools/cartopy/issues/2279
    
    if (scale_raw_speed):
        input_rlon0 = uvar_stack.rlon - 0.5*uvar_deg
        input_rlon1 = uvar_stack.rlon + 0.5*uvar_deg
        # Clamp latitude to ±90 degrees.  This will result in inaccurate
        # conversions for north/south winds at the poles, but those are
        # ambiguous to begin with.
        input_rlat0 = np.clip(uvar_stack.rlat - 0.5*vvar_deg,-90,90)
        input_rlat1 = np.clip(uvar_stack.rlat + 0.5*vvar_deg,-90,90)
    else:
        coord_eps = 3e-2 # (Normalie lengths to 0.01deg)
        source_length_deg = (uvar_deg**2 + vvar_deg**2)**0.5
        input_rlon0 = uvar_stack.rlon - 0.5*uvar_deg*coord_eps/source_length_deg
        input_rlon1 = uvar_stack.rlon + 0.5*uvar_deg*coord_eps/source_length_deg
        # Clamp latitude to ±90 degrees.  This will result in inaccurate
        # conversions for north/south winds at the poles, but those are
        # ambiguous to begin with.
        input_rlat0 = np.clip(uvar_stack.rlat - 0.5*vvar_deg*coord_eps/source_length_deg,-90,90)
        input_rlat1 = np.clip(uvar_stack.rlat + 0.5*vvar_deg*coord_eps/source_length_deg,-90,90)
    
    # Convert the rotated points to the lat/lon grid
    target_pts0 = target_crs.transform_points(input_crs,input_rlon0,input_rlat0)
    target_pts1 = target_crs.transform_points(input_crs,input_rlon1,input_rlat1)
    
    dpts = target_pts1 - target_pts0
    dx = dpts[:,0]
    dy = dpts[:,1]
    # Correct dx to undo any grid-wrapping in longitude
    unwrap_pt = np.abs(dx) > 180
    dx[unwrap_pt] = np.mod(dx[unwrap_pt]+180,360)-180
    
    
    if (not scale_raw_speed):
        # Re-scale winds to the correct length.  Using source_length_deg also
        # brings this back into the DataArray type with full coordinate
        # information, rather than a bare numpy array
        uout_deg = dx* source_length_deg / coord_eps
        vout_deg = dy* source_length_deg / coord_eps
        uout_ms = uout_deg * target_crs.ellipsoid.semi_major_metre * np.cos(uvar_stack.lat*np.pi/180) * np.pi/180
        vout_ms = vout_deg * target_crs.ellipsoid.semi_major_metre * np.pi/180
    else:
        uout_deg = dx
        vout_deg = dy
        # Convert degrees/sec back to m/s, using the nonrotated latitude
        uout_ms = uout_deg * target_crs.ellipsoid.semi_major_metre * np.cos(uvar_stack.lat*np.pi/180) * np.pi/180 * source_speed/speed_eps
        vout_ms = vout_deg * target_crs.ellipsoid.semi_major_metre * np.pi/180 * source_speed/speed_eps
    
    # Unstack the coordinate for returning
    uout = uout_ms.unstack('coord').astype(uvar.dtype)
    vout = vout_ms.unstack('coord').astype(vvar.dtype)

    return(uout,vout)

def transform_winds_by_basis(uvar,vvar,input_crs,target_crs):
    '''Transforms 4D fields (uvar, vll) on a rotated coordinate system to the target coordinate system
    by building a transformation basis'''
    
    # Construct basic DataArrays to later form [1,0] and [0,1] vector fields
    slice_2d = (0,)*(uvar.data.ndim-2) + (...,) # Abstract slice to grab the last two dimensions only
    ones_rot = xr.DataArray(np.ones(uvar[slice_2d].shape),coords=uvar[slice_2d].coords)
    zeros_rot = xr.DataArray(np.zeros(uvar[slice_2d].shape),coords=uvar[slice_2d].coords)
    
    # Get the basis vector fields by using the 2D transformation
    (u_to_u, u_to_v) = transform_2d_winds(ones_rot,zeros_rot,input_crs,target_crs)
    (v_to_u, v_to_v) = transform_2d_winds(zeros_rot,ones_rot,input_crs,target_crs)
    
    # The basis vectors have a shape that can be broadcast with uvar/vvar, but
    # by default it will re-add the lost dimensions to the end of the dimension list
    # rather than the beginning.  With a specific slicing operator that inclues
    # np.newaxis, we can force the new dimensions to the start.
    newaxis_2d = (np.newaxis,)*(uvar.data.ndim-2) + (slice(None),slice(None))
    u_target = (u_to_u.data[newaxis_2d]*uvar + v_to_u.data[newaxis_2d]*vvar).astype(uvar.dtype)
    v_target = (u_to_v.data[newaxis_2d]*uvar + v_to_v.data[newaxis_2d]*vvar).astype(vvar.dtype)
    
    return(u_target,v_target)

def vertical_bisection_search(source,target):
    # Perform a bisection search on 4D arrays of shape [1,Nz,Nlat,Nlon]
    # to find the floating-point vertical level where source[0,idx[:,:],:,:] ~= target
    
    idx_lb = 0*target
    idx_ub = 0*target + source.shape[1]-1
    
    lat_idx = np.arange(source.shape[-2])[np.newaxis,:,np.newaxis]
    lon_idx = np.arange(source.shape[-1])[np.newaxis,np.newaxis,:]
    
    
    for iter in range(9):
        # bisection search: pick a point halfway between lower and upper bounds
        idx_trial = 0.5*(idx_lb + idx_ub)
    
        trial_floor = np.floor(idx_trial).astype(int)
        trial_ceil = np.ceil(idx_trial).astype(int)
        trial_frac = np.mod(idx_trial,1)
        
        test = (1-trial_frac)*source[0,trial_floor,lat_idx,lon_idx] + trial_frac*source[0,trial_ceil,lat_idx,lon_idx]
        
        mark = (test > target)
        # GZ decreases towards higher indices.  
        # If test_gz > target_gz, then the target must be above the guessed value; replace the LB
        idx_lb[mark] = idx_trial[mark]
        # Otherwise, the target is below the guessed value; replace the UB
        idx_ub[~mark] = idx_trial[~mark]

    # # Now, the index should be determined within 1; perform a linear interpolation.

    # Interpolated value at the floating-point lower bound
    lb_floor = np.floor(idx_lb).astype(int)
    lb_ceil = np.ceil(idx_lb).astype(int)
    lb_frac = np.mod(idx_lb,1)
    val_lb = (1-lb_frac)*source[0,lb_floor,lat_idx,lon_idx] + lb_frac*source[0,lb_ceil,lat_idx,lon_idx]

    # Interpolated value at the floating-point upper bound
    ub_floor = np.floor(idx_ub).astype(int)
    ub_ceil = np.ceil(idx_ub).astype(int)
    ub_frac = np.mod(idx_ub,1)
    val_ub = (1-ub_frac)*source[0,ub_floor,lat_idx,lon_idx] + ub_frac*source[0,ub_ceil,lat_idx,lon_idx]

    # Interpolate between val[lb] and val[ub]
    dval = val_ub - val_lb
    tfrac = (target-val_lb)/dval    
    # test = val_lb + tfrac*dval # test = target by construction
    
    idx_trial = idx_lb + tfrac*(idx_ub-idx_lb)
    
    # If the interpolated value falls outside the range, clip it
    test = np.select( (idx_trial < idx_lb, idx_trial > idx_ub), (val_lb, val_ub), target)
    idx_trial = np.clip(idx_trial,idx_lb,idx_ub)

    # Return the floating-point index and the interpolated value there
    return(idx_trial,test)

class FSTDReader():
    '''Class to wrap reading of several .fstd files, building an initial condition dataset'''
    def __init__(self,dbase_path: [str,None], cache: [CacheManager,None], task_config, model_config, dbase_type='zarr',cache_type='netcdf_dir',verbose=False):
        self.task_config = task_config
        self.model_config = model_config
        self.dbase_path = pathlib.Path(dbase_path)
        self.verbose = verbose

        self.cache = cache

        (model_lat, model_lon) = data_reader.model_latlon(model_config)
        self.latitude = model_lat
        self.longitude = model_lon

        lat_corner = np.concatenate((model_lat[:1],0.5*(model_lat[1:] + model_lat[:-1]),model_lat[-1:]))
        lon_corner = np.concatenate((model_lon - model_config.resolution/2, model_lon[-1:] + model_config.resolution/2))
        self.dset_out = xr.Dataset({'lat' : ('lat',model_lat),
                            'lon' : ('lon',model_lon),
                            'lat_b' : (lat_corner),
                            'lon_b' : (lon_corner)})
        self.dset_out['lat'].attrs['bounds'] = 'lat_b'
        self.dset_out['lon'].attrs['bounds'] = 'lon_b'

    def readdate(self,indate : [datetime.datetime, np.datetime64]):
        dtime_ns = np.datetime64(indate).astype('datetime64[ns]')
        fstd_datestamp = np.datetime64(indate).astype('datetime64[us]').astype(datetime.datetime).strftime('%Y%m%d%H')
        minus6_datestamp = (np.datetime64(indate).astype('datetime64[us]')-6*3600*1_000_000).astype(datetime.datetime).strftime('%Y%m%d%H')

        input_dataset = xr.Dataset(coords={'time':[dtime_ns],
                                        'level':list(self.task_config.pressure_levels),
                                        'latitude':self.latitude,
                                        'longitude':self.longitude}) 

        if (self.cache is not None and \
            (cache_ds := self.cache.readdate(dtime_ns)) is not None):
            input_dataset = data_reader.update_from_xarray(input_dataset,dtime_ns,self.task_config,
                                                           self.latitude,self.longitude,cache_ds,None)

        # Check to see whether we still need any input variables
        missing_variables = set.difference(set(self.task_config['input_variables']),set(input_dataset.data_vars))
        missing_variables = set.difference(missing_variables,set(DERIVED_VARIABLES))
        if (len(missing_variables) == 0):
            # We have all required variables, return
            return input_dataset

        # We do not have all required variables, continue

        # Re-label the dataset dimensions to lat/lon/pres convention
        input_dataset = input_dataset.rename(latitude='lat',longitude='lon',level='pres')        

        geophy_file = self.dbase_path.joinpath('geophy.fst')
        if (not geophy_file.exists()):
            geophy_file = None

        # Helper function to return the single (first) file following a patter if it exists,
        # or None if it doesn't
        def file_or_none(path : pathlib.Path,extension : str):
            files = list(path.glob(extension))
            if (len(files) == 0):
                return None
            else:
                return files[0]

        # Analysis at pressure levels
        analysis_pres_file = file_or_none(self.dbase_path,f'*anal*/*pres*/{fstd_datestamp}_000')
        # Analysis at hybrid levels (surface variables used only)
        analysis_hyb_file = file_or_none(self.dbase_path,f'*anal*/*hyb*/{fstd_datestamp}_000')
        # Forecast at pressure levels (WW)
        forecast_pres_file = file_or_none(self.dbase_path,f'*prog*/*pres*/{minus6_datestamp}_006')
        # Forecast at hybrid levels (PR)
        forecast_hyb_file = file_or_none(self.dbase_path,f'*prog*/*hyb*/{minus6_datestamp}_006')

        input_dataset = self.add_file_contribution(input_dataset,geophy_file,'geophy')
        input_dataset = self.add_file_contribution(input_dataset,analysis_pres_file,'analysis_pres')
        input_dataset = self.add_file_contribution(input_dataset,analysis_hyb_file,'analysis_hyb')
        input_dataset = self.add_file_contribution(input_dataset,forecast_pres_file,'forecast_pres')
        input_dataset = self.add_file_contribution(input_dataset,forecast_hyb_file,'forecast_hyb')
        input_dataset = self.ww_fixup(input_dataset,forecast_pres_file,forecast_hyb_file)

        # with fst files processed, rename variables to level/lat/lon
        input_dataset = input_dataset.rename(pres='level',lat='latitude',lon='longitude')

        input_variables = self.task_config['input_variables'] # things required as input to GraphCast
        present_input_variables = set(input_dataset.data_vars).union(set(DERIVED_VARIABLES)) # Values we have or can get
        missing_input_variables = list(set.difference(set(input_variables),present_input_variables))
        if (len(missing_input_variables) > 0):
            #print(f'Error: required input variables {missing_input_variables} are missing from the database')
            print(f'{analysis_pres_file=}')
            print(f'{analysis_hyb_file=}')
            print(f'{forecast_pres_file=}')
            print(f'{forecast_hyb_file=}')
            raise(ValueError(f'Error: required input variables {missing_input_variables} are missing from the database'))

        # Write the computed dataset back to the cache
        if (self.cache is not None):
            self.cache.update(input_dataset)

        return(input_dataset)
    
    def add_file_contribution(self,input_dataset,file_name,file_role):

        # Bail early if given a null file
        if (file_name) is None:
            return input_dataset
        task_config = self.task_config

        file_vars = set(v for v in fstdvars if v.source_file == file_role and v.name_xarray not in input_dataset)
        file_scalars = set(v for v in file_vars if v.vector_u == False and v.vector_v == False)
        file_vectors = set(v for v in file_vars if v.vector_u or v.vector_v)
        
        # Process scalars
        file_fstdvars = set(v.name_fstd for v in file_scalars)
        if (len(file_fstdvars) > 0):
            if (self.verbose): print(f'Loading {file_name}, scalar vars {sorted(list(file_fstdvars))}')
            yin_view = fstd2nc.Buffer(file_name,vars=list(file_fstdvars),bounds=True,yin=True).to_xarray()
            yang_view = fstd2nc.Buffer(file_name,vars=list(file_fstdvars),bounds=True,yang=True).to_xarray()
            
            (regridder_yin, regridder_yang, yyg_weights) = get_regridders(self.dset_out,yin_view,yang_view)
            
            for var in file_scalars:
                fname = var.name_fstd
                xname = var.name_xarray
                if (self.verbose): print(xname)
                if (var.surface): # Select only the surface level, if present
                    if 'level' in yin_view[fname].coords:
                        var_yin = yin_view[fname].sel(level=1.00,drop=True)
                        var_yang = yang_view[fname].sel(level=1.00,drop=True)
                    elif 'height' in yin_view[fname].coords:
                        var_yin = yin_view[fname].isel(height=0,drop=True)
                        var_yang = yang_view[fname].isel(height=0,drop=True)
                    else:
                        var_yin = yin_view[fname]
                        var_yang = yang_view[fname]
                else: # Select specified pressure levels
                    var_yin = yin_view[fname].isel(pres = yin_view[fname].pres.isin(tuple(task_config.pressure_levels)))
                    var_yang = yang_view[fname].isel(pres = yang_view[fname].pres.isin(tuple(task_config.pressure_levels)))
            
                input_dataset[xname] = (regridder_yin(var_yin).fillna(0) + regridder_yang(var_yang).fillna(0))/yyg_weights
                if (var.conversion is not None):
                    input_dataset[xname] = var.conversion[0]*input_dataset[xname] + var.conversion[1]

        # Process vectors
        file_fstdvars = set(v.name_fstd for v in file_vectors)
        if (len(file_fstdvars) > 0):
            if (self.verbose): print(f'Loading {file_name}, vector vars {sorted(list(file_fstdvars))}')
        
            # We only handle the case of a single [U,V] demand, since at present there's insufficient
            # information to match (u,v) components over multiple semantic variables.
            assert(len(file_fstdvars) == 2)
            
            yin_view = fstd2nc.Buffer(file_name,vars=list(file_fstdvars),bounds=True,yin=True).to_xarray()
            yang_view = fstd2nc.Buffer(file_name,vars=list(file_fstdvars),bounds=True,yang=True).to_xarray()
            
            (regridder_yin, regridder_yang, yyg_weights) = get_regridders(self.dset_out,yin_view,yang_view)
            
            uvar = [v for v in file_vectors if v.vector_u == True][0]
            vvar = [v for v in file_vectors if v.vector_v == True][0]
        
            if (self.verbose): 
                print(uvar.name_xarray)
                print(vvar.name_xarray)
            
            if (uvar.surface): # Surface vector field
                if 'level' in yin_view[uvar.name_fstd].coords:
                    uvar_yin = yin_view[uvar.name_fstd].sel(level=1.00,drop=True)
                    uvar_yang = yang_view[uvar.name_fstd].sel(level=1.00,drop=True)
                    vvar_yin = yin_view[vvar.name_fstd].sel(level=1.00,drop=True)
                    vvar_yang = yang_view[vvar.name_fstd].sel(level=1.00,drop=True)
                elif 'height' in yin_view[uvar.name_fstd].coords:
                    uvar_yin = yin_view[uvar.name_fstd].sel(height=10.0,drop=True)
                    uvar_yang = yang_view[uvar.name_fstd].sel(height=10.0,drop=True)
                    vvar_yin = yin_view[vvar.name_fstd].sel(height=10.0,drop=True)
                    vvar_yang = yang_view[vvar.name_fstd].sel(height=10.0,drop=True)
                else:
                    uvar_yin = yin_view[uvar.name_fstd]
                    uvar_yang = yang_view[uvar.name_fstd]
                    vvar_yin = yin_view[vvar.name_fstd]
                    vvar_yang = yang_view[vvar.name_fstd]
            else: # 3D vector field
                uvar_yin = yin_view[uvar.name_fstd].isel(pres = yin_view.pres.isin(tuple(task_config.pressure_levels)))
                uvar_yang = yang_view[uvar.name_fstd].isel(pres = yin_view.pres.isin(tuple(task_config.pressure_levels)))
                vvar_yin = yin_view[vvar.name_fstd].isel(pres = yin_view.pres.isin(tuple(task_config.pressure_levels)))
                vvar_yang = yang_view[vvar.name_fstd].isel(pres = yin_view.pres.isin(tuple(task_config.pressure_levels)))
        
            # Get projections for yin, yang, and the global latlon
            import cartopy
            yin_crs = fstd2nc.extra.get_crs(yin_view)
            yang_crs = fstd2nc.extra.get_crs(yang_view)
            latlon_crs = cartopy.crs.PlateCarree()
            
            # build derived CRSes on a sphere
            # GEMSphere = cartopy.crs.Globe(datum='WGS84',ellipse='sphere',
            #                               semimajor_axis=6371.22e3,
            #                               semiminor_axis=6371.22e3)
            # ydict = y1_crs.to_dict()
            # yin_crs = cartopy.crs.RotatedGeodetic(pole_longitude=ydict['lon_0']-180,
            #                                       pole_latitude=ydict['o_lat_p'],
            #                                       central_rotated_longitude=ydict['o_lon_p'],
            #                                       globe=GemSphere)
            # ydict = y2_crs.to_dict()
            # yang_crs = cartopy.crs.RotatedGeodetic(pole_longitude=ydict['lon_0']-180,
            #                                        pole_latitude=ydict['o_lat_p'],
            #                                        central_rotated_longitude=ydict['o_lon_p'],
            #                                        globe=GemSphere)
            # latlon_crs = cartopy.crs.Geodetic(globe=GemSphere)
        
            # If necessary, convert u/v from their own units (knots) to m/s
            if (uvar.conversion is not None):
                uvar_yin = uvar.conversion[0]*uvar_yin + uvar.conversion[1]
                uvar_yang = uvar.conversion[0]*uvar_yang + uvar.conversion[1]
            if (vvar.conversion is not None):
                vvar_yin = vvar.conversion[0]*vvar_yin + vvar.conversion[1]
                vvar_yang = vvar.conversion[0]*vvar_yang + vvar.conversion[1]
        
            if (uvar.surface):
                # Use the FD transformer directly
                (ugeo_yin, vgeo_yin) = transform_2d_winds(uvar_yin[0,:,:],vvar_yin[0,:,:],yin_crs,latlon_crs)
                (ugeo_yang, vgeo_yang) = transform_2d_winds(uvar_yang[0,:,:],vvar_yang[0,:,:],yang_crs,latlon_crs)
            else:
                # Save some time by building a transform basis and applying it to all levels
                (ugeo_yin, vgeo_yin) = transform_winds_by_basis(uvar_yin,vvar_yin,yin_crs,latlon_crs)
                (ugeo_yang, vgeo_yang) = transform_winds_by_basis(uvar_yang,vvar_yang,yang_crs,latlon_crs)

            # Interpolate zonal/meridional winds to target lat/lon grid, treating zonal/meridional winds as scalars,
            # and assign to dataset
            input_dataset[uvar.name_xarray] = (regridder_yin(ugeo_yin).fillna(0) + regridder_yang(ugeo_yang).fillna(0))/yyg_weights
            input_dataset[vvar.name_xarray] = (regridder_yin(vgeo_yin).fillna(0) + regridder_yang(vgeo_yang).fillna(0))/yyg_weights

            # Ensure that wind fields have 'time' dimension
            if ('time' not in input_dataset[uvar.name_xarray].coords):
                input_dataset[uvar.name_xarray] = input_dataset[uvar.name_xarray].expand_dims('time')
                input_dataset[vvar.name_xarray] = input_dataset[vvar.name_xarray].expand_dims('time')

        return input_dataset
    
    def ww_fixup(self,input_dataset,forecast_pres_file,forecast_hyb_file):
        '''Replaces missing vertical velocity values in a target dataset with values
        interpolated from hybrid levels'''
        from forecast.fstd_reader import get_regridders, vertical_bisection_search

        # Bail early if either forecast file is None (not present)
        if (forecast_pres_file is None or forecast_hyb_file is None):
            return input_dataset

        # Ensure vertical velocity is computed
        input_dataset['vertical_velocity'] = input_dataset['vertical_velocity'].compute()
        missing_level_mask = input_dataset['vertical_velocity'][0,:,0,0].isnull()

        # If all levels are present, return without further calculation
        if (not np.any(missing_level_mask.data)):
            return (input_dataset)
        
        missing_level_list = input_dataset['pres'][missing_level_mask.data]
        # print(missing_level_list)
        
        if (self.verbose):
            print(f'Fixing up {missing_level_list.size} missing WW levels')

        # Open the forecast files in both yin and yang view, looking at GZ and WW fields
        fc_hyb_gz_yin = fstd2nc.Buffer(forecast_hyb_file,vars=['GZ'],yin=True,bounds=True).to_xarray().compute()
        fc_hyb_ww_yin = fstd2nc.Buffer(forecast_hyb_file,vars=['WW'],yin=True).to_xarray().compute()
        fc_pres_gz_yin = fstd2nc.Buffer(forecast_pres_file,vars=['GZ'],yin=True).to_xarray().compute()
        fc_hyb_gz_yang = fstd2nc.Buffer(forecast_hyb_file,vars=['GZ'],yang=True,bounds=True).to_xarray().compute()
        fc_hyb_ww_yang = fstd2nc.Buffer(forecast_hyb_file,vars=['WW'],yang=True).to_xarray().compute()
        fc_pres_gz_yang = fstd2nc.Buffer(forecast_pres_file,vars=['GZ'],yang=True).to_xarray().compute()

        (regridder_yin, regridder_yang, yyg_weights) = get_regridders(self.dset_out,fc_hyb_gz_yin,fc_hyb_gz_yang)

        # Convert fields to the target lat-lon grid
        fc_hyb_gz_ll = ((regridder_yin(fc_hyb_gz_yin.GZ).fillna(0) + regridder_yang(fc_hyb_gz_yang.GZ).fillna(0))/yyg_weights)
        fc_hyb_ww_ll = ((regridder_yin(fc_hyb_ww_yin.WW).fillna(0) + regridder_yang(fc_hyb_ww_yang.WW).fillna(0))/yyg_weights)
        fc_pres_gz_ll = ((regridder_yin(fc_pres_gz_yin.GZ).fillna(0) + regridder_yang(fc_pres_gz_yang.GZ).fillna(0))/yyg_weights)

        # Variables later used for indexing
        lat_idx = np.arange(input_dataset['vertical_velocity'].data.shape[-2])[np.newaxis,:,np.newaxis]
        lon_idx = np.arange(input_dataset['vertical_velocity'].data.shape[-1])[np.newaxis,np.newaxis,:]

        # Get index of pressure coordinates corresponding to missing levels (they may not be exactly present in the file)
        # Interpolate with respect to log pressure, which makes the vertical coordinate more linear
        # print(np.log(missing_level_list))
        # print(np.log(fc_pres_gz_ll.pres.data))
        # print(np.arange(fc_pres_gz_ll.pres.size))
        pressure_idx = np.interp(np.log(missing_level_list.data),np.log(fc_pres_gz_ll.pres.data),np.arange(fc_pres_gz_ll.pres.size))

        # Get the geopotential heights of the missing levels (via linear interpolation of gz)
        idx_floor = np.floor(pressure_idx).astype(int)
        idx_ceil = np.ceil(pressure_idx).astype(int)
        idx_frac = pressure_idx - idx_floor
        
        gz_of_missing_levels = ((1-idx_frac[np.newaxis,:,np.newaxis,np.newaxis])*(fc_pres_gz_ll.data[:,idx_floor,:,:]) + \
                                idx_frac[np.newaxis,:,np.newaxis,np.newaxis]*(fc_pres_gz_ll.data[:,idx_ceil,:,:]))

        # print(gz_of_missing_levels[0,:,0,0])

        # # Get the geopotential heights of the missing WW values
        # gz_of_missing_levels = fc_pres_gz_ll.isel(pres = fc_pres_gz_ll.pres.isin(missing_level_list))

        # print('mask',missing_level_mask.shape)
        # print('gz_of',gz_of_missing_levels.shape)
        # print(gz_of_missing_levels.pres)
        # print(fc_pres_gz_ll.pres.data)

        # Search for the index and GZ corresponding to the missing pressure levels
        (target_idx, target_gz) = vertical_bisection_search(fc_hyb_gz_ll.data,gz_of_missing_levels)

        # print('target_idx',target_idx.shape)

        # Turn that index into a vertical coordinate
        tfloor = np.floor(target_idx).astype(int)
        tceil = np.ceil(target_idx).astype(int)
        tfrac = np.mod(target_idx,1)
        target_coord = (1-tfrac)*fc_hyb_gz_ll.level.data[tfloor] + tfrac*fc_hyb_gz_ll.level.data[tceil]

        # print('target_coord',target_coord.shape)

        # Turn the target coordinate back into an index, this time on the WW array
        idx_ww = np.interp(target_coord,fc_hyb_ww_ll.level.data,np.arange(fc_hyb_ww_ll.level.size))

        # print('idx_ww',idx_ww.shape)

        # Finally, linearly interpolate to give vertical velocity
        idx_ww_floor = np.floor(idx_ww).astype(int)
        idx_ww_ceil = np.ceil(idx_ww).astype(int)
        idx_ww_frac = idx_ww - idx_ww_floor
        
        ww_out = (1-idx_ww_frac)*fc_hyb_ww_ll.data[0,idx_ww_floor,lat_idx,lon_idx] + idx_ww_frac*fc_hyb_ww_ll.data[0,idx_ww_ceil,lat_idx,lon_idx]


        input_dataset['vertical_velocity'][:,missing_level_mask.data,:,:] = ww_out
        return input_dataset





