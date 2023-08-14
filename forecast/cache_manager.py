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


# Implements a 'cache manager', providing a read and write wrapper for a backing ZArr
# (xarray format).  This manager includes support for a MutableMapping 'store' factory
# (default DirectoryStore), allowing transparent support for database or network-backed
# ZArr arrays.

# This CacheManager does not implement multithread/multiprocess locking, so it should be
# used only in serial.

import xarray as xr
import zarr as zr
import collections.abc
import warnings

class CacheManager():
    def __init__(self,cachename: str, storeFactory : collections.abc.MutableMapping = zr.DirectoryStore, encoders = {}, **kwargs):
        '''Initializes a new CacheManager responsible for <cachename>.  Any keyword arguments are pased to
        the storeFactory parameter (default zarray.DirectoryStore).  Optional parameter 'encoders' specifies
        encoding parameters to use for writes to the cache'''

        self.cachename = cachename
        self.store = storeFactory(cachename,**kwargs)
        self.encoders = encoders

        pass

    def readdate(self,indate):
        # Reads the cache for time <indate> plus any time-independent fields
        try:
            ds = xr.open_zarr(self.store)
        except (FileNotFoundError):
            # File doesn't exist, so return None
            return None

        if ('time' not in ds.dims or not (ds.time == indate).any()):
            # Time was not found, so drop the dimension. This keeps all time-invariant variables
            # and drops all time-varying ones.
            return(ds.drop_dims('time'))
        else:
            return ds.sel(time=slice(indate,indate))
        
    def update(self,in_ds):
        '''Updates the cache by adding <in_ds>, extending along the time dimension'''
        # Zeroth, chunk the data to store whole 3D arrays as single chunks
        in_ds = in_ds.chunk(chunks = {'time' : 1, 'latitude' : -1, 'longitude' : -1, 'level' : -1})

        # First, try to open the data store to see if it is a valid file
        try:
            cache_ds = xr.open_zarr(self.store)
        except FileNotFoundError:
            # It isn't.  The new dataset can be written in its entirety
            # print('writing',in_ds)
            in_ds.to_zarr(self.store,mode='w',encoding=self.encoders)
            return
        
        # It is a valid zarr.  Does it have a time dimension?
        if ('time' in cache_ds.coords):
            # Is it nontrivial?
            if (cache_ds.time.size > 0):
                if ((cache_ds.time == in_ds.time[0]).any()):
                    # Time is nontrivial and the current value already exists in the database.
                    # Modifying the cache implies reading and writing the entire cache, a
                    # presumably expensive operation.  Instead, return without modifying.
                    warnings.warn(f'Time {in_ds.time[0]} already exists in cache f{self.cachename}.  Skipping modification')
                    return # Exit the function early
                else:
                    # The current time doesn't exist in the cache, so write all time-related
                    # variables and append along the time dimension.
                    time_vars = [v for v in in_ds.data_vars if 'time' in in_ds[v].dims]
                    # print('appending along time',in_ds[time_vars])
                    # Do not use encoding here; dimensions being appended will pick up on the
                    # existing encoding of the file
                    in_ds[time_vars].to_zarr(self.store,mode='a',append_dim='time')
            else:
                # Time is trivial (size 0), so write in append mode without an append_dim.  This
                # will also write any time-independent variables.
                # print('appending all',in_ds)
                in_ds.to_zarr(self.store,mode='a',encoding=self.encoders)
        else:
            # Time is not in the database, use append mode to overwrite variables.
            # print('appending all',in_ds)
            in_ds.to_zarr(self.store,mode='a',encoding=self.encoders)
