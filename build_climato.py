#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zarr
import dask
import dask.distributed
import dask.array
from forecast import encabulator
import numpy as np
import glob
import xarray as xr
import matplotlib.pyplot as plt

## The "encabulator" compressor is loaded and registered with numcodecs on import, but this means it has to be done
## in each Dask process.
def do_import():
    from forecast import encabulator



# After a lot of experimentation, it seems like Dasks's automatic array hnadling is not _quite_ up to the task of performant
# climatology generation.  It doesn't really understand how to interleave data loading, multiplication by smoothing weights
# (turning one sample into O(100) fractions), and performing the sum-reduce.  Its naive computational creates up to terrabytes
# of temporary data which has to spill to disk.

# Instead, we'll use the "Actor" mechanism to directly schedule tasks on the Dask workers.  We give up Dask's automatic scheduling,
# but in exchange we can explicitly control data locality and make sure that everything fits in core memory.

class AccumulatedActor():
    accum = None # Accessible as a data member, the fractional climatology accumulated here
    me = None # Accessible as a data member, string referencing this actor's worker
    def __init__(self,param=None):
        import numpy as np
        import dask
        import dask.distributed
        self.accum = None
        self.me = dask.distributed.get_worker().address

    def accumulate(self,dvar,weights):
        # Given a data variable (dask array) and set of time-weights, accumulate the influence of this data variable
        # (weights[frac]*var[z,y,x])
        import numpy as np
        import scipy.linalg
        
        # Happy case: all weights are 0, no work necessary
        if np.all(weights == 0): return
        
        import dask
        import dask.distributed

        if self.accum is None:
            # Accumulation array hasn't been set yet; create it based on the shape of dvar and
            # weights
            self.accum = np.zeros((weights.shape[0],) + dvar.shape,dtype=np.float32)
        
        me = dask.distributed.get_worker().address
        Client = dask.distributed.get_client()

        # Realize the expanded product, giving a numpy array.  Ensure that only this dask worker
        # participates in the computation, avoiding data movement
        # Use an explicitly-constructed tuple to address weights so that it can operate with dvar
        # of both 2 and 3 dimensions
        dvar = Client.compute(dvar,sync=True,workers=me,allow_other_workers=False)
        # realized_prod = Client.compute(weights[(slice(None,None,None),) + (None,)*dvar.ndim]*dvar[None,...],
        #                               sync=True, 
        #                               workers=me,
        #                               allow_other_workers=False)
        for ii in range(weights.size):
            scipy.linalg.blas.saxpy(dvar.ravel(),self.accum[ii,:].ravel(),a=weights[ii])

        # Add the realized product to the accumulator
        # self.accum += realized_prod
        
        return    

    def accumulate_many(self,dvars,weights):
        # Accumulate many vars/weights
        for (v,w) in zip(dvars,weights):
            self.accumulate(v,w)
        return

    def retrieve(self):
        # Retrieve the accumulator as a Dask array, chunked along the year-fraction dimension
        import dask.array
        import dask.distributed
        Client = dask.distributed.get_client()
        # with dask.distributed.worker_client() as Client:
        me = dask.distributed.get_worker().address
        myarray = dask.array.from_array(self.accum,chunks=(-1,) + (-1,)*(self.accum.ndim-1))
        return myarray

    def accumulate_from(self,other):
        # Given another Actor, take its accumulator and add it to our own.  By using the member
        # reference here, we avoid transferring data through the main/scheduler process
        oa = other.accum
        if (self.accum is None):
            self.accum = oa
        elif (oa is None):
            # No work
            pass
        else:
            self.accum += other.accum
        return

    def reset_accumulator(self):
        # Reset the accumulator variable
        self.accum = None
        return
        
       
if __name__ == '__main__':
    dask.config.set({'distributed.scheduled.active-memory-manager.measure' : 'managed',
                    'distributed.worker.memory.rebalance.measure' : 'managed',
                    'distributed.worker.memory.spill' : False,
                    'distributed.worker.memory.pause' : False,
                    'distributed.worker.memory.terminate' : False,
                    'temporary_directory': '/fs/site6/eccc/mrd/rpnatm/csu001/ppp6/tmp',
                    'array.slicing.split_large_chunks': False})



    Client = dask.distributed.Client(processes=True,n_workers=64)
    Client.run(do_import);

    # In[6]:


    # Create one actor per Dask process.
    # What's the collective noun for actors? A cast.

    remote_cast = []
    for worker in Client.processing().keys():
        remote_cast.append(Client.submit(AccumulatedActor,actors=True,workers=worker,allow_other_workers=False).result())


    # In[7]:


    dbase_path='/fs/site6/eccc/mrd/rpnatm/csu001/ppp6/era5_025deg'
    start_year = 1980
    end_year = 2010


    # In[8]:


    dbase_month_paths = sorted(list(glob.glob(f'{dbase_path}/*/*')))


    # In[9]:


    def check_year(path):
        int_year = int(path.split('/')[-2])
        return (int_year >= start_year and int_year <= end_year)
    dbase_month_paths_filtered = [d for d in dbase_month_paths if check_year(d)]


    # In[10]:


    dbase_years = [int(p.split('/')[-2]) for p in dbase_month_paths_filtered]
    dbase_year_parity = [ (1 if (y % 4 == 0) else 0) for y in dbase_years]
    dbase_months = [int(p.split('/')[-1]) for p in dbase_month_paths_filtered]


    # In[11]:


    dbase_month_paths_filtered[:10]


    # In[12]:


    dbase_by_month = [xr.open_zarr(d,chunks={}) for d in dbase_month_paths_filtered]


    # In[13]:


    # Build the time-weights for the climatology, a triangle function with width of (window_size) (61) days.
    # Since the climatology is based in year-fraction coordinates rather than a straight day coordinate,
    # scale things appropriately.
    window_size = 61
    NYear_Frac = 365
    out_year_fraction = (np.arange(NYear_Frac) / NYear_Frac).astype(np.float32)
    def build_time_weight(dbase):
        times = dbase.time.data
        years = times.astype('datetime64[Y]') 
        hours = times.astype('datetime64[h]')
        days = times.astype('datetime64[D]')
        years_int = ((years - np.datetime64('1900-01-01','Y')) / np.timedelta64(1,'Y')).astype(np.int64)
        numdays_int = 365 + ((years_int % 4) == 0)
        yearfrac = (times - years)/np.timedelta64(1,'D')/(numdays_int)
        hour_of_day = ((hours - days)/np.timedelta64(1,'h')).astype(np.int64)

        # NYear_Frac = 100
        # out_year_fraction = (np.arange(NYear_Frac) / NYear_Frac).astype(np.float32)
        window_size_fraction = window_size / 365 / 2 # Divide by 2 to make symmetric

        year_lag = out_year_fraction[:,None] - yearfrac[None,:] # Signed difference, no year wraparound
        year_distance = (np.abs((year_lag + 0.5) % 1 - 0.5)).astype(np.float32) # Shift to [-0.5,0.5] scale with wraparound and take absolute value
        year_window = np.maximum(0,1 - year_distance/window_size_fraction) # Apply triangular window function
        year_window = year_window[:,:] / np.sum(year_window,axis=0)[None,:] # Normalize to have sum of 1
        year_window_by_hour = year_window.reshape((NYear_Frac,-1,4))

        norm_weight = np.sum(year_window_by_hour,axis=0)

        return year_window_by_hour / norm_weight[None,:,:]


    # In[14]:


    # Build time-weights for each sample of each month-dataset
    month_weights = [build_time_weight(d) for d in dbase_by_month]


    # In[15]:


    # To have a globally normalized set of weights, first sum all day-weights, keeping the hour-of-day dimension
    # separate
    total_weights = sum(m.sum(axis=1) for m in month_weights)


    # In[16]:


    # And each set of weights by its corresponding total, so that the sum over all data points is 1.  This means
    # that the aggregation will also directly compute the mean.
    month_weights_normed = [m / total_weights[:,None,:] for m in month_weights]


    # In[17]:


    month_weights_normed[1].shape


    # In[18]:


    # Flatten the weights to prepare for aggregation
    weights_flat = np.concatenate(month_weights_normed,axis=1).reshape((NYear_Frac,-1))


    # In[19]:


    def get_var_weights(var_name,hour_choice,OUTPUT_START,OUTPUT_END):
        # Compute (sample, weight) pairings that can contribute to the output, ending the per-month
        # distinction.
        # Special case for wind speed
        if (var_name == 'wind_speed'):
            var_month = [((dbase['u_component_of_wind']**2 + dbase['v_component_of_wind']**2)**0.5).data for dbase in dbase_by_month]
        elif (var_name == '10m_wind_speed'):
            var_month = [((dbase['10m_u_component_of_wind']**2 + dbase['10m_v_component_of_wind']**2)**0.5).data for dbase in dbase_by_month]
        else:
            # Default case, load the data
            var_month = [dbase[var_name].data for dbase in dbase_by_month] # Select the target variable out of each dataset
        live_product = []
        for (idx,(v,w)) in enumerate(zip(var_month,month_weights_normed)):
            month_weights = w[OUTPUT_START:OUTPUT_END,:,:] # Select the weights corresponding to the output
            samples = [] 
            for day in range(month_weights.shape[1]):
                if not np.all(month_weights[:,day,hour_choice] == 0): # Skip any sample that does not contribute to the output
                    samples.append( (v[hour_choice + 4*(day-1),...] , month_weights[:,day,hour_choice]) )
            live_product.extend(samples)
        return live_product
            


    # In[20]:


    def dispatch(remote_cast,live_product):
        # Separate the pairs into distinct lists for easier iteration
        live_vars, live_weights = zip(*live_product)
        # Dispatch the (variable,weights) pairs to each remote Actor for accumulation
        app_fut = []
        stride = len(remote_cast)
        for (idx,actor) in enumerate(remote_cast):
            app_fut.append(actor.accumulate_many(live_vars[idx::stride],live_weights[idx::stride]))
        return app_fut


    # In[21]:


    def recursive_sum(remote_cast):
        # Sum the Actor's accumulators together using a tree structure, dispatching sum operations
        to_sum = remote_cast.copy()
        resets = []
        while len(to_sum) > 1:
            dispatched = []
            # Retrieve Actors in pairs
            while len(to_sum) > 1:
                a1 = to_sum.pop()
                a2 = to_sum.pop()
                fut = a1.accumulate_from(a2)
                # print(f'Summing {a1.me} and {a2.me}')
                dispatched.append((a1,a2,fut))
            # Wait for sums to complete
            for (actor_rec,actor_send,future) in dispatched:
                res = future.result() # Blocks
                # print(f'Sum on {actor_rec.me} complete')
                to_sum.append(actor_rec) # Return the live actor to the summing pool
                resets.append(actor_send.reset_accumulator())
        # Reset the sum
        for fut in resets: fut.result()
        # Download results from the last actor standing
        climato_var = to_sum[0].accum
        # Reset the final actor
        to_sum[0].reset_accumulator().result()
        return climato_var


    # In[22]:


    def output_var(filename,climato_var,var_name,coords,hour_choice,OUTPUT_START,OUTPUT_END):
        if (climato_var.ndim == 4): # 3D variable
            out_xr = xr.DataArray(data=climato_var[:,None,:,:,:],dims=('year_fraction','hour','level','latitude','longitude'))
        else: # 2D variable
            out_xr = xr.DataArray(data=climato_var[:,None,:,:],dims=('year_fraction','hour','latitude','longitude'))
        # Chunk by year fraction
        out_xr = out_xr.chunk(year_fraction=1)

        # Construct a dataset for output, giving dimensions semantic meaning
        out_ds = xr.Dataset(coords={'year_fraction' : out_year_fraction[OUTPUT_START:OUTPUT_END],
                                            'hour' : [6*hour_choice,], 
                                            'level' : coords['level'],
                                            'latitude' : coords['latitude'],
                                            'longitude' : coords['longitude']})
        # Assign the variable to the dataset
        out_ds[var_name] = out_xr

        # Define the compression
        encoding = {var_name : {'compressor' :  encabulator.LayerQuantizer(nbits=16)}}
        out_comp = out_ds.to_zarr(filename,encoding = encoding,compute=True)
        return


    # In[23]:


    eligible_vars = [d for d in dbase_by_month[0].data_vars if 'time' in dbase_by_month[0][d].dims] + ['wind_speed','10m_wind_speed']

    vars_3d = [v for v in dbase_by_month[0].data_vars if 'level' in dbase_by_month[0][v].dims] + ['wind_speed']
    vars_2d = [v for v in dbase_by_month[0].data_vars if 'level' not in dbase_by_month[0][v].dims] + ['10m_wind_speed']


    divisions = np.floor(np.linspace(0,NYear_Frac,20)).astype(np.int32)


    # In[ ]:


    import datetime
    import os
    for var_name in eligible_vars:
        for hour_choice in [0,1,2,3]:
            if (var_name in vars_3d): # 3D var, use output divisions already set
                my_divisions = divisions
            elif (var_name in vars_2d): # for 2D var, compute the whole field at once
                my_divisions = np.array([0,divisions[-1]],dtype=np.int32)
            else:
                raise ValueError(f'Variable {var_name} is neither 2D nor 3D?')
            for idx in range(my_divisions.size-1):
                OUTPUT_START=my_divisions[idx]
                OUTPUT_END=my_divisions[idx+1]
                filename = f'../climato_37/{var_name}.{OUTPUT_START:03d}.{hour_choice*6}.zarr'
                if (os.path.exists(filename)): # Output file exists, it may have been generated already
                    print(f'{filename} exists already, trying to open')
                    goodfile = True
                    try:
                        trial_db = xr.open_zarr(filename) # Try to open it
                        print(f'... open successful')
                    except BaseException:
                        goodfile = False
                    if (goodfile and len(trial_db.data_vars) == 1):
                        print(f'... and it contains data.  Continuing')
                        continue
                    else:
                        printf('... but there are problems.  Deleting and regenerating')
                        import shutil
                        shutil.rmtree(filename)

                print(f'Processing {var_name} @ {hour_choice*6}h, output {OUTPUT_START} to {OUTPUT_END}')
                tic = datetime.datetime.now()
                live_product = get_var_weights(var_name,hour_choice,OUTPUT_START,OUTPUT_END)
                app_fut = dispatch(remote_cast,live_product)
                # Wait for results
                for f in app_fut:
                    f.result()
                climato_var = recursive_sum(remote_cast)
                output_var(filename,climato_var,var_name,dbase_by_month[0].coords,hour_choice,OUTPUT_START,OUTPUT_END)
                toc = datetime.datetime.now()
                print(f'   done in {(toc-tic).total_seconds()/60:.2f}m')

    # In[ ]:




