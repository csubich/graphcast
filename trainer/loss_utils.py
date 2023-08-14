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


def normalize(ds,base,mean_by_level,stddev_by_level,diffs_stddev_by_level):
    #delta = ds - base
    import xarray as xr
    delta = xr.Dataset()
    for var in ds.data_vars:
        if (var in base.data_vars):
            delta[var] = (ds[var] - base[var])/diffs_stddev_by_level[var]
        else:
            delta[var] = (ds[var] - mean_by_level[var])/stddev_by_level[var]
    return delta
                
def losses_over_time(forecast,targets,analysis,per_variable_weights,norm_fn):
    from graphcast import xarray_jax
    from graphcast import losses
    norm_fc = norm_fn(forecast,analysis)
    norm_tg = norm_fn(targets,analysis)
    persist = 0*norm_fc.isel(time=0)
    forecast_losses = [xarray_jax.unwrap_data(losses.weighted_mse_per_level(norm_fc.isel(time=lead),
                                                                            norm_tg.isel(time=lead),
                                                                            per_variable_weights)[0]) for lead in range(forecast.time.size)]
    persist_losses = [xarray_jax.unwrap_data(losses.weighted_mse_per_level(persist,
                                                                           norm_tg.isel(time=lead),
                                                                           per_variable_weights)[0]) for lead in range(forecast.time.size)]
    return(forecast_losses,persist_losses)

def make_loss(norm_by_level,per_variable_weights,level_weights,latitude_weights):
    '''Define and return a custom loss function which uses user-specified standard deviations,
    per-variable weights, level weights, and latitude weights.  Using this inside the gradient
    calculation adds a small penalty because the losses and gradients are calculated in float32
    precision rather than bfloat16.
    
    The paramter norm_by_level gives the per-level, per-variable normalization weight to apply
    for the loss.  To match the GC calculation, this dataset should match diffs_stddev_by_level
    for all predicted variables that are also input variables and stddev_by_level for predicted
    variables that are not input variables (precipitation in GC-13).
    '''
    def my_loss(prediction,targets):
        # print('Compiling loss/gradient (inside my_loss)')
        vars = list(targets.data_vars)
        prediction = prediction[vars]
        diffs = targets - prediction
        diffs = diffs/norm_by_level
        mse = (diffs**2 * latitude_weights * level_weights).sum(dim=('level')).mean(dim=('lat','lon','time','batch'))
        total = sum((mse[i]*per_variable_weights.get(i,1.0) for i in mse.data_vars))
        return(total,mse)

    return my_loss
