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


def load_model(params_file):
    from graphcast import checkpoint
    from graphcast import graphcast
    with open(params_file,"rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    return(ckpt.model_config, ckpt.task_config, ckpt.params)

def build_loss_and_grad(model_config, task_config, use_float16=True, custom_loss_fn = None, 
                        diffs_stddev_by_level = None, mean_by_level = None, stddev_by_level = None):
    '''Construct a wrapped GraphCast function to compute RMSE loss and
    gradients, as per the demonstration notebook'''
    import jax
    import haiku as hk
    from graphcast import graphcast, casting, normalization, autoregressive, rollout, xarray_jax, xarray_tree
    import xarray as xr
    import functools
    
    # Load normalization factors if not provided
    if (diffs_stddev_by_level is None):
        diffs_stddev_by_level = xr.load_dataset("stats/diffs_stddev_by_level.nc").compute()
    if (mean_by_level is None):
        mean_by_level = xr.load_dataset("stats/mean_by_level.nc").compute()
    if (stddev_by_level is None):
        stddev_by_level = xr.load_dataset("stats/stddev_by_level.nc").compute()
    
    def construct_wrapped_graphcast(model_config,task_config):
        predictor = graphcast.GraphCast(model_config,task_config)
    
        # If running on a GPU, operate in BFloat16 mode
        if (use_float16):
            predictor = casting.Bfloat16Cast(predictor)
        
        # Apply normalization
        predictor = normalization.InputsAndResiduals(predictor,
                                                     diffs_stddev_by_level=diffs_stddev_by_level,
                                                     mean_by_level=mean_by_level,
                                                     stddev_by_level=stddev_by_level)
        # And wrap in the autoregressive magic to take multi-step predictions.
        predictor = autoregressive.Predictor(predictor,gradient_checkpointing=True)
        return predictor
    
    @hk.transform_with_state
    def loss_fn(model_config, task_config, inputs, targets, forcings):
        predictor = construct_wrapped_graphcast(model_config, task_config)
        if (custom_loss_fn is not None):
            loss, diagnostics = custom_loss_fn(predictor(inputs,targets,forcings),targets)
        else:
            loss, diagnostics = predictor.loss(inputs, targets, forcings)
        return xarray_tree.map_structure(
            lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
            (loss, diagnostics))
    
    def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
        def _aux(params, state, i, t, f):
            (loss, diagnostics), next_state = loss_fn.apply(
                params, state, jax.random.PRNGKey(0), model_config, task_config,
                i, t, f)
            return loss, (diagnostics, next_state)
        (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
            _aux, has_aux=True)(params, state, inputs, targets, forcings)
        return loss, diagnostics, next_state, grads    
    
    def with_configs(fn):
        return functools.partial(
            fn, model_config=model_config, task_config=task_config)
    
    # def with_params(fn):
    #     return functools.partial(fn, params=params, state=state)  
    
    # def drop_state(fn)
    #     return lambda **kw: fn(**kw)[0]
    
    jit_loss = jax.jit(with_configs(loss_fn.apply))
    jit_grad = jax.jit(with_configs(grads_fn))
    
    def loss_wrapper(params,inputs,targets,forcings):
        ((loss, diagnostics),_) = jit_loss(params=params,
                                    inputs=inputs,
                                    targets=targets,
                                    forcings=forcings,
                                    rng=jax.random.PRNGKey(0),
                                    state={})
        return(loss,diagnostics)
    
    def grad_wrapper(params,inputs,targets,forcings):
        (loss, diagnostics, _, grad) = jit_grad(params=params,
                            inputs=inputs,
                            targets=targets,
                            forcings=forcings,
                            state={})
        return(loss,diagnostics,grad)

    return(loss_wrapper, grad_wrapper)



def build_predictor_params(model_config, task_config, use_float16=True,
                           diffs_stddev_by_level = None, mean_by_level = None, stddev_by_level = None):
    '''Construct a GraphCast predictor for making forecasts.

    This function is heavily based on the GraphCast demonstration code.'''

    import os
    # JAX's behaviour is controlled by environment variables
    import jax
    import haiku as hk
    from graphcast import graphcast, casting, normalization, autoregressive, rollout
    import xarray as xr
    import functools

    # Load normalization factors if not provided
    if (diffs_stddev_by_level is None):
        diffs_stddev_by_level = xr.load_dataset("stats/diffs_stddev_by_level.nc").compute()
    if (mean_by_level is None):
        mean_by_level = xr.load_dataset("stats/mean_by_level.nc").compute()
    if (stddev_by_level is None):
        stddev_by_level = xr.load_dataset("stats/stddev_by_level.nc").compute()

    def construct_wrapped_graphcast(model_config,task_config):
        predictor = graphcast.GraphCast(model_config,task_config)

        # If running on a GPU, operate in BFloat16 mode
        if (use_float16):
            predictor = casting.Bfloat16Cast(predictor)
        
        # Apply normalization
        predictor = normalization.InputsAndResiduals(predictor,
                                                     diffs_stddev_by_level=diffs_stddev_by_level,
                                                     mean_by_level=mean_by_level,
                                                     stddev_by_level=stddev_by_level)
        # And wrap in the autoregressive magic to take multi-step predictions.
        predictor = autoregressive.Predictor(predictor,gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config,task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)
    
    state = {}
    # def with_params(fn):
    #     return functools.partial(fn,params=params,state=state)

    # def drop_state(fn):
    #     return lambda **kw: fn(**kw)[0]
    
    jit_apply = jax.jit(with_configs(run_forward.apply))
    
    run_forward_jit = jit_apply

    def out_predictor(inputs, targets, forcings, params):
        # Per comments in graphcast/xarray_jax, the JAX wrapper for xarray
        # treats coordinates as static variables; they must be *identical*
        # between different invocations of a JITted function to re-use
        # the precompiled version.  The 'datetime' coordinate will not remain
        # the same between runs, so drop it before the prediction and add
        # it back afterwards
        if ('datetime' in forcings.coords):
            inputs_dv = inputs.drop_vars('datetime')
            targets_dv = targets.drop_vars('datetime')
            forcings_dv = forcings.drop_vars('datetime')
        else:
            inputs_dv = inputs
            targets_dv = targets
            forcings_dv = forcings
        predictions = run_forward_jit(
            rng = jax.random.PRNGKey(0),
            inputs = inputs_dv,
            targets_template = targets_dv,
            forcings = forcings_dv,
            params = params,
            state={}
        )[0]
        # Use the 'chunked prediction' method to minimize GPU memory.  Otherwise,
        # a full 10-day forecast exhausts the memory on an A100
        # predictions = rollout.chunked_prediction(
        #     run_forward_jit,
        #     rng=jax.random.PRNGKey(0),
        #     inputs=inputs_dv,
        #     targets_template=targets_dv,
        #     forcings=forcings_dv)
        if ('datetime' in forcings.coords):
            predictions.coords['datetime'] = forcings.coords['datetime']
        return predictions
    return out_predictor

def build_predictor(model_config, task_config, params, use_gpu=True, use_float16=True,
                    diffs_stddev_by_level = None, mean_by_level = None, stddev_by_level = None):   
    import os    
    if (not use_gpu):
        # Hide any CUDA devices, forcing CPU-only mode
        os.environ['CUDA_VISIBLE_DEVICES']=''
    else:
        # Keep CUDA devices if present, but force the use of the platform
        # memory allocator.  This is slightly slower than using preallocated
        # memory, but it plays nicely with GPU-sharing
        # os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='PLATFORM'
        # os.environ['TF_FORCE_UNIFIED_MEMORY']='1'
        # os.environ['XLA_CLIENT_MEM_FRACTION'] = '3.0'
        pass
    import jax
    import haiku as hk
    from graphcast import graphcast, casting, normalization, autoregressive, rollout
    import xarray as xr
    import functools

    # Load normalization factors if not provided
    if (diffs_stddev_by_level is None):
        diffs_stddev_by_level = xr.load_dataset("stats/diffs_stddev_by_level.nc").compute()
    if (mean_by_level is None):
        mean_by_level = xr.load_dataset("stats/mean_by_level.nc").compute()
    if (stddev_by_level is None):
        stddev_by_level = xr.load_dataset("stats/stddev_by_level.nc").compute()

    def construct_wrapped_graphcast(model_config,task_config):
        predictor = graphcast.GraphCast(model_config,task_config)

        # If running on a GPU, operate in BFloat16 mode
        if (use_float16 and use_gpu):
            predictor = casting.Bfloat16Cast(predictor)
        
        # Apply normalization
        predictor = normalization.InputsAndResiduals(predictor,
                                                     diffs_stddev_by_level=diffs_stddev_by_level,
                                                     mean_by_level=mean_by_level,
                                                     stddev_by_level=stddev_by_level)
        # And wrap in the autoregressive magic to take multi-step predictions.
        predictor = autoregressive.Predictor(predictor,gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config,task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)
    
    state = {}
    def with_params(fn):
        return functools.partial(fn,params=params,state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]
    
    jit_apply = jax.jit(drop_state(with_params(with_configs(run_forward.apply))))
    
    run_forward_jit = jit_apply

    def out_predictor(inputs, targets, forcings):
        # Per comments in graphcast/xarray_jax, the JAX wrapper for xarray
        # treats coordinates as static variables; they must be *identical*
        # between different invocations of a JITted function to re-use
        # the precompiled version.  The 'datetime' coordinate will not remain
        # the same between runs, so drop it before the prediction and add
        # it back afterwards
        if ('datetime' in forcings.coords):
            inputs_dv = inputs.drop_vars('datetime')
            targets_dv = targets.drop_vars('datetime')
            forcings_dv = forcings.drop_vars('datetime')
        else:
            inputs_dv = inputs
            targets_dv = targets
            forcings_dv = forcings
        # predictions = run_forward_jit(
        #     rng = jax.random.PRNGKey(0),
        #     inputs = inputs_dv,
        #     targets_template = targets_dv,
        #     forcings = forcings_dv,
        #     params = params,
        #     state={}
        # )[0]
        # Use the 'chunked prediction' method to minimize GPU memory.  Otherwise,
        # a full 10-day forecast exhausts the memory on an A100
        predictions = rollout.chunked_prediction(
            run_forward_jit,
            rng=jax.random.PRNGKey(0),
            inputs=inputs_dv,
            targets_template=targets_dv,
            forcings=forcings_dv)
        if ('datetime' in forcings.coords):
            predictions.coords['datetime'] = forcings.coords['datetime']
        return predictions
    return out_predictor

def init_params(model_config,task_config,inputs,targets,forcings,seed=0,
                diffs_stddev_by_level = None, mean_by_level = None, stddev_by_level = None):
    # Return randomly-initialized parameters for a Graphcast model specified by
    # given model_config and task_config dictionaries, applied to representative
    # inputs / targets / forcings datasets; intended for initialization of from-scratch
    # training.  Based on Deepmind's Graphcast demo workbook

    import jax
    import haiku as hk
    import xarray as xr

    # Load normalization factors if not provided
    if (diffs_stddev_by_level is None):
        diffs_stddev_by_level = xr.load_dataset("stats/diffs_stddev_by_level.nc").compute()
    if (mean_by_level is None):
        mean_by_level = xr.load_dataset("stats/mean_by_level.nc").compute()
    if (stddev_by_level is None):
        stddev_by_level = xr.load_dataset("stats/stddev_by_level.nc").compute()

    def construct_wrapped_graphcast(model_config,task_config):
        from graphcast import graphcast, casting, normalization, autoregressive
        predictor = graphcast.GraphCast(model_config,task_config)

        # If running on a GPU, operate in BFloat16 mode
        # if (use_float16 and use_gpu):
        #     predictor = casting.Bfloat16Cast(predictor)
        
        # Apply normalization
        predictor = normalization.InputsAndResiduals(predictor,
                                                    diffs_stddev_by_level=diffs_stddev_by_level,
                                                    mean_by_level=mean_by_level,
                                                    stddev_by_level=stddev_by_level)
        # And wrap in the autoregressive magic to take multi-step predictions.
        predictor = autoregressive.Predictor(predictor,gradient_checkpointing=True)
        return predictor

    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(model_config,task_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    params_random, _ = run_forward.init(jax.random.PRNGKey(seed),model_config,task_config,inputs,targets,forcings)

    return params_random
