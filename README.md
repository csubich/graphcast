# graphcast_train: training infrastructure for GraphCast

This repository implements the training code used for [(Subich 2024)](https://arxiv.org/abs/2408.14587), which fine-tuned GraphCast on the analyses of the Canadian Global Determinstic Prediction system at ¼°, over the same 37 pressure levels as the "headline" GraphCast model.

[Lam 2022](https://arxiv.org/abs/2212.12794) did not provide the training infrastructure needed to effectively train GraphCast, nor is the code otherwise included in the forked-from repository, and this branch fills that gap.

## Changes to GraphCast

The changes to GraphCast code (under the [graphcast/](graphcast/) directory) are minimal, consisting of:

* Implementation of additional gradient checkpoint inside the graph neural network, rather than only after each complete step (i.e. in autoregressive mode only).  Adding gradient checkpointing in this way allows computation of gradients for a single forecast step within the 40GiB on-card memory limit of a baseline NVIDIA A100.  

  When additionally using "unified memory" on a host with enough system RAM, the model will also compute gradients for longer forecasts.  Training up to 2.5d (10 steps) was possible at reasonable performance with 100GiB of node memory, but see (Subich 2024) for split-horizon trianing behond that limit.

* Replacement of `xarray.Dataset.dims[dim]` with `xarray.Dataset.size[dim]`, to avoid a deprecation warning with more recent versions of XArray.

## Bulk forecaster

As a first step towards training, a bulk (command-line) forecaster was implemented.  See [gforecast.py](gforecast.py) and [gforecast_demo.ipynb](gforecast_demo.ipynb) to the command-line invocation and an example of how the internal library structure works.  This is not the primary goal of this fork, but it might prove useful to others.  Files created for this forecaster that remain relevant for training are:

* [forecast/generate_model.py](forecast/generate_model.py) contains model-generating functions to create a predictor, loss/gradient computation, and initialization functions.  Modelled after (and directly borrowing from) the demonstration GraphCast notebook, these functions add model parameters as an explicit parameter (rather than closed-over variable).  This is necessary to allow parameters to vary over a training cycle (after optimizer updates), since closed-over variables are implicitly static and require recompilation.

* [forecast/toa_radiation.py](forecast/toa_radiation.py) computes top-of-atmosphere radiation appropriate for a supplied input/forcing dataset with valid time, latitude, and longitude coordinates.  The computation is based on the references noted in the file's comments, and it is necessary when radiation is not included in the input dataset (or when forecasting into the future).  The computation does not take the solar cycle or secular trends into account.

  The modern GraphCast repository now contains its own top-of-atmosphere radiation computation.  That is probably essentially equivalent to this in practice, but I have not tested it.

* [forecast/encabulator.py](forecast/encabulator.py) implements a data compressor compatible with Zarr/numcodecs that quantizes atmospheric variables by layer (typically 16 bits of precision between the per-time-and-level minimum and maximum values), then encodes the 2D differences (Lorenzo encoding).  This is based on a compression technique used on ECCC-internal files, and it reduces the on-disk usage of ERA5-like data to about 25% of its float32 equivalent (half the gain from quantization, the other half from encoding.)  Further development of this (for more flexible array shapes) and a technical report will hopefully be coming in the medium-term future, but the encoder is usable as-is for GraphCast-like needs.

## Training infrastructure

The training loop is primarily implemented in [train.py](train.py), and it is controlled by various command-line parameters.

### Theory of operation

The training loop separates data-loading from gradient computation, with dask.distributed (threading) used to load and preprocess data in parallel with gradient computation.  With Dask, threading appears to be a necessary evil.  A process-based distributed Client seems to serialize data to disk when handing it between processes, and that disk-based I/O is exactly the thing we want to have parallelized, in the background.

The loop supports multi-GPU training, using one computational thread per GPU and explicit accumulation of gradients.  When training over a few GPUs (up to 4 A100s in testing), this showed better performance than Jax-builtin data parallelsim via sharding over the 'batch' dimension.  My speculation is that the performance gains came from being able to dispatch training samples to the GPUs as they were ready/as individual GPUs become free, minimizing synchronization delays.

Each GPU thread (via a ThreadPoolExecutor) computes gradients over the samples it sees, accumulating the gradients to an on-GPU buffer.  At the end of each training batch, these buffers are gathered to a single device and given to the optimizer for the parameter update; the parameters are then scattered back to the GPUs.

### Loss/utility functions

* [trainer/dataloader.py](trainer/dataloader.py) contains the routines necessary to open a mirrored WeatherBench-like dataset (assumed to be stored as monthly .zarr DirectoryStores, but other arrangements might work).  All 'heavyweight' data manipulation, including assembling of the `inputs`, `targets`, and `forcings` Datasets per training example, is left as an unrealized Dask array so that the work can occur inside the dask.distributed client.  For performance reasons, this module also includes an ad-hoc "cache" consisting of time-based subsets of the database, since dask.optimize calls take a very, very long time to select the right portion of the ~50,000-long database.

* [trainer/grad_utils.py](trainer/grad_utils.py) contains utility functions for gradient manipulation, notably for accumulation and for creation of a zero gradient (as the baseline value).

* [trainer/loss_utils.py](trainer/loss_utils.py) contains a loss-function-builder.  (Subich 2024) allows for variation in vertical (pressure-based, in (Lam 2022)) loss-weighting, but the loss computation built into the GraphCast codebase applies fixed weighting.  Rather than change the GraphCast code in a messy way, the training loop instead just supplies a custom loss function to the loss/gradient-computing function, which is included as a closure for the purposes of Jax compilation.

## Diagnostic files

* [validate.py](validate.py) computes the validation loss of a provided set of checkpoints over a specific validation period.

* [build_scorecard.py](build_scorecard.py) computes an ECWMF-style "scorecard" for a single checkpoint over a provided period, creating forecasts up to N-steps in length to compute physically relevant quantities such as the root mean squared error and anomaly correlations.  This is much more comprehensive than the simple validation loss.

## WeatherBench / CDS mirroring

* [wb_download.py](wb_download.py) is a convenience script to mirror a portion of the [WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html) data to local storage.
* [cds_download.py](cds_download.py) and [cds_process.py](cds_process.py) are convenience scripts to manage downloads of ERA5 data from [ECMWF's climate data store](https://cds.climate.copernicus.eu/).  This was used for (Subich 2024) to download calendar-year 2023 data before it was available on WeatherBench (which is preferred for higher speed and simpler interface).

## Python environment

Development of this code and execution for (Subich 2024) occurred within a conda environment, the details of which are in [environment.yml](environment.yml).  The underlying execution envrionment was Red Hat Enterprise Linux release 8.3, using Intel Xeon CPUs and NVIDIA A100 (40GiB) GPUs, and the environment will likely require modification for other systems.

## License

As with the original version of GraphCast, this code is released under the Apache 2.0 license.
