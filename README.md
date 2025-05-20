# GraphCast-AMSE

GraphCast model supporting AMSE-based trianing, per [Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified Spherical Harmonic Loss Function](https://arxiv.org/abs/2501.19374).

This branch is closely based on the [graphcast-train](https://github.com/csubich/graphcast/tree/graphcast_train) branch, and training works in the same way.  To train with the AMSE error function, invoke `train.py` with the `--spectral-amse`` command-line option.

This code retains the Apache 2.0 license of GraphCast.
