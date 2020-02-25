# Large-Scale Spatiotemporal Density Smoothing with the Graph-fused Elastic Net

This repository contains all the code to necessary to reproduce our results. Here's a brief description of the contents of the repository. See the file `MANIFEST.Rmd` for a description of software versions I used. For the preprocessing steps it is necessary to have `R` and `python 3`. For convenience, I include the processed data in this repository, so that you don't have to go through those steps. The main algorithm is implemented using the high-performance computing software `Julia` with an open-source package we developed for this paper called `GraphFusedElasticNet.jl`. Below we provide further instructions.


## 1. Raw Data

Our data consists of approximately 1.4 million trips from RideAustin. The raw database can be found at [`data.world`](https://data.world/ride-austin). For convenience, it can also be found in this repository as we used in in our analysis in the folder `raw_data/`. We also use geographical data from Traffic Analysis Zones (TAZs) in the city of Austin. The shapefiles for the TAZs are in the folder `raw_data/shapefiles/`.


## 2. Data and Graph Preprocessing

We clean the data and obtain a measure of the productivity in $/hours as explained in our paper for each trip. This is the expected productivity given the location of a trip. The cleaned database is in the folder `processed_data/rideaustin_productivity`. This dataset also contains other generated quantities (e.g., trip start and end coordinates, idle time, reach time, duration of subsequent trip) which are used to compute the productivity measure.

To reproduce the cleaning steps it is necessary to run the R script `1_process_data_and_adjacency.R`. This script also uses the TAZ polygon information to find adjacent TAZs, the adjacency information is in the file `processed_data/taz_adjacency.csv`. Finally, it saves the quantiles of the all-combined empirical distribution of productivity that will be used a splitting values for our binary tree density estimation approach and saves them in the file `processed_data/splitlevels.csv`.


To perform spatiotemporal smoothing we need a spatiotemporal graph that contains edges from temporal and spatial adjacency. The graph edges are generated in the Python script `2_process_graph_and_splits.py` with approx. 200k nodes and 950k edges. The vertices of the graph are enumerated and information about vertex and space-TAZ correspondence is stored in the file `processed_data/vertex_data.csv`. This Python script also partitions the productivity data using the split levels generated in the preprocessing step. The resulting splits are stored in the folder `productivity_splits/`. There is exactly one file per split, for which we will training a Binomial model. The format of the data is the following, corresponding to a split `(a, b)`, each row contains to entries `(s_i, n_i)` where `n_i` are the number of counts occurring in the range `(a, b)` and `s_i` are the number of occurrences in the range `(a, (a + b) / 2)`. See our paper for a longer description.


## 3. Model fitting


### Julia Package Installation

All the algorithms to run the GFEN are available in our Julia package `GraphFusedElasticNet.jl`. To install the package permanently you can use the following command in the Julia REPL:

```julia
using Pkg
Pkg.add("https://github.com/mauriciogtec/GraphFusedElasticNet.jl")
```

In the future, after we test it in a few other problems, we will upload these package as an official Julia package. 

As an alternative to direct installing the package in the system, the package can be loaded directly from the source code. Our Julia package is registered as a git submodule of this Github repository. To clone the repository and activate the submodule one must execute the following command when cloning from this repo:

```bash
git clone --recurse-submodules https://github.com/mauriciogtec/gfen-reproduce
```

All the Julia scripts that we describe below assume that the package source is locally available as when using the above command. To load the package locally in Julia one can run

```julia
using Pkg
Pkg.add("https://github.com/mauriciogtec/GraphFusedElasticNet.jl")
```

```julia
Pkg.activate("./GraphFusedElasticNet.jl/")
Pkg.instantiate()
```

This will create a local environment to run the code that guarantees the right dependencies. Remove these lines if you want to install the code in the base machine environment.


### Trails


For our scalable algorithm we need to decompose the graph into trails. These trails are the basis for scalable performance since can be smoothed in linear time. The script that generates the trails is `3_generate_trails.jl`. It will produce the file `processed_data/trails.json` that contains the resulting trails. It contains for objects: a _pointer_ array that shows the nodes visited sequentially by the trail, note that all the trails are concatenated; a _breaks_ array indicates the beginning index and end index of each trail in the pointer; a _temporal indicator_ array indicates if the edge in-between visited vertices in the order specified by the pointer is temporal or spatial (used to assign different hyper-parameters); finally, since the algorithm can be run with a trail partition that visits each node more than once, there is a _weights_ array inversely proportional to the number of visits.


### Model fit with Bayesian Optimization

The file `4_modelfit_script.jl` executes a Graph Fused Elastic-Net Binomial model at each split level. Each time it is executed it will produce a file with model fit metrics such as cross-validation error and the fitted Gaussian process for the hyper-parameters at the folder `modelfit_metrics`. It will use the best hyper-parameters to fit a model and the output is stored at `best_betas`.


![smoothing-animation](./map_animation.gif)