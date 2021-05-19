# Outflow tracking

Code for tracking substructures in protostellar outflows.

The method linearly runs through the following steps:
 * Convert the simulation outputs (ndarrays) to grayscale images
 * Compute time-series of dendrograms from grayscale images
 * Initialize simple links between consecutive dendrograms using simulated annealing
 * Convert simple links to linking matrices and optimize using maximum likelihood estimation
 * Compute trajectories from optimized linking matrices

A _dendrogram_ is a tree data structure that describes how mass is hierarchically distributed in real space. A _link_ temporally connects a branch from one dendrogram to a single branch in the next dendrogram. A _linking matrix_ does the same, but allows mass from each branch in one dendrogram to be split up among all the branches in the next dendrogram. This flexibility allows the tracker to capture changes in the hierarchical structure of gas from frame to frame.

The notebook `read_hdf5.ipynb` produces numpy outputs from the raw simulation outputs. The numpy files should be placed in the folder `data/sim/`. The two notebooks `Tracking.ipynb` and `Trajectories.ipynb` contain all the tracking code. The first time dendrograms are computed, they are saved in the folder `data/dendros/`; similarly, after annealing performs initialization, the results are saved in `data/initialization.pickle`. (The `RECALCULATE` flag in `Tracking.ipynb` controls whether dendrogram and annealing results are recomputed.) Finally, the optimized linking matrices are saved in `results/linkmats.pickle`.

`Tracking.ipynb` takes care of the first four steps out of the five steps listed above. Trajectory extraction and analysis is done in `Trajectories.ipynb`. This notebook relies on the optimized linking matrices and dendrograms (which are saved to disk in the former notebook).

More details of the code are explained within the notebooks. If you have questions or clarifications, please contact dkarkada@gmail.com.