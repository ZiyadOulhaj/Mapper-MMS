# Mapper Gromov-Wasserstein distance examples

We give examples of how Gromov-Wasserstein distances can be computed for Mapper graphs by treating them as metric-measure spaces.
The Gromov-Wasserstein computation is done using the [POT library](https://pythonot.github.io).

## filter_change.ipynb
We compute several Mapper graphs on the same 3d point cloud, by varying the filter function. We then compare the variation of the Gromov-Wasserstein distance to that of the filter function itself.

## measure_change.ipynb
We sample points from a torus using different measures and visualize the resulting distance matrix using MDS.
