# Notes from Yasuki

Please note that the centrality classes have not been combined yet.

The normalization is based on the number of all triggered charged jets, including those failing the Soft Drop condition. This is in accordance with James' paper. If you want to have normalization by the number of jets passing the Soft Drop, please normalize by the integral of the histogram.

Here, all data are stored in ASCII text files.
In files, each row contains the values in the order:

> x_center, x_bin_low, x_bin_high, y, y_err_stat, extra_value_1(0s), extra_value_2(0s)

The `combined` folder just copies all of the files into a convenient location.

## DyG

You can find the parameters in grooming right before the filename extensions, e.g.

`..._aDyn0.50_zCut0.20.txt`

means a= 0.5 with iterative splittings z > 0.2, and

`..._aDyn0.50_zCut0.00.txt`

means standard dynamical grooming with a= 0.5  and without any cut for z
