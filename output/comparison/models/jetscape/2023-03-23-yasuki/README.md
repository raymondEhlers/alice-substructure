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

## R = 0.4

Yasuki provided predictions for R = 0.4 after QM with the following notes:

Just in case, I have two different jet pseudorapidity cuts: 0-0.5 and 0-0.7 and two different charged jet pT cuts: 60-80 GeV and 80-100 GeV.
You can find those values in the filename e.g., ptj60-80_rapj0.0-0.5. In the same way as the grooming parameter values.

---

As just references, results from Pythia Monash2013 with options:

HardQCD:all=on, PartonLevel:ISR = on, PartonLevel:MPI = on, PartonLevel:FSR = on, ParticleDecays:limitTau0=on, ParticleDecays:tau0Max = 10

are provided

### Notes

- I haven't copied in the PYTHIA since I already have the PYTHIA results from ALICE)
- I had to rename to add a `pp_` prefix to the files to match the R=0.2. I did this in the `combined/pp`
  directory

