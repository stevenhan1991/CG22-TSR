# CG22-TSR
Pytorch implementation for TSR-VFD: Generating Temporal Super-resolution for Unsteady Vector Field Data.

## Data format

The vector data at each time step is saved as a .dat file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis.

## Archiecture

TSR-VFD contains two branches. The first branch predicts the intermediate vector fields given two endding vectors. The second branch produces masks for sovling the problem of unbalance value range on different vector componenet.


## Train

```
cd Code 
```

```
python3 main.py 
```

## Citation 
```
@article{HAN-CG22,
title = {TSR-VFD: Generating temporal super-resolution for unsteady vector field data},
journal = {Computers & Graphics},
volume = {103},
pages = {168-179},
year = {2022},
author = {Jun Han and Chaoli Wang},
}
```
