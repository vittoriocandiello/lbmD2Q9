# LBM 2D Wave Solver

D2Q9 BGK Lattice Boltzmann for wave propagation in a 2D pipe. Single file, OpenMP parallelized.

## Build & Run

```bash
gcc -O3 -fopenmp -o solver solver.c
./solver                   # defaults: 300 steps, 200x20
./solver 1000 400 40       # ntimesteps nx ny
```

## Output

Results go in `output/`. Visualize with:

```bash
paraview output/solution.pvd
```
