# LBM 2D Gaussian Pulse Solver

Compact OpenMP D2Q9 BGK lattice Boltzmann solver for a Gaussian acoustic
plane-wave pulse. The numerical kernel is in `src/lbm.c`; the simulation
setup, boundary conditions, probes, and output are in
`src/gaussianpulse.c`.

## Build & Run

```bash
make
./solver
./solver -nx=2500 -ny=8 -nt=1000
./solver --help
```

Options:

- `-nx=N`: streamwise grid size (default `5000`, minimum `24`)
- `-ny=N`: periodic transverse grid size (default `8`, minimum `3`)
- `-nt=N`: number of time steps (default `5000`)
- `-save=N`: write a VTK snapshot every `N` steps
- `-no-sol`: disable VTK solution output

By default, a solution snapshot is saved every 10% of the run. The physical
x-domain is fixed to `[-4, 6] m`; changing `nx` changes its resolution.
Left/right boundaries use no-slip bounce-back and top/bottom are periodic.
The ten probes remain fixed from `0.4 m` to `4.0 m` and are sampled every
time step.

## Output

Every run writes:

- `output/gaussianpulse_probes.csv`
- `output/gaussianpulse_metadata.json`

Unless `-no-sol` is used, the run also writes `output/solution.pvd` and
binary VTK image files. Visualize them with:

```bash
paraview output/solution.pvd
```

## Validation plots

The post-processor requires only NumPy and Matplotlib. It reads the probe CSV
and matching metadata from `output/` and writes the dissipation and dispersion
error plots there. Only resolved wavelengths with `NPPW >= 20` are plotted.

```bash
python3 validation_post.py
python3 validation_post.py --input path/to/gaussianpulse_probes.csv
```
