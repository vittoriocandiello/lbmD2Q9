#include "gaussianpulse.h"
#include "lbm.h"

#include <assert.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#define N_PROBES 10
#define AUX_PROBE_OFFSET 5
#define MIN_NX 24

static const scalar_t x_min = -4.0;
static const scalar_t x_max = 6.0;

static scalar_t acoustic_pressure(const State *state, size_t x, size_t y,
                                  scalar_t rho0,
                                  scalar_t pressure_lu_to_pa) {
    const size_t n = state->nx * state->ny;
    const size_t j = y * state->nx + x;
    scalar_t rho = 0.0;

    for (size_t i = 0; i < Q; ++i)
        rho += state->data[i * n + j];

    return cs2 * (rho - rho0) * pressure_lu_to_pa;
}

static void initialize_gaussian_pulse(State *state, Field *rho,
                                      Field *ux, Field *uy,
                                      scalar_t rho0, scalar_t amplitude,
                                      scalar_t sigma, scalar_t x0,
                                      scalar_t x_min, scalar_t dx,
                                      scalar_t pressure_lu_to_pa) {
    const size_t nx = state->nx;
    const size_t ny = state->ny;
    const scalar_t log2_value = log(2.0);

    set_constantfield(ux, 0.0);
    set_constantfield(uy, 0.0);

    #pragma omp parallel for collapse(2)
    for (size_t y = 0; y < ny; ++y) {
        for (size_t x = 0; x < nx; ++x) {
            const scalar_t position = x_min + (scalar_t)x * dx;
            const scalar_t distance = position - x0;
            const scalar_t pressure =
                amplitude * exp(-log2_value * distance * distance /
                                (sigma * sigma));
            rho->data[y * nx + x] =
                rho0 + pressure / (pressure_lu_to_pa * cs2);
        }
    }

    compute_equilibrium(state, rho, ux, uy);
}

static void set_y_periodic(State *state, const State *postcollision) {
    const size_t nx = state->nx;
    const size_t ny = state->ny;
    const size_t n = nx * ny;
    scalar_t *f = state->data;
    const scalar_t *fstar = postcollision->data;

    for (size_t x = 0; x < nx; ++x) {
        const size_t bottom = x;
        const size_t top = (ny - 1) * nx + x;

        for (size_t i = 0; i < Q; ++i) {
            if (cy[i] == 1) {
                const int xs = (int)x - cx[i];
                if (xs >= 0 && xs < (int)nx)
                    f[i * n + bottom] =
                        fstar[i * n + (ny - 1) * nx + (size_t)xs];
            } else if (cy[i] == -1) {
                const int xs = (int)x - cx[i];
                if (xs >= 0 && xs < (int)nx)
                    f[i * n + top] = fstar[i * n + (size_t)xs];
            }
        }
    }
}

static void set_x_bounceback(State *state) {
    const size_t nx = state->nx;
    const size_t ny = state->ny;
    const size_t n = nx * ny;
    scalar_t *f = state->data;

    for (size_t y = 0; y < ny; ++y) {
        const size_t left = y * nx;
        const size_t right = left + nx - 1;

        f[1 * n + left] = f[3 * n + left];
        f[5 * n + left] = f[7 * n + left];
        f[8 * n + left] = f[6 * n + left];

        f[3 * n + right] = f[1 * n + right];
        f[6 * n + right] = f[8 * n + right];
        f[7 * n + right] = f[5 * n + right];
    }
}

static FILE *open_probe_file(const char *path) {
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", path);
        abort();
    }

    fprintf(fp, "# Gaussian plane-wave probe data\n");
    fprintf(fp, "# Pressure is the acoustic component in Pa\n");
    fprintf(fp, "t_step,t_phys");
    for (size_t i = 0; i < N_PROBES; ++i)
        fprintf(fp, ",p_probe_%02zu,p_probe_%02zu_minus,p_probe_%02zu_plus",
                i, i, i);
    fprintf(fp, "\n");
    return fp;
}

static void write_probe_row(FILE *fp, const State *state, size_t timestep,
                            scalar_t dt, const size_t *probe_x,
                            size_t yprobe, scalar_t rho0,
                            scalar_t pressure_lu_to_pa) {
    fprintf(fp, "%zu,%.16e", timestep, (scalar_t)timestep * dt);
    for (size_t i = 0; i < N_PROBES; ++i) {
        const size_t x = probe_x[i];
        const scalar_t pressure =
            acoustic_pressure(state, x, yprobe, rho0, pressure_lu_to_pa);
        const scalar_t pressure_minus =
            acoustic_pressure(state, x - AUX_PROBE_OFFSET, yprobe,
                              rho0, pressure_lu_to_pa);
        const scalar_t pressure_plus =
            acoustic_pressure(state, x + AUX_PROBE_OFFSET, yprobe,
                              rho0, pressure_lu_to_pa);
        fprintf(fp, ",%.16e,%.16e,%.16e",
                pressure, pressure_minus, pressure_plus);
    }
    fprintf(fp, "\n");
}

static void write_metadata(const struct SimulationParameters *parameters,
                           scalar_t amplitude, scalar_t sigma, scalar_t x0,
                           scalar_t gas_constant, scalar_t p0, scalar_t t0,
                           scalar_t rho0_phys, scalar_t nu_physical,
                           scalar_t dx, scalar_t dt, scalar_t c0,
                           scalar_t pressure_lu_to_pa,
                           const scalar_t *probe_positions,
                           const size_t *probe_indices,
                           size_t yprobe) {
    FILE *fp = fopen("output/gaussianpulse_metadata.json", "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open output/gaussianpulse_metadata.json\n");
        abort();
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"case\": \"gaussianpulse\",\n");
    fprintf(fp, "  \"probe_csv\": \"output/gaussianpulse_probes.csv\",\n");
    fprintf(fp, "  \"amplitude_pa\": %.16e,\n", amplitude);
    fprintf(fp, "  \"sigma_m\": %.16e,\n", sigma);
    fprintf(fp, "  \"x0_m\": %.16e,\n", x0);
    fprintf(fp, "  \"gas_constant\": %.16e,\n", gas_constant);
    fprintf(fp, "  \"ambient_pressure_pa\": %.16e,\n", p0);
    fprintf(fp, "  \"ambient_temperature_k\": %.16e,\n", t0);
    fprintf(fp, "  \"rho0_lattice\": 1.0,\n");
    fprintf(fp, "  \"rho0_physical\": %.16e,\n", rho0_phys);
    fprintf(fp, "  \"nu_physical\": %.16e,\n", nu_physical);
    fprintf(fp, "  \"dx_m\": %.16e,\n", dx);
    fprintf(fp, "  \"dt_s\": %.16e,\n", dt);
    fprintf(fp, "  \"sound_speed_m_per_s\": %.16e,\n", c0);
    fprintf(fp, "  \"pressure_lu_to_pa\": %.16e,\n",
            pressure_lu_to_pa);
    fprintf(fp, "  \"nu_lattice\": %.16e,\n",
            get_lattice_viscosity());
    fprintf(fp, "  \"tau\": %.16e,\n", get_relaxation_time());
    fprintf(fp, "  \"omega\": %.16e,\n", get_relaxation_rate());
    fprintf(fp, "  \"ntimesteps\": %zu,\n", parameters->ntimesteps);
    fprintf(fp, "  \"nx\": %zu,\n", parameters->nx);
    fprintf(fp, "  \"ny\": %zu,\n", parameters->ny);
    fprintf(fp, "  \"domain_x_m\": [%.16e, %.16e],\n",
            x_min, x_max);
    fprintf(fp, "  \"boundary_x\": \"no-slip bounce-back\",\n");
    fprintf(fp, "  \"boundary_y\": \"periodic\",\n");
    fprintf(fp, "  \"probe_y_index\": %zu,\n", yprobe);
    fprintf(fp, "  \"aux_probe_offset_cells\": %d,\n",
            AUX_PROBE_OFFSET);
    fprintf(fp, "  \"probe_x_positions_m\": [");
    for (size_t i = 0; i < N_PROBES; ++i)
        fprintf(fp, "%s%.16e", i ? ", " : "", probe_positions[i]);
    fprintf(fp, "],\n");
    fprintf(fp, "  \"probe_x_indices\": [");
    for (size_t i = 0; i < N_PROBES; ++i)
        fprintf(fp, "%s%zu", i ? ", " : "", probe_indices[i]);
    fprintf(fp, "]\n");
    fprintf(fp, "}\n");
    fclose(fp);
}

int gaussianpulse(const struct SimulationParameters *parameters) {
    assert(parameters);
    if (parameters->nx < MIN_NX || parameters->ny < 3 ||
        parameters->ntimesteps == 0) {
        fprintf(stderr,
                "ERROR: nx must be at least %d, ny at least 3, "
                "and nt positive\n", MIN_NX);
        return 1;
    }

    if (mkdir("output", 0755) != 0 && errno != EEXIST) {
        perror("ERROR: cannot create output");
        return 1;
    }

    const scalar_t amplitude = 100.0;
    const scalar_t sigma = 0.01;
    const scalar_t x0 = -0.1;
    const scalar_t gas_constant = 287.0;
    const scalar_t p0 = 1.0e5;
    const scalar_t t0 = 293.0;
    const scalar_t nu_physical = 1.5e-2;
    const scalar_t dx =
        (x_max - x_min) / (scalar_t)(parameters->nx - 1);
    const scalar_t rho0 = 1.0;
    const scalar_t rho0_phys = p0 / (gas_constant * t0);
    const scalar_t c0 = sqrt(gas_constant * t0);
    const scalar_t dt = dx * sqrt(cs2) / c0;
    const scalar_t pressure_lu_to_pa = p0 / (cs2 * rho0);
    scalar_t probe_positions[N_PROBES];
    size_t probe_indices[N_PROBES];
    for (size_t i = 0; i < N_PROBES; ++i) {
        probe_positions[i] = 0.4 * (scalar_t)(i + 1);
        const long long index =
            llround((probe_positions[i] - x_min) / dx);
        if (index < AUX_PROBE_OFFSET ||
            (size_t)index + AUX_PROBE_OFFSET >= parameters->nx) {
            fprintf(stderr, "ERROR: nx is too small for the fixed probes\n");
            return 1;
        }
        probe_indices[i] = (size_t)index;
    }

    const scalar_t nu_lattice = nu_physical * dt / (dx * dx);
    set_lattice_viscosity(nu_lattice);
    const size_t saveevery =
        parameters->saveevery ? parameters->saveevery :
        (parameters->ntimesteps / 10 ? parameters->ntimesteps / 10 : 1);

    printf("================================================\n");
    printf(" LATTICE BOLTZMANN GAUSSIAN PULSE (D2Q9)\n");
    printf("================================================\n");
    printf(" Grid:       %zu x %zu\n", parameters->nx, parameters->ny);
    printf(" x-domain:   [%.4f, %.4f] m\n", x_min, x_max);
    printf(" Steps:      %zu\n", parameters->ntimesteps);
    printf(" Save every: %zu%s\n", saveevery,
           parameters->save_solution ? "" : " (disabled)");
    printf(" Pulse:      A=%.3e Pa, sigma=%.3e m, x0=%.3e m\n",
           amplitude, sigma, x0);
    printf(" tau:        %.8f\n", get_relaxation_time());
    printf("------------------------------------------------\n");

    State *state = create_state(parameters->nx, parameters->ny);
    State *postcollision = create_state(parameters->nx, parameters->ny);
    State *equilibrium = create_state(parameters->nx, parameters->ny);
    Field *rho = create_scalarfield(parameters->nx, parameters->ny);
    Field *ux = create_scalarfield(parameters->nx, parameters->ny);
    Field *uy = create_scalarfield(parameters->nx, parameters->ny);

    initialize_gaussian_pulse(state, rho, ux, uy, rho0, amplitude,
                              sigma, x0, x_min, dx,
                              pressure_lu_to_pa);

    const size_t yprobe = parameters->ny / 2;
    write_metadata(parameters, amplitude, sigma, x0,
                   gas_constant, p0, t0, rho0_phys, nu_physical,
                   dx, dt, c0, pressure_lu_to_pa,
                   probe_positions, probe_indices, yprobe);
    FILE *probe_file = open_probe_file("output/gaussianpulse_probes.csv");
    write_probe_row(probe_file, state, 0, dt, probe_indices,
                    yprobe, rho0, pressure_lu_to_pa);

    scalar_t *pressure_buffer = NULL;
    scalar_t *velocity_buffer = NULL;
    if (parameters->save_solution) {
        pressure_buffer = malloc(parameters->nx * parameters->ny *
                                 sizeof(*pressure_buffer));
        velocity_buffer = malloc(3 * parameters->nx * parameters->ny *
                                 sizeof(*velocity_buffer));
        if (!pressure_buffer || !velocity_buffer) {
            fprintf(stderr, "ERROR: RUN OUT OF MEMORY\n");
            abort();
        }
        save_macros(rho, ux, uy, "output/solution", 0,
                    pressure_buffer, velocity_buffer);
    }

    const size_t progress_interval =
        parameters->ntimesteps / 100 ? parameters->ntimesteps / 100 : 1;
    const double start = get_time();
    for (size_t t = 1; t <= parameters->ntimesteps; ++t) {
        compute_macros(state, rho, ux, uy);
        compute_equilibrium(equilibrium, rho, ux, uy);
        relaxation_step(state, postcollision, equilibrium);
        streaming(state, postcollision);
        set_y_periodic(state, postcollision);
        set_x_bounceback(state);

        write_probe_row(probe_file, state, t, dt, probe_indices,
                        yprobe, rho0, pressure_lu_to_pa);

        if (parameters->save_solution &&
            (t % saveevery == 0 || t == parameters->ntimesteps)) {
            compute_macros(state, rho, ux, uy);
            save_macros(rho, ux, uy, "output/solution", t,
                        pressure_buffer, velocity_buffer);
        }

        if (t % progress_interval == 0 || t == parameters->ntimesteps) {
            printf("\r Progress: [%3.0f%%]  Step: %zu/%zu",
                   100.0 * (double)t / (double)parameters->ntimesteps,
                   t, parameters->ntimesteps);
            fflush(stdout);
        }
    }
    const double elapsed = get_time() - start;
    fclose(probe_file);

    if (parameters->save_solution)
        save_pvd("output/solution", parameters->ntimesteps, saveevery);

    printf("\n------------------------------------------------\n");
    printf(" Simulation completed in %.2f seconds\n", elapsed);
    printf(" Wrote output/gaussianpulse_probes.csv and "
           "output/gaussianpulse_metadata.json\n");

    destroy_state(state);
    destroy_state(postcollision);
    destroy_state(equilibrium);
    destroy_field(rho);
    destroy_field(ux);
    destroy_field(uy);
    free(pressure_buffer);
    free(velocity_buffer);
    return 0;
}
