/*
 * Compact 2D Lattice Boltzmann D2Q9 BGK implementation.
 */
#include "lbm.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

const scalar_t cs2 = 1.0 / 3.0;
const int cx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const scalar_t w[Q] = {
    4.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

static scalar_t viscosity = 0.033333333333;
static scalar_t tau = 0.6;
static scalar_t omega = 1.0 / 0.6;

static void allocation_error(void) {
    fprintf(stderr, "ERROR: RUN OUT OF MEMORY\n");
    abort();
}

static void *allocate(size_t count, size_t size) {
    if (size != 0 && count > SIZE_MAX / size)
        allocation_error();

    void *ptr = malloc(count * size);
    if (!ptr) allocation_error();
    return ptr;
}

static void check_state(const State *state) {
    assert(state);
    assert(state->data);
}

static void check_field(const Field *field, size_t nx, size_t ny) {
    assert(field);
    assert(field->data);
    assert(field->nx == nx && field->ny == ny);
}

double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void set_lattice_viscosity(scalar_t value) {
    assert(value > 0.0);
    viscosity = value;
    tau = 0.5 + viscosity / cs2;
    omega = 1.0 / tau;
}

scalar_t get_lattice_viscosity(void) {
    return viscosity;
}

scalar_t get_relaxation_time(void) {
    return tau;
}

scalar_t get_relaxation_rate(void) {
    return omega;
}

State *create_state(size_t nx, size_t ny) {
    assert(nx > 0 && ny > 0);
    if (nx > SIZE_MAX / ny) allocation_error();
    const size_t n = nx * ny;

    State *state = allocate(1, sizeof(*state));
    state->nx = nx;
    state->ny = ny;
    state->data = allocate(n, Q * sizeof(*state->data));
    return state;
}

void destroy_state(State *state) {
    if (!state) return;
    free(state->data);
    free(state);
}

Field *create_scalarfield(size_t nx, size_t ny) {
    assert(nx > 0 && ny > 0);
    if (nx > SIZE_MAX / ny) allocation_error();
    const size_t n = nx * ny;

    Field *field = allocate(1, sizeof(*field));
    field->nx = nx;
    field->ny = ny;
    field->data = allocate(n, sizeof(*field->data));
    return field;
}

void set_constantfield(Field *field, scalar_t value) {
    check_field(field, field->nx, field->ny);
    const size_t n = field->nx * field->ny;

    if (value == 0.0) {
        memset(field->data, 0, n * sizeof(*field->data));
    } else {
        #pragma omp parallel for
        for (size_t j = 0; j < n; ++j)
            field->data[j] = value;
    }
}

void destroy_field(Field *field) {
    if (!field) return;
    free(field->data);
    free(field);
}

void compute_macros(const State *state, Field *rho, Field *ux, Field *uy) {
    check_state(state);
    const size_t nx = state->nx;
    const size_t ny = state->ny;
    const size_t n = nx * ny;
    check_field(rho, nx, ny);
    check_field(ux, nx, ny);
    check_field(uy, nx, ny);

    const scalar_t *f = state->data;
    #pragma omp parallel for
    for (size_t j = 0; j < n; ++j) {
        scalar_t density = 0.0;
        scalar_t momentum_x = 0.0;
        scalar_t momentum_y = 0.0;

        for (size_t i = 0; i < Q; ++i) {
            const scalar_t fi = f[i * n + j];
            density += fi;
            momentum_x += fi * cx[i];
            momentum_y += fi * cy[i];
        }

        assert(density > 0.0);
        rho->data[j] = density;
        ux->data[j] = momentum_x / density;
        uy->data[j] = momentum_y / density;
    }
}

void compute_equilibrium(State *equilibrium, const Field *rho,
                         const Field *ux, const Field *uy) {
    check_state(equilibrium);
    const size_t nx = equilibrium->nx;
    const size_t ny = equilibrium->ny;
    const size_t n = nx * ny;
    check_field(rho, nx, ny);
    check_field(ux, nx, ny);
    check_field(uy, nx, ny);

    scalar_t *feq = equilibrium->data;
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < Q; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const scalar_t vx = ux->data[j];
            const scalar_t vy = uy->data[j];
            const scalar_t cu = cx[i] * vx + cy[i] * vy;
            const scalar_t uu = vx * vx + vy * vy;
            feq[i * n + j] =
                w[i] * rho->data[j] *
                (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * uu);
        }
    }
}

void relaxation_step(const State *state, State *postcollision,
                     const State *equilibrium) {
    check_state(state);
    check_state(postcollision);
    check_state(equilibrium);
    assert(postcollision->nx == state->nx && postcollision->ny == state->ny);
    assert(equilibrium->nx == state->nx && equilibrium->ny == state->ny);

    const size_t count = Q * state->nx * state->ny;
    const scalar_t *f = state->data;
    const scalar_t *feq = equilibrium->data;
    scalar_t *fstar = postcollision->data;

    #pragma omp parallel for
    for (size_t i = 0; i < count; ++i)
        fstar[i] = f[i] - omega * (f[i] - feq[i]);
}

void streaming(State *state, const State *postcollision) {
    check_state(state);
    check_state(postcollision);
    assert(postcollision->nx == state->nx && postcollision->ny == state->ny);

    const size_t nx = state->nx;
    const size_t ny = state->ny;
    const size_t n = nx * ny;
    scalar_t *f = state->data;
    const scalar_t *fstar = postcollision->data;

    #pragma omp parallel for collapse(3)
    for (size_t i = 0; i < Q; ++i) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const int xs = (int)x - cx[i];
                const int ys = (int)y - cy[i];
                if (xs >= 0 && xs < (int)nx && ys >= 0 && ys < (int)ny) {
                    const size_t offset = i * n;
                    f[offset + y * nx + x] =
                        fstar[offset + (size_t)ys * nx + (size_t)xs];
                }
            }
        }
    }
}

void save_macros(const Field *rho, const Field *ux, const Field *uy,
                 const char *basename, size_t timestep,
                 scalar_t *pressure, scalar_t *velocity) {
    assert(basename);
    check_field(rho, rho->nx, rho->ny);
    check_field(ux, rho->nx, rho->ny);
    check_field(uy, rho->nx, rho->ny);
    assert(pressure && velocity);

    const size_t nx = rho->nx;
    const size_t ny = rho->ny;
    const size_t n = nx * ny;
    char filename[512];
    snprintf(filename, sizeof(filename), "%s_%06zu.vti", basename, timestep);

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", filename);
        abort();
    }

    for (size_t j = 0; j < n; ++j) {
        pressure[j] = cs2 * rho->data[j];
        velocity[3 * j] = ux->data[j];
        velocity[3 * j + 1] = uy->data[j];
        velocity[3 * j + 2] = 0.0;
    }

    const uint64_t scalar_nbytes = (uint64_t)(n * sizeof(scalar_t));
    const uint64_t vector_nbytes = (uint64_t)(3 * n * sizeof(scalar_t));
    const uint64_t offset_density = 0;
    const uint64_t offset_pressure = sizeof(uint64_t) + scalar_nbytes;
    const uint64_t offset_velocity =
        2 * sizeof(uint64_t) + 2 * scalar_nbytes;

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" "
                "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
    fprintf(fp, "  <ImageData WholeExtent=\"0 %zu 0 %zu 0 0\" "
                "Origin=\"0 0 0\" Spacing=\"1 1 1\">\n", nx - 1, ny - 1);
    fprintf(fp, "    <Piece Extent=\"0 %zu 0 %zu 0 0\">\n", nx - 1, ny - 1);
    fprintf(fp, "      <PointData>\n");
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"density\" "
                "format=\"appended\" offset=\"%llu\"/>\n",
            (unsigned long long)offset_density);
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"pressure\" "
                "format=\"appended\" offset=\"%llu\"/>\n",
            (unsigned long long)offset_pressure);
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"velocity\" "
                "NumberOfComponents=\"3\" format=\"appended\" offset=\"%llu\"/>\n",
            (unsigned long long)offset_velocity);
    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </ImageData>\n");
    fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");
    fwrite(&scalar_nbytes, sizeof(scalar_nbytes), 1, fp);
    fwrite(rho->data, sizeof(scalar_t), n, fp);
    fwrite(&scalar_nbytes, sizeof(scalar_nbytes), 1, fp);
    fwrite(pressure, sizeof(scalar_t), n, fp);
    fwrite(&vector_nbytes, sizeof(vector_nbytes), 1, fp);
    fwrite(velocity, sizeof(scalar_t), 3 * n, fp);
    fprintf(fp, "\n  </AppendedData>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);
}

void save_pvd(const char *basename, size_t last_timestep, size_t saveevery) {
    assert(basename && saveevery > 0);
    char filename[512];
    snprintf(filename, sizeof(filename), "%s.pvd", basename);

    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "ERROR: cannot open %s\n", filename);
        abort();
    }

    const char *leaf = strrchr(basename, '/');
    leaf = leaf ? leaf + 1 : basename;

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" "
                "byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <Collection>\n");
    for (size_t t = 0; t <= last_timestep; ++t) {
        if (t % saveevery == 0 || t == last_timestep) {
            fprintf(fp, "    <DataSet timestep=\"%zu\" group=\"\" part=\"0\" "
                        "file=\"%s_%06zu.vti\"/>\n", t, leaf, t);
        }
    }
    fprintf(fp, "  </Collection>\n");
    fprintf(fp, "</VTKFile>\n");
    fclose(fp);
}
