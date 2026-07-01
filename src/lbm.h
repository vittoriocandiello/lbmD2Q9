#ifndef LBM_H
#define LBM_H

#include <stddef.h>

#define DIM 2
#define Q 9

typedef double scalar_t;

extern const scalar_t cs2;
extern const int cx[Q];
extern const int cy[Q];
extern const scalar_t w[Q];

typedef struct {
    size_t nx;
    size_t ny;
    scalar_t *data;
} State;

typedef struct {
    size_t nx;
    size_t ny;
    scalar_t *data;
} Field;

double get_time(void);

void set_lattice_viscosity(scalar_t viscosity);
scalar_t get_lattice_viscosity(void);
scalar_t get_relaxation_time(void);
scalar_t get_relaxation_rate(void);

State *create_state(size_t nx, size_t ny);
void destroy_state(State *state);

Field *create_scalarfield(size_t nx, size_t ny);
void set_constantfield(Field *field, scalar_t value);
void destroy_field(Field *field);

void compute_macros(const State *state, Field *rho, Field *ux, Field *uy);
void compute_equilibrium(State *equilibrium, const Field *rho,
                         const Field *ux, const Field *uy);
void relaxation_step(const State *state, State *postcollision,
                     const State *equilibrium);
void streaming(State *state, const State *postcollision);

void save_macros(const Field *rho, const Field *ux, const Field *uy,
                 const char *basename, size_t timestep,
                 scalar_t *pressure_buffer, scalar_t *velocity_buffer);
void save_pvd(const char *basename, size_t last_timestep, size_t saveevery);

#endif
