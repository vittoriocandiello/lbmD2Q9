#ifndef GAUSSIANPULSE_H
#define GAUSSIANPULSE_H

#include <stddef.h>

struct SimulationParameters {
    size_t nx;
    size_t ny;
    size_t ntimesteps;
    size_t saveevery;
    int save_solution;
};

int gaussianpulse(const struct SimulationParameters *parameters);

#endif
