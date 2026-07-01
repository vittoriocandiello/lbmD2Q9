#include "gaussianpulse.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *program) {
    printf("Usage: %s [-nx=N] [-ny=N] [-nt=N] [-save=N] [-no-sol]\n",
           program);
}

static int parse_size(const char *argument, const char *name, size_t *value) {
    const size_t length = strlen(name);
    if (strncmp(argument, name, length) != 0) return 0;

    char *end;
    errno = 0;
    if (argument[length] == '-') return -1;
    const unsigned long long parsed = strtoull(argument + length, &end, 10);
    if (errno || end == argument + length || *end != '\0' ||
        parsed > SIZE_MAX)
        return -1;

    *value = (size_t)parsed;
    return 1;
}

int main(int argc, char **argv) {
    struct SimulationParameters parameters = {
        .nx = 5000,
        .ny = 8,
        .ntimesteps = 5000,
        .saveevery = 0,
        .save_solution = 1
    };

    for (int i = 1; i < argc; ++i) {
        int parsed = parse_size(argv[i], "-nx=", &parameters.nx);
        if (parsed == 0) parsed = parse_size(argv[i], "-ny=", &parameters.ny);
        if (parsed == 0)
            parsed = parse_size(argv[i], "-nt=", &parameters.ntimesteps);
        if (parsed < 0) {
            fprintf(stderr, "Invalid argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
        if (parsed > 0) continue;

        parsed = parse_size(argv[i], "-save=", &parameters.saveevery);
        if (parsed < 0 || (parsed > 0 && parameters.saveevery == 0)) {
            fprintf(stderr, "Invalid argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
        if (parsed > 0) {
            parameters.save_solution = 1;
            continue;
        } else if (strcmp(argv[i], "-no-sol") == 0) {
            parameters.save_solution = 0;
        } else if (strcmp(argv[i], "-h") == 0 ||
                   strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    return gaussianpulse(&parameters);
}
