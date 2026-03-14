/* 
 * 2D Lattice Boltzmann Method D2Q9 BGK 
 * Vittorio Candiello Implementation
 *                                       */
#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<string.h>
#include<limits.h>
#include<time.h>
#include<omp.h>
#include<sys/stat.h>

/* ===== 2D Lattice Boltzmann Method --- Model Parameters ===== */
#define DIM 2
#define Q 9

typedef double scalar_t;

const static scalar_t cs2 = 1.0 / 3.0;
static scalar_t nu    = 0.033333333333;
static scalar_t tau   = 0.6;
static scalar_t omega = 1.0 / 0.6;

static const int cx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
static const int cy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
static const scalar_t w[Q] = {
    4.0/9.0,
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
};
/* ============================================================ */


/* =============== Data Structures =============== */

// population state structure //
typedef struct {
	size_t nx;
	size_t ny;
	scalar_t *data;
} State;

// scalar or vector field for pressure, density, velocity, forces //
typedef struct {
	size_t nx;
	size_t ny;
	scalar_t *data;
} Field;


/* =============================================== */

/* ================== Utils ====================== */
void check_memalloc(const void *ptr) {
	 if (!ptr) {
		 printf("ERROR: RUN OUT OF MEMORY\n");
		 assert(0);
	 }
}
void check_ptrvalidity(const void *ptr) {
	 if (!ptr) {
		 printf("ERROR: NOT A VALID POINTER\n");
		 assert(0);
	 }
}
void check_gridsizestate(const State *ptr, const size_t nx, const size_t ny) {
	if (ptr->nx != nx || ptr->ny != ny) {
		printf("ERROR: INVALID GRID SIZE\n");
		assert(0);
	}
}
void check_gridsizefield(const Field *ptr, const size_t nx, const size_t ny) {
	if (ptr->nx != nx || ptr->ny != ny) {
		printf("ERROR: INVALID GRID SIZE\n");
		assert(0);
	}
}
void check_positivity(const scalar_t value) {
	if (value <= 0) {
		printf("ERROR: NON POSITIVE SCALAR VALUE. IT MUST BE POSITIVE.\n");
		assert(0);
	}
}
size_t parse_intctl(int val) {
    if (val < 1) return 1;
    if (val > INT_MAX) return INT_MAX;
    return (size_t)val;
}
double get_time() { // Return clock time in seconds
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
/* =============================================== */

/* ================== State Functions  =================== */
State *create_state (size_t nx, size_t ny) {
	if (nx < 1 || ny < 1) {printf("ERROR: not enough dof for a well defined state\n"); assert(0);}
	State *ptr = malloc(sizeof(*ptr));
	check_memalloc(ptr);
	ptr->nx = nx;
	ptr->ny = ny;
	ptr->data = malloc(Q * nx * ny * sizeof(scalar_t));
	check_memalloc(ptr->data);
	return ptr;
}

void set_initialstate_rest(State *state, scalar_t rho0) {
	check_ptrvalidity(state);
	check_positivity(rho0);

    scalar_t *statedata = state->data;
	const size_t nx = state->nx;
	const size_t ny = state->ny;
	const size_t n = nx * ny;

	#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < Q; ++i)
		for (size_t j = 0; j < n; ++j) {
			const scalar_t f_i = w[i] * rho0;
			const size_t itimesn = i * n;
			statedata[itimesn + j] = f_i;
		}
}

void set_initialstate_perturbed(State *state, scalar_t rho0) {
	check_ptrvalidity(state);
	check_positivity(rho0);

    scalar_t *statedata = state->data;
	const size_t nx = state->nx;
	const size_t ny = state->ny;
	const size_t n = nx * ny;
    const scalar_t eps = 1e-2 * rho0;

	#pragma omp parallel for collapse(3)
	for (size_t i = 0; i < Q; ++i) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
				const scalar_t f_i = w[i] * rho0;
				const size_t itimesn = i * n;
				const size_t ytimesnx = y * nx;
				const size_t x0 = nx / 4; 
                const scalar_t xx0 = ((scalar_t)x - (scalar_t)x0);
				const size_t y0 = ny / 2; 
                const scalar_t yy0 = ((scalar_t)y - (scalar_t)y0);
                statedata[itimesn + ytimesnx + x] = f_i + eps * (1.0 / (1.0 + 0.01*xx0*xx0 + 0.03*yy0*yy0));
            }
        }
	}
}

void destroy_state(State *ptr) {
	if (ptr) free(ptr->data);
	free(ptr);
}
/* ======================================================= */


/* ================== Field Functions  =================== */
Field *create_scalarfield (size_t nx, size_t ny) {
	if (nx < 1 || ny < 1) {printf("ERROR: not enough dof for a well defined state\n"); assert(0);}
	Field *ptr = malloc(sizeof(*ptr));
	check_memalloc(ptr);
	ptr->nx = nx;
	ptr->ny = ny;
	ptr->data = malloc(nx * ny * sizeof(scalar_t));
	check_memalloc(ptr->data);
	return ptr;
}

void set_constantfield(Field *field, scalar_t value) {
	check_ptrvalidity(field);
	scalar_t *data = field->data;
	size_t n = field->nx * field->ny;
	if (value == 0.0) 
		memset(data, 0, n * sizeof(scalar_t));
	else { 
		#pragma omp parallel for
		for (size_t i = 0; i < n; ++i) data[i] = value;
	}
}

void destroy_field(Field *ptr) {
	if (ptr) free(ptr->data);
	free(ptr);
}
/* ======================================================= */


/* ===================== From State to Field functions ================== */
void compute_density(const State *state, Field *rho) {
	check_ptrvalidity(state);
	const size_t nx = state->nx;
    const size_t ny = state->ny;
	check_ptrvalidity(rho); check_gridsizefield(rho, nx, ny);
	// maybe add a check for the positivity of the rho field

	const scalar_t *f = state->data;
	scalar_t *rhodata = rho->data;
	size_t n = nx * ny;
	#pragma omp parallel for
	for (size_t j = 0; j < n; j++) {
		scalar_t rholocal = 0.0;
		for (size_t i = 0; i < Q; ++i) rholocal += f[i * n + j];
		rhodata[j] = rholocal;
	}
}

void compute_momentumx(const State *state, Field *momentumx) {
	check_ptrvalidity(state);
	const size_t nx = state->nx;
    const size_t ny = state->ny;
	check_ptrvalidity(momentumx); check_gridsizefield(momentumx, nx, ny);

	const scalar_t *f = state->data;
	scalar_t *momentumxdata = momentumx->data;
	const size_t n = nx * ny;
	#pragma omp parallel for
	for (size_t j = 0; j < n; j++) {
		scalar_t momentumxlocal = 0.0;
		for (size_t i = 0; i < Q; ++i) momentumxlocal += f[i * n + j] * cx[i];
		momentumxdata[j] = momentumxlocal;
	}
}

void compute_momentumy(const State *state, Field *momentumy) {
	check_ptrvalidity(state);
	const size_t nx = state->nx;
    const size_t ny = state->ny;
	check_ptrvalidity(momentumy); check_gridsizefield(momentumy, nx, ny);

	const scalar_t *f = state->data;
	scalar_t *momentumydata = momentumy->data;
	const size_t n = nx * ny;
	#pragma omp parallel for
	for (size_t j = 0; j < n; j++) {
		scalar_t momentumylocal = 0.0;
		for (size_t i = 0; i < Q; ++i) momentumylocal += f[i * n + j] * cy[i];
		momentumydata[j] = momentumylocal;
	}
}

void compute_ux(const State *state, Field *ux) {
	check_ptrvalidity(state);
	const size_t nx = state->nx;
    const size_t ny = state->ny;
	check_ptrvalidity(ux); check_gridsizefield(ux, nx, ny);

	const scalar_t *f = state->data;
	scalar_t *uxdata = ux->data;
	const size_t n = nx * ny;
	#pragma omp parallel for
	for (size_t j = 0; j < n; j++) {
		scalar_t momentumxlocal = 0.0;
		scalar_t rholocal = 0.0;
		for (size_t i = 0; i < Q; ++i) {
			const scalar_t fi = f[i * n + j];
			rholocal += fi;
			momentumxlocal += fi * cx[i];
		}
        assert(rholocal > 0);
		uxdata[j] = momentumxlocal / rholocal;
	}
}

void compute_uy(const State *state, Field *uy) {
	check_ptrvalidity(state);
	const size_t nx = state->nx;
    const size_t ny = state->ny;
	check_ptrvalidity(uy); check_gridsizefield(uy, nx, ny);

	const scalar_t *f = state->data;
	scalar_t *uydata = uy->data;
	const size_t n = nx * ny;
	#pragma omp parallel for
	for (size_t j = 0; j < n; j++) {
		scalar_t momentumylocal = 0.0;
		scalar_t rholocal = 0.0;
		for (size_t i = 0; i < Q; ++i) {
			const scalar_t fi = f[i * n + j];
			rholocal += fi;
			momentumylocal += fi * cy[i];
		}
        assert(rholocal > 0);
		uydata[j] = momentumylocal / rholocal;
	}
}

void compute_macros(const State *state, Field *rho, Field *ux, Field *uy) {
	check_ptrvalidity(state);
	const size_t nx = state->nx;
    const size_t ny = state->ny;
	check_ptrvalidity(rho); check_gridsizefield(rho, nx, ny);
	check_ptrvalidity(ux); check_gridsizefield(ux, nx, ny);
	check_ptrvalidity(uy); check_gridsizefield(uy, nx, ny);

	const scalar_t *f = state->data;
	scalar_t *rhodata = rho->data;
	scalar_t *uxdata = ux->data;
	scalar_t *uydata = uy->data;
	const size_t n = nx * ny;
    #pragma omp parallel for
	for (size_t j = 0; j < n; j++) {
		scalar_t momentumxlocal = 0.0;
		scalar_t momentumylocal = 0.0;
		scalar_t rholocal = 0.0;
		for (size_t i = 0; i < Q; ++i) {
			const scalar_t fi = f[i * n + j];
			rholocal += fi;
			momentumxlocal += fi * cx[i];
			momentumylocal += fi * cy[i];
		}
        assert(rholocal > 0);
		rhodata[j] = rholocal;
		uxdata[j] = momentumxlocal / rholocal;
		uydata[j] = momentumylocal / rholocal;
	}
}

void compute_pressure(const Field *rho, Field *pressure) {
    check_ptrvalidity(rho);
    check_ptrvalidity(pressure);
    check_gridsizefield(pressure, rho->nx, rho->ny);

    const size_t n = rho->nx * rho->ny;
    const scalar_t *rhodata = rho->data;
    scalar_t *pdata = pressure->data;

    for (size_t j = 0; j < n; ++j) pdata[j] = cs2 * rhodata[j];
}

void compute_equilibrium(State *equilibriumstate, const Field *rho, const Field *ux, const Field *uy) {
	check_ptrvalidity(equilibriumstate);
	const size_t nx = equilibriumstate->nx;
    const size_t ny = equilibriumstate->ny;
	check_ptrvalidity(rho); check_gridsizefield(rho, nx, ny);
	check_ptrvalidity(ux); check_gridsizefield(ux, nx, ny);
	check_ptrvalidity(uy); check_gridsizefield(uy, nx, ny);

	scalar_t *feq = equilibriumstate->data;
	const scalar_t *rhodata = rho->data;
	const scalar_t *uxdata = ux->data;
	const scalar_t *uydata = uy->data;
	const size_t n = nx * ny;

	#pragma omp parallel for collapse(2)
	for (size_t i = 0; i < Q; ++i) {
		for (size_t j = 0; j < n; ++j) {
			const scalar_t uxj = uxdata[j];
			const scalar_t uyj = uydata[j];
			const scalar_t cu = cx[i] * uxj + cy[i] * uyj;
			const scalar_t uu = uxj * uxj + uyj * uyj;
			feq[i * n + j] = w[i] * rhodata[j] * (1.0 + 3.0 * cu +  4.5 * cu * cu - 1.5 * uu);
		}
	}
}

void save_macros(const Field *rho,
                 const Field *ux,
                 const Field *uy,
                 const char *basename,
                 size_t timestep);
void save_pvd(const char *basename, size_t ntimesteps, size_t saveevery);

/* ================================================================= */

/* ===================== Time Evolution Functions ================== */
void relaxation_step(const State *state, State *statestar, const State *equilibriumstate) {
	check_ptrvalidity(state);
	check_ptrvalidity(statestar);
	check_ptrvalidity(equilibriumstate);
	const size_t nx = state->nx;
	const size_t ny = state->ny;
	check_gridsizestate(statestar, nx, ny);
	check_gridsizestate(equilibriumstate, nx, ny);
	
	const scalar_t *f = state->data;
	scalar_t *fstar = statestar->data;
	const scalar_t *feq = equilibriumstate->data;

	const size_t n = nx * ny;
	#pragma omp parallel for
	for (size_t i = 0; i < n * Q; ++i) fstar[i] = f[i] - omega * (f[i] - feq[i]);
}

void streaming (State *state, const State *statestar) {
    check_ptrvalidity(state);
    check_ptrvalidity(statestar);
    const size_t nx = state->nx;
    const size_t ny = state->ny;
    check_gridsizestate(statestar, nx, ny);

    scalar_t *f = state->data;
    const scalar_t *fstar = statestar->data;

    const size_t n = nx * ny;

	#pragma omp parallel for collapse(3)
    for (size_t i = 0; i < Q; ++i) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const int xs = (int)x - cx[i];
                const int ys = (int)y - cy[i];
                if ( (xs >= 0 && xs < (int)nx) && (ys >= 0 && ys < (int)ny) ) {
					const size_t itimesn = i * n;
					const size_t idxsrc = itimesn + (size_t)ys * nx + (size_t)xs;
					const size_t idxdest = itimesn + y  * nx + x;
					f[idxdest] = fstar[idxsrc];
                } // boundary conditions are handled in a dedicated function
             }
         }
    }
}
/* ================================================================= */


/* ========================= Boundary Conditions ====================== */

// NB: bc are hard coded, no generality for other domains... just rectangles
void set_boundaryconditions(State *state) {
    check_ptrvalidity(state);

    const size_t nx = state->nx;
    const size_t ny = state->ny;
    const size_t n  = nx * ny;
    scalar_t *f = state->data;

    // bottom wall: y = 0
    const size_t y0 = 0;
	#pragma omp parallel for
    for (size_t x = 0; x < nx; ++x) {
        const size_t j = y0 * nx + x;
        f[2 * n + j] = f[4 * n + j];
        f[5 * n + j] = f[7 * n + j];
        f[6 * n + j] = f[8 * n + j];
    }

    // top wall: noslip
    const size_t ytop = ny - 1;
	#pragma omp parallel for
    for (size_t x = 0; x < nx; ++x) {
        const size_t j = ytop * nx + x;
        f[4 * n + j] = f[2 * n + j];
        f[7 * n + j] = f[5 * n + j];
        f[8 * n + j] = f[6 * n + j];
    }

    /*
    // left wall: prescribed velocity
    const size_t x0 = 0;
    for (size_t y = 1; y < ny - 1; ++y) { // Not on the corners.. we leave no slip
        const size_t j = y * nx + x0;

        const scalar_t f0 = f[0 * n + j];
        const scalar_t f2 = f[2 * n + j];
        const scalar_t f3 = f[3 * n + j];
        const scalar_t f4 = f[4 * n + j];
        const scalar_t f6 = f[6 * n + j];
        const scalar_t f7 = f[7 * n + j];
        const scalar_t rholocal = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - ux_in);

        f[1 * n + j] = f3 + (2.0 / 3.0) * rholocal * ux_in;
        f[5 * n + j] = f7 + 0.5 * (f4 - f2) + (1.0 / 6.0) * rholocal * ux_in + 0.5 * rholocal * uy_in;
        f[8 * n + j] = f6 + 0.5 * (f2 - f4) + (1.0 / 6.0) * rholocal * ux_in - 0.5 * rholocal * uy_in;
    }
    */
    // left wall: noslip
    const size_t x0 = 0;
	//#pragma omp parallel for // Here there is no reason to parallelize unleass it's a very fine mesh
    for (size_t y = 1; y < ny -1; ++y) { // Not on the corners.. we leave no slip
        const size_t j = y * nx + x0;
        f[1 * n + j] = f[3 * n + j];
        f[8 * n + j] = f[6 * n + j];
        f[5 * n + j] = f[7 * n + j];
    }

    // right wall: noslip
    const size_t xright = nx - 1;
	//#pragma omp parallel for // Here there is no reason to parallelize unleass it's a very fine mesh
    for (size_t y = 1; y < ny -1; ++y) { // Not on the corners.. we leave no slip
        const size_t j = y * nx + xright;
        f[3 * n + j] = f[1 * n + j];
        f[6 * n + j] = f[8 * n + j];
        f[7 * n + j] = f[5 * n + j];
    }
}
/* ==================================================================== */


/* =============================================== Main Function ===================================== */
int main (int argc, char **argv) {

	/* ----- Simulation Parameters ----- */
    size_t ntimesteps = 300;
	size_t nx = 200;
	size_t ny = 20;

    // --- if provided in the command line --- //
    if (argc == 2) 
        ntimesteps = parse_intctl(atoi(argv[1]));
    else if (argc == 3) {
        ntimesteps = parse_intctl(atoi(argv[1]));
        nx = parse_intctl(atoi(argv[2]));
    } else if (argc == 4) {
        ntimesteps = parse_intctl(atoi(argv[1]));
        nx = parse_intctl(atoi(argv[2]));
        ny = parse_intctl(atoi(argv[3]));
    }
	// change nu tau omega if you want
    
	mkdir("output", 0755);

	// --- Lattice Units --- //
	const scalar_t rho0 = 1.0;
	const scalar_t dx = 1.0;
	const scalar_t dt = 1.0;
	// --------------------- //
	/* --------------------------------- */

    printf("================================================\n");
    printf(" LATTICE BOLTZMANN SIMULATION (D2Q9)\n");
    printf("================================================\n");
    printf(" Parameters:\n");
    printf("  - Grid:       %zu x %zu\n", nx, ny);
    printf("  - Total Nodes: %zu\n", nx * ny);
    printf("  - Steps:      %zu\n", ntimesteps);
    printf("  - Rho0:       %.4f\n", rho0);
    printf("  - Viscosity:  %.4e\n", nu); // Scientific se è molto piccola
    printf("------------------------------------------------\n");
    printf("----------------------------------------\n\n");
    printf(" Initializing fields... ");
    fflush(stdout);

	/* ------------ Initialize Data ----------- */
    // --- Main Current State --- //
	State *state = create_state(nx, ny);
	set_initialstate_perturbed(state, rho0);
    set_boundaryconditions(state);

    // --- Auxiliary States --- //
    State *statestar = create_state(nx, ny);
    State *equilibrium = create_state(nx, ny);

    // --- Field Variables --- //
    Field *rho = create_scalarfield(nx, ny);
    Field *ux = create_scalarfield(nx, ny);
    Field *uy = create_scalarfield(nx, ny);
	/* --------------------------------------- */

    printf("done.\n\n");

    size_t saveevery = 1; 

    /* ----- Time Evolution ----- */    
    double start_clock, end_clock;
    start_clock = get_time();
    for (size_t t = 0; t < ntimesteps; ++t) {
        compute_macros(state, rho, ux, uy); // computes rho, ux, uy
        compute_equilibrium(equilibrium, rho, ux, uy); // computes equilibrium
        relaxation_step(state, statestar, equilibrium); // computes statestar
        streaming(state, statestar); // computes next state
        set_boundaryconditions(state);

        // verbose print
        if (t % saveevery == 0 || t == ntimesteps - 1) {
            save_macros(rho, ux, uy, "output/solution", t);
            double progress = (100.0 * (t + 1)) / ntimesteps;
            printf("\r Progress: [%3.0f%%]  Step: %zu/%zu", progress, t + 1, ntimesteps);
            fflush(stdout);
        }
    }
    end_clock = get_time();
    /* -------------------------- */    

    save_pvd("output/solution", ntimesteps, saveevery);
    printf("\n------------------------------------------------\n");
    printf(" Simulation completed successfully.\n");
    printf(" Total execution time: %.2f seconds\n", end_clock - start_clock);
    printf(" Results saved. Cleaning memory... ");
    fflush(stdout);

    /* ----- Free memory ----- */
    destroy_state(state);
    destroy_state(statestar);
    destroy_state(equilibrium);
    destroy_field(rho);
    destroy_field(ux);
    destroy_field(uy);
    /* ----------------------- */

    printf("done.\n");
    printf("------------------------------------------------\n");

	return 0;
}
/* =================================================================================================== */



void save_macros(const Field *rho,
                 const Field *ux,
                 const Field *uy,
                 const char *basename,
                 size_t timestep) {
    const size_t nx = rho->nx;
    const size_t ny = rho->ny;

    char filename[512];
    snprintf(filename, sizeof(filename), "%s_%06zu.vti", basename, timestep);

    FILE *fp = fopen(filename, "w");

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <ImageData WholeExtent=\"0 %zu 0 %zu 0 0\" Origin=\"0 0 0\" Spacing=\"1 1 1\">\n",
            nx - 1, ny - 1);
    fprintf(fp, "    <Piece Extent=\"0 %zu 0 %zu 0 0\">\n", nx - 1, ny - 1);
    fprintf(fp, "      <PointData>\n");

    // density
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"density\" format=\"ascii\">\n");
    for (size_t j = 0; j < nx * ny; ++j) {
        fprintf(fp, " %.16e", rho->data[j]);
    }
    fprintf(fp, "\n        </DataArray>\n");

    // pressure
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"pressure\" format=\"ascii\">\n");
    for (size_t j = 0; j < nx * ny; ++j) {
        fprintf(fp, " %.16e", cs2 * rho->data[j]);
    }
    fprintf(fp, "\n        </DataArray>\n");

    // velocity
    fprintf(fp, "        <DataArray type=\"Float64\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for (size_t j = 0; j < nx * ny; ++j) {
        fprintf(fp, " %.16e %.16e %.16e", ux->data[j], uy->data[j], 0.0);
    }
    fprintf(fp, "\n        </DataArray>\n");

    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "      <CellData>\n");
    fprintf(fp, "      </CellData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </ImageData>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
}

void save_pvd(const char *basename, size_t ntimesteps, size_t saveevery) {
    char filename[512];
    snprintf(filename, sizeof(filename), "%s.pvd", basename);

    FILE *fp = fopen(filename, "w");

    fprintf(fp, "<?xml version=\"1.0\"?>\n");
    fprintf(fp, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <Collection>\n");

    for (size_t t = 0; t < ntimesteps; ++t) {
        if (t % saveevery == 0 || t == ntimesteps - 1) {
            fprintf(fp,
                    "    <DataSet timestep=\"%zu\" group=\"\" part=\"0\" file=\"solution_%06zu.vti\"/>\n",
                    t, t);
        }
    }

    fprintf(fp, "  </Collection>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
}
