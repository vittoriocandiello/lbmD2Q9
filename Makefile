CC ?= gcc
CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic
CFLAGS += -fopenmp
LDLIBS += -lm

OBJECTS = build/main.o build/gaussianpulse.o build/lbm.o

solver: $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

build/main.o: src/main.c src/gaussianpulse.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/gaussianpulse.o: src/gaussianpulse.c src/gaussianpulse.h src/lbm.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build/lbm.o: src/lbm.c src/lbm.h | build
	$(CC) $(CFLAGS) -c $< -o $@

build:
	mkdir -p $@

.PHONY: clean
clean:
	rm -rf build solver
	rm -f main.o gaussianpulse.o lbm.o
