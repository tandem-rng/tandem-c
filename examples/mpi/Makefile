CC ?= cc
MPICC ?= mpicc
CFLAGS ?= -std=c11 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wshadow
LIB = ../../tandem.c ../../tandem.h

.PHONY: all check clean

all: serial mpi omp

serial: serial.c common.h $(LIB)
	$(CC) $(CFLAGS) -I../.. -o $@ serial.c ../../tandem.c -lm

mpi: mpi.c common.h $(LIB)
	$(MPICC) $(CFLAGS) -I../.. -o $@ mpi.c ../../tandem.c -lm

omp: omp.c common.h $(LIB)
	$(CC) $(CFLAGS) -fopenmp -I../.. -o $@ omp.c ../../tandem.c -lm

check: all
	./check.sh

clean:
	rm -f serial mpi omp
