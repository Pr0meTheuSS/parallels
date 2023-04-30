Compiler:=mpicc

libs:= -lgsl -lm
keys:= -O0 -g3

SHELL:=/bin/bash

.PHONY: build
build:
	$(Compiler) $(wildcard src/*.c) $(keys) -o src/Exec/app $(libs)

.PHONY: meas_time
meas_time:
	rm TM
	for m_size in {64..128..8}; do \
		for threads in {1..16}; do \
			export OMP_NUM_THREADS=$$threads; \
			echo 'Run time measurement.' ;\
			echo 'omp num threads set in value:' $$OMP_NUM_THREADS; \
			echo 'matrix linear size:' $$m_size; \
			./parallel_1/Exec/app matrixes/matrix_$${m_size}_$${m_size}.dat >> TM; \
			echo 'Finish measurement iteration.' ; \
		done; \
	done; 
