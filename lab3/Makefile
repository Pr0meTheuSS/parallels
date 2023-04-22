Compiler:=gcc

libs:= -lgsl -lm
keys:= -O3 -Werror -Wextra -Wall -pedantic -fopenmp

SHELL:=/bin/bash

.PHONY: build_seq
build_seq:
	$(Compiler) $(wildcard sequentional/*.c) $(keys) -o sequentional/Exec/app $(libs)

.PHONY: build_prll_1
build_prll_1:
	$(Compiler) $(wildcard parallel_1/*.c) $(keys) -o parallel_1/Exec/app $(libs)

.PHONY: build_prll_2
build_prll_2:
	$(Compiler) $(wildcard parallel_2/*.c) $(keys)  -o parallel_2/Exec/app $(libs)


.PHONY: meas_time_seq
meas_time_seq:
	rm TM_seq
	./sequentional/Exec/app matrixes/matrix_128_128.dat >> TM_seq; \

.PHONY: meas_time_prll_1
meas_time_prll_1:
	rm TM_prll_1
	for m_size in {64..128..8}; do \
		for threads in {1..16}; do \
			export OMP_NUM_THREADS=$$threads; \
			echo 'Run time measurement.' ;\
			echo 'omp num threads set in value:' $$OMP_NUM_THREADS; \
			echo 'matrix linear size:' $$m_size; \
			./parallel_1/Exec/app matrixes/matrix_$${m_size}_$${m_size}.dat >> TM_prll_1; \
			echo 'Finish measurement iteration.' ; \
		done; \
	done; 

.PHONY: meas_time_prll_2
meas_time_prll_2:
	rm TM_prll_2
	for m_size in {64..128..8}; do \
		for threads in {1..16}; do \
			export OMP_NUM_THREADS=$$threads; \
			echo 'Run time measurement.' ;\
			echo 'omp num threads set in value:' $$OMP_NUM_THREADS; \
			echo 'matrix linear size:' $$m_size; \
			./parallel_2/Exec/app matrixes/matrix_$${m_size}_$${m_size}.dat >> TM_prll_2; \
			echo 'Finish measurement iteration.' ; \
		done \
	done
