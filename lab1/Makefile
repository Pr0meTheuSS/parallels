MpiCompiler:=mpecc
Compiler:=gcc

libs:= -lgsl -lm
Wkeys:= -Werror -Wextra -Wall -pedantic

threads:=1

input:=matrix_100_100.dat


.PHONY: build_seq
build_seq:
	$(Compiler) $(wildcard sequentional/*.c) $(Wkeys) -o sequentional/Exec/app $(libs)

.PHONY: build_prll_1
build_prll_1:
	$(MpiCompiler) $(wildcard parallel_1/*.c) $(Wkeys) -mpilog -o parallel_1/Exec/app_1 $(libs)

.PHONY: build_prll_2
build_prll_2:
	$(MpiCompiler) $(wildcard parallel_2/*.c) $(Wkeys) -mpilog -o parallel_2/Exec/app_prll_2 $(libs)

