all:
	#nvcc GAMP_main.cu GAMP_cpuFuncs.cu GAMP_gpuFuncs.cu GAMP_main_cpy.cu -o executable -lm
	nvcc GAMP_main.cu GAMP_cpuFuncs.cu GAMP_gpuFuncs.cu  -o executable.out -lm
run:
	./executable.out

clean: 
	rm -rf *.o
	rm -rf *.out

