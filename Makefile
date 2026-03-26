NVCC  = nvcc
FLAGS = -O3 -arch=sm_35 -Xcompiler -fopenmp

all: a2

a2: solution.cu
	$(NVCC) $(FLAGS) solution.cu -o $@

clean:
	rm -f a2 *.txt results.csv

.PHONY: all clean
