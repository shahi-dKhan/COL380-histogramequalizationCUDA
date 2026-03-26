NVCC    = nvcc
FLAGS   = -O3 -arch=sm_35 -Xcompiler -fopenmp
TARGETS = histogram_eq histogram_eq_notiled

all: $(TARGETS)

histogram_eq: solution.cu
	$(NVCC) $(FLAGS) -DUSE_TILED_KNN=1 solution.cu -o $@

histogram_eq_notiled: solution.cu
	$(NVCC) $(FLAGS) -DUSE_TILED_KNN=0 solution.cu -o $@

clean:
	rm -f $(TARGETS) *.txt results.csv

.PHONY: all clean
