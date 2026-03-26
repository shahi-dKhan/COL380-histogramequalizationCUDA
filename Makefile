NVCC    = nvcc
FLAGS   = -O3 -arch=sm_35 -Xcompiler -fopenmp
TARGETS = histogram_eq histogram_eq_notiled \
          histogram_eq_spatialhash histogram_eq_notiled_spatialhash \
          histogram_eq_unsorted

all: $(TARGETS)

# Tiled KNN  +  grid approx  (default)
histogram_eq: solution.cu
	$(NVCC) $(FLAGS) -DUSE_TILED_KNN=1 -DUSE_SPATIAL_HASH_APPROX=0 solution.cu -o $@

# Non-tiled KNN  +  grid approx
histogram_eq_notiled: solution.cu
	$(NVCC) $(FLAGS) -DUSE_TILED_KNN=0 -DUSE_SPATIAL_HASH_APPROX=0 solution.cu -o $@

# Tiled KNN  +  spatial-hash approx
histogram_eq_spatialhash: solution.cu
	$(NVCC) $(FLAGS) -DUSE_TILED_KNN=1 -DUSE_SPATIAL_HASH_APPROX=1 solution.cu -o $@

# Non-tiled KNN  +  spatial-hash approx
histogram_eq_notiled_spatialhash: solution.cu
	$(NVCC) $(FLAGS) -DUSE_TILED_KNN=0 -DUSE_SPATIAL_HASH_APPROX=1 solution.cu -o $@

# Tiled KNN  +  unsorted array + dynamic CDF
histogram_eq_unsorted: solution.cu
	$(NVCC) $(FLAGS) -DUSE_TILED_KNN=1 -DUSE_SPATIAL_HASH_APPROX=0 -DUSE_UNSORTED_ARRAY=1 solution.cu -o $@

clean:
	rm -f $(TARGETS) *.txt results.csv

.PHONY: all clean
