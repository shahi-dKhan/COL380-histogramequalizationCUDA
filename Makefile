NVCC      = nvcc
CFLAGS    = -O3 -arch=sm_35 -Xcompiler -fopenmp

all: histogram_eq histogram_eq_v2

histogram_eq: solution.cu
	$(NVCC) solution.cu $(CFLAGS) -o histogram_eq

histogram_eq_v2: solution_v2.cu
	$(NVCC) solution_v2.cu $(CFLAGS) -o histogram_eq_v2

clean:
	rm -f histogram_eq histogram_eq_v2 knn.txt approx_knn.txt kmeans.txt knn_v2.txt approx_knn_v2.txt kmeans_v2.txt

.PHONY: all clean
