NVCC      = nvcc
TARGET    = histogram_eq

all: $(TARGET)

$(TARGET): solution.cu
	$(NVCC) solution.cu -O3 -arch=sm_35 -Xcompiler -fopenmp -o $(TARGET)

clean:
	rm -f $(TARGET) *.txt results.csv

.PHONY: all clean
