NVCC       = nvcc
# -std=c++17 for structured bindings; adjust -arch for your GPU
# sm_70 = Volta, sm_86 = Ampere (RTX 30xx), sm_89 = Ada (RTX 40xx)
NVCCFLAGS = -std=c++17 -O3 -arch=sm_70 -Xcompiler "-O3 -fopenmp" -ccbin g++LDFLAGS    = -lm -fopenmp

TARGET     = histogram_eq

all: $(TARGET)

$(TARGET): solution.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET) knn.txt approx_knn.txt kmeans.txt

run: $(TARGET)
	./$(TARGET) input.txt

.PHONY: all clean run
