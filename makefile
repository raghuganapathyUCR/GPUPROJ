# # Makefile for building bpn project

# CC=gcc-13
# CFLAGS=
# LDFLAGS=-lm
# OBJS=base.o neural_net_app.o neural_net_core.o neural_net_io.o neural_net_propagation.o neural_net_training.o neural_net_utils.o
# EXECUTABLE=bpn

# all: $(EXECUTABLE)

# $(EXECUTABLE): $(OBJS)
# 	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(OBJS) $(LDFLAGS)

# %.o: %.c
# 	$(CC) $(CFLAGS) -c $<

# clean:
# 	rm -f $(OBJS) $(EXECUTABLE)
NVCC=nvcc
CFLAGS=-std=c99
CUFLAGS=
LDFLAGS=-lm
NVCC = nvcc
NVCC_FLAGS = -g -O2
# ARCH_FLAGS = -arch=sm_30  # Adjust this to match your GPU architecture
CUDA_INCLUDE_PATH = /usr/local/cuda/include
CUDA_LIB_PATH = /usr/local/cuda/lib64

# Source files
CU_SOURCES = $(wildcard *.cu)
CU_OBJS = $(CU_SOURCES:.cu=.o)

# Executable name
EXECUTABLE = bpn

# Default target
all: $(EXECUTABLE)

# Linking
$(EXECUTABLE): $(CU_OBJS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -L$(CUDA_LIB_PATH) $(CU_OBJS) -lcurand -o $@

# Compiling CUDA source files
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -I$(CUDA_INCLUDE_PATH) -c $< -o $@

# Clean up
clean:
	rm -f $(CU_OBJS) $(EXECUTABLE)

# NVCC = nvcc
# NVCC_FLAGS = -g -O2 -dc
# LD_FLAGS = -dlink
# # ARCH_FLAGS = -arch=sm_30  # Uncomment and adjust to match your GPU architecture
# CUDA_INCLUDE_PATH = /usr/local/cuda/include
# CUDA_LIB_PATH = /usr/local/cuda/lib64

# # Source files
# CU_SOURCES = $(wildcard *.cu)
# CU_OBJS = $(CU_SOURCES:.cu=.o)

# # Executable name
# EXECUTABLE = bpn

# # Default target
# all: $(EXECUTABLE)

# # Linking
# $(EXECUTABLE): $(CU_OBJS)
# 	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -L$(CUDA_LIB_PATH) $(CU_OBJS) -lcurand -o $@

# # Compiling CUDA source files
# %.o: %.cu
# 	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -I$(CUDA_INCLUDE_PATH) -c $< -o $@

# # Clean up
# clean:
# 	rm -f $(CU_OBJS) $(EXECUTABLE)




