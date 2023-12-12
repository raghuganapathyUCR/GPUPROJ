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
# NVCC=nvcc
# CFLAGS=-std=c99
# CUFLAGS=
# LDFLAGS=-lm

# # Source files
# CU_SOURCES=neural_net_app.cu
# C_SOURCES=base.c neural_net_core.c neural_net_io.c neural_net_propagation.c neural_net_training.c neural_net_utils.c
# CU_OBJS=$(CU_SOURCES:.cu=.o)
# C_OBJS=$(C_SOURCES:.c=.o)

# # Executable name
# EXECUTABLE=bpn

# # Default target
# all: $(EXECUTABLE)

# # Linking
# $(EXECUTABLE): $(C_OBJS) $(CU_OBJS)
# 	$(NVCC) $(C_OBJS) $(CU_OBJS) -o $@ $(LDFLAGS)

# # Compiling C source files
# %.o: %.c
# 	$(NVCC) -Xcompiler $(CFLAGS) -c $< -o $@

# # Compiling CUDA source files
# %.o: %.cu
# 	$(NVCC) $(CUFLAGS) -c $< -o $@

# # Clean up
# clean:
# 	rm -f $(C_OBJS) $(CU_OBJS) $(EXECUTABLE)

#######
# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -g -O2

# CUDA architecture flags
# Can be modified to match the architecture of your GPU (e.g., -arch=sm_35)
ARCH_FLAGS = -arch=sm_30

# Object files
OBJS = $(patsubst %.cu,%.o,$(wildcard *.cu))

# Target
TARGET = neural_net_app

# Rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) $(OBJS) -o $(TARGET)

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH_FLAGS) -c $< -o $@

# Clean
clean:
	rm -f *.o $(TARGET)

.PHONY: all clean



