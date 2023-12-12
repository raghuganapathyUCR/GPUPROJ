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

# Makefile for building the bpn project with CUDA support

# Compiler settings
# CC=nvcc
# CFLAGS=
# LDFLAGS=-lm

# # Source files
# CU_SOURCES=neural_net_app.cu
# C_SOURCES=base.c neural_net_app.c neural_net_core.c neural_net_io.c neural_net_propagation.c neural_net_training.c neural_net_utils.c
# CU_OBJS=$(CU_SOURCES:.cu=.o)
# C_OBJS=$(C_SOURCES:.c=.o)

# # Executable name
# EXECUTABLE=bpn

# # Default target
# all: $(EXECUTABLE)

# # Linking
# $(EXECUTABLE): $(C_OBJS) $(CU_OBJS)
# 	$(CC) $(CFLAGS) -o $(EXECUTABLE) $(C_OBJS) $(CU_OBJS) $(LDFLAGS)

# # Compiling C source files
# %.o: %.c
# 	$(CC) $(CFLAGS) -x cu -c $<

# # Compiling CUDA source files
# %.o: %.cu
# 	$(CC) $(CFLAGS) -c $<

# # Clean up
# clean:
# 	rm -f $(C_OBJS) $(CU_OBJS) $(EXECUTABLE)


CC=gcc
NVCC=nvcc
CFLAGS=
NVCCFLAGS=
LDFLAGS=-lm

# Source files
CU_SOURCES=neural_net_app.cu
C_SOURCES=base.c neural_net_core.c neural_net_io.c neural_net_propagation.c neural_net_training.c neural_net_utils.c
CU_OBJS=$(CU_SOURCES:.cu=.o)
C_OBJS=$(C_SOURCES:.c=.o)

# Executable name
EXECUTABLE=bpn

# Default target
all: $(EXECUTABLE)

# Linking
$(EXECUTABLE): $(C_OBJS) $(CU_OBJS)
    $(NVCC) $(NVCCFLAGS) -o $(EXECUTABLE) $(C_OBJS) $(CU_OBJS) $(LDFLAGS)

# Compiling C source files
%.o: %.c
    $(CC) $(CFLAGS) -c $<

# Compiling CUDA source files
%.o: %.cu
    $(NVCC) $(NVCCFLAGS) -c $<

# Clean up
clean:
    rm -f $(C_OBJS) $(CU_OBJS) $(EXECUTABLE)
