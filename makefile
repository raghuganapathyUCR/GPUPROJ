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
CFLAGS=
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
	$(NVCC) $(CFLAGS) -o $(EXECUTABLE) $(C_OBJS) $(CU_OBJS) $(LDFLAGS)

# Compiling source files
%.o: %.c
	$(NVCC) $(CFLAGS) -c $<

%.o: %.cu
	$(NVCC) $(CFLAGS) -c $<

# Clean up
clean:
	rm -f $(C_OBJS) $(CU_OBJS) $(EXECUTABLE)


