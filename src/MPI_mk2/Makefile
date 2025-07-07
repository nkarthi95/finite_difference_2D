# compiler
CXX = mpic++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2 -fopenmp

# flags
USE_SENDV_1D ?= 0
ifeq ($(USE_SENDV_1D),1)
    CXXFLAGS += -DUSE_SENDV_1D
endif

USE_OMP ?= 1
ifeq ($(USE_OMP),1)
    CXXFLAGS += -DUSE_OMP
endif

USE_OLDMPI ?= 0
ifeq ($(USE_OLDMPI),1)
    CXXFLAGS += -DUSE_OLDMPI
endif

# source files
SRC = main.cpp finite_difference.cpp
OBJ = $(SRC:.cpp=.o)

# output executable
EXEC = heat_solver
all: $(EXEC)

# build target
$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

# object file rule
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

# clean rule
clean:
	rm -f $(OBJ) $(EXEC)
