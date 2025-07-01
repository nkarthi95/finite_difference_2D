# compiler
CXX = mpic++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2 -fopenmp

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
