CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2

SRC = main.cpp finite_difference.cpp
OBJ = $(SRC:.cpp=.o)
EXEC = heat_solver

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJ) $(EXEC)
