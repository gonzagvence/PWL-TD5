CC = g++
CFLAGS = -std=c++17
SRC = main.cpp
OBJ = $(SRC:.cpp=.o)
EXEC = pwl_fit

all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(OBJ) -o $(EXEC)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(EXEC)