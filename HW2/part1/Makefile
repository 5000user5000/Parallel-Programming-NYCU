CXX := g++
CXXFLAGS := -O3 -march=native -ffast-math -funroll-loops -flto -Wall -pthread

TARGET := pi.out
SRC := pi.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET) 4 1000000000

pg: $(SRC)
	$(CXX) -pg -o $(TARGET) $<
	./$(TARGET)
	gprof ./$(TARGET) gmon.out -b > profiling_result

perf: $(SRC)
	$(CXX) -g -o $(TARGET) $<
	perf record ./$(TARGET)
	perf report

clean:
	rm -f $(TARGET) *.o gmon.out perf.data profiling_result
