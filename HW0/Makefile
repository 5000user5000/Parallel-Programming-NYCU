CXX := g++
CXXFLAGS := -O2 -Wall

TARGET := pi.out
SRC := pi.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

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
