CXX := g++
CXXFLAGS := -O3 -march=x86-64-v3 -mtune=skylake -ffast-math -funroll-loops -flto -DNDEBUG -fno-stack-protector -Wall -pthread

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
	$(CXX) $(CXXFLAGS)  -g -fno-omit-frame-pointer -o $(TARGET) $<
	perf record -g --call-graph dwarf ./$(TARGET) 4 100000000
	perf report

clean:
	rm -f $(TARGET) *.o gmon.out perf.data profiling_result
