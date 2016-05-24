CXX = g++
I_DIR =  ../caffe/include /opt/cuda/include ../caffe/.build_release/src
L_DIR =  ../caffe/.build_release/lib /opt/cuda/lib64
CXXFLAGS = $(patsubst %, -I%, $(I_DIR))
LIBS = $(patsubst %, -L%, $(L_DIR)) -lcaffe -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem -lm -lhdf5_hl -lhdf5 -lboost_thread -lstdc++ -lopenblas `pkg-config --cflags --libs opencv`

BIN = test
SRC = test.cpp

$(BIN): $(SRC) ../caffe/.build_release/lib/libcaffe.so
	$(CXX) $(CXXFLAGS) $(LIBS) $(SRC) -o $@

.PHONY: clean 

clean:
	rm -rf $(BIN)

