CUDA_DIR = /opt/cuda
PROTOBUF_DIR = /
CXX = g++
NVCC = $(CUDA_DIR)/bin/nvcc
PROTOC = protoc

I_DIR =  /usr/include $(CUDA_DIR)/include include
L_DIR =  /usr/lib /usr/lib64 $(CUDA_DIR)/lib64
CXXFLAGS = $(patsubst %, -I%, $(I_DIR))
LIBS = $(patsubst %, -L%, $(L_DIR)) -lcudart -lcublas -lcurand -lglog -lgflags -lprotobuf -lboost_system -lboost_filesystem -lm -lhdf5_hl -lhdf5 -lboost_thread -lstdc++ -lopenblas `pkg-config --cflags --libs opencv`


BIN = test
BUILD_DIR = obj
CXXSRCS = $(shell find src/ -type f -name '*.cpp')
CXXOBJS = $(patsubst src/%.cpp, obj/%.o, $(CXXSRCS))
CUSRCS = $(shell find src/ -type f -name '*.cu')
CUOBJS = $(patsubst src/%.cu, obj/%.cu.o, $(CUSRCS))
PROTOSRCS = $(shell find src/ -type f -name '*.proto')
PROTOOS= $(patsubst src/%.proto, obj/%.pb.o, $(PROTOSRCS))

$(BIN) : $(CXXOBJS) $(CUOBJS) $(PROTOOS)
	$(CXX) $(LIBS) $(CXXOBJS) $(CUOBJS) $(PROTOOS) -o $@

include/caffe/proto/caffe.pb.cc include/caffe/proto/caffe.pb.h: src/caffe/proto/caffe.proto
	$(PROTOC) --proto_path=$(dir $(PROTOSRCS)) --cpp_out=$(dir $(PROTOHS)) $<

obj/caffe/proto/caffe.pb.o : include/caffe/proto/caffe.pb.cc include/caffe/proto/caffe.pb.h
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c include/caffe/proto/caffe.pb.cc -o $@ 

obj/caffe/util/math_functions.o : src/caffe/util/math_functions.cpp include/caffe/util/math_functions.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/util/math_functions.cpp -o $@ 

obj/caffe/util/math_functions.cu.o : src/caffe/util/math_functions.cu include/caffe/util/math_functions.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/util/math_functions.cu -o $@

obj/caffe/util/im2col.o : src/caffe/util/im2col.cpp include/caffe/util/im2col.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/util/im2col.cpp -o $@ 

obj/caffe/util/im2col.cu.o : src/caffe/util/im2col.cu include/caffe/util/im2col.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/util/im2col.cu -o $@

obj/caffe/util/upgrade_proto.o : src/caffe/util/upgrade_proto.cpp include/caffe/util/upgrade_proto.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/util/upgrade_proto.cpp -o $@ 

obj/caffe/util/io.o : src/caffe/util/io.cpp include/caffe/util/io.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/util/io.cpp -o $@ 

obj/caffe/util/insert_splits.o : src/caffe/util/insert_splits.cpp include/caffe/util/insert_splits.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/util/insert_splits.cpp -o $@ 

obj/caffe/util/blocking_queue.o : src/caffe/util/blocking_queue.cpp include/caffe/util/blocking_queue.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/util/blocking_queue.cpp -o $@ 

obj/caffe/syncedmem.o : src/caffe/syncedmem.cpp include/caffe/syncedmem.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/syncedmem.cpp -o $@

obj/caffe/common.o : src/caffe/common.cpp include/caffe/common.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/common.cpp -o $@

obj/caffe/blob.o : src/caffe/blob.cpp include/caffe/blob.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/blob.cpp -o $@

obj/caffe/layer.o : src/caffe/layer.cpp include/caffe/layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layer.cpp -o $@

obj/caffe/layer_factory.o : src/caffe/layer_factory.cpp include/caffe/layer_factory.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layer_factory.cpp -o $@

obj/caffe/layers/base_conv_layer.o : src/caffe/layers/base_conv_layer.cpp include/caffe/layers/base_conv_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/base_conv_layer.cpp -o $@

obj/caffe/layers/conv_layer.o : src/caffe/layers/conv_layer.cpp include/caffe/layers/conv_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/conv_layer.cpp -o $@

obj/caffe/layers/conv_layer.cu.o : src/caffe/layers/conv_layer.cu include/caffe/layers/conv_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/conv_layer.cu -o $@

obj/caffe/layers/eltwise_layer.o : src/caffe/layers/eltwise_layer.cpp include/caffe/layers/eltwise_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/eltwise_layer.cpp -o $@

obj/caffe/layers/eltwise_layer.cu.o : src/caffe/layers/eltwise_layer.cu include/caffe/layers/eltwise_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/eltwise_layer.cu -o $@

obj/caffe/layers/lrn_layer.o : src/caffe/layers/lrn_layer.cpp include/caffe/layers/lrn_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/lrn_layer.cpp -o $@

obj/caffe/layers/lrn_layer.cu.o : src/caffe/layers/lrn_layer.cu include/caffe/layers/lrn_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/lrn_layer.cu -o $@

obj/caffe/layers/neuron_layer.o : src/caffe/layers/neuron_layer.cpp include/caffe/layers/neuron_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/neuron_layer.cpp -o $@
	

obj/caffe/layers/power_layer.o : src/caffe/layers/power_layer.cpp include/caffe/layers/power_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/power_layer.cpp -o $@

obj/caffe/layers/power_layer.cu.o : src/caffe/layers/power_layer.cu include/caffe/layers/power_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/power_layer.cu -o $@

obj/caffe/layers/pooling_layer.o : src/caffe/layers/pooling_layer.cpp include/caffe/layers/pooling_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/pooling_layer.cpp -o $@

obj/caffe/layers/pooling_layer.cu.o : src/caffe/layers/pooling_layer.cu include/caffe/layers/pooling_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/pooling_layer.cu -o $@

obj/caffe/layers/relu_layer.o : src/caffe/layers/relu_layer.cpp include/caffe/layers/relu_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/relu_layer.cpp -o $@

obj/caffe/layers/relu_layer.cu.o : src/caffe/layers/relu_layer.cu include/caffe/layers/relu_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/relu_layer.cu -o $@

obj/caffe/layers/sigmoid_layer.o : src/caffe/layers/sigmoid_layer.cpp include/caffe/layers/sigmoid_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/sigmoid_layer.cpp -o $@

obj/caffe/layers/sigmoid_layer.cu.o : src/caffe/layers/sigmoid_layer.cu include/caffe/layers/sigmoid_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/sigmoid_layer.cu -o $@

obj/caffe/layers/softmax_layer.o : src/caffe/layers/softmax_layer.cpp include/caffe/layers/softmax_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/softmax_layer.cpp -o $@

obj/caffe/layers/softmax_layer.cu.o : src/caffe/layers/softmax_layer.cu include/caffe/layers/softmax_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/softmax_layer.cu -o $@

obj/caffe/layers/tanh_layer.o : src/caffe/layers/tanh_layer.cpp include/caffe/layers/tanh_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/tanh_layer.cpp -o $@

obj/caffe/layers/tanh_layer.cu.o : src/caffe/layers/tanh_layer.cu include/caffe/layers/tanh_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/tanh_layer.cu -o $@

obj/caffe/layers/split_layer.o : src/caffe/layers/split_layer.cpp include/caffe/layers/split_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/split_layer.cpp -o $@

obj/caffe/layers/split_layer.cu.o : src/caffe/layers/split_layer.cu include/caffe/layers/split_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/split_layer.cu -o $@

obj/caffe/layers/memory_data_layer.cu.o : src/caffe/layers/memory_data_layer.cu include/caffe/layers/memory_data_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/memory_data_layer.cu -o $@


obj/caffe/layers/inner_product_layer.o : src/caffe/layers/inner_product_layer.cpp include/caffe/layers/inner_product_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/inner_product_layer.cpp -o $@

obj/caffe/layers/inner_product_layer.cu.o : src/caffe/layers/inner_product_layer.cu include/caffe/layers/inner_product_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/inner_product_layer.cu -o $@


obj/caffe/layers/base_data_layer.o : src/caffe/layers/base_data_layer.cpp include/caffe/layers/base_data_layer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/layers/base_data_layer.cpp -o $@

obj/caffe/layers/base_data_layer.cu.o : src/caffe/layers/base_data_layer.cu include/caffe/layers/base_data_layer.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/caffe/layers/base_data_layer.cu -o $@

obj/caffe/data_transformer.o : src/caffe/data_transformer.cpp include/caffe/data_transformer.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/data_transformer.cpp -o $@

obj/caffe/internal_thread.o : src/caffe/data_transformer.cpp include/caffe/internal_thread.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/internal_thread.cpp -o $@

obj/caffe/net.o : src/caffe/net.cpp include/caffe/net.hpp
	mkdir -p $(@D) 
	$(CXX) $(CXXFLAGS) -c src/caffe/net.cpp -o $@

obj/test.cu.o : src/test.cu include/test.hpp
	mkdir -p $(@D) 
	$(NVCC) $(CXXFLAGS) -c src/test.cu -o $@

.PHONY: clean 

clean:
	rm -rf $(BIN) $(BUILD_DIR)

