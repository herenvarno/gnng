#ifndef CAFFE_LSTM_LAYER_HPP_
#define CAFFE_LSTM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
template <typename Dtype>
void caffe_bound(const int n, const Dtype* a, const Dtype min, 
    const Dtype max, Dtype* y);

	template <typename Dtype>
	class LstmLayer : public Layer<Dtype> {
	 public:
	  explicit LstmLayer(const LayerParameter& param)
		  : Layer<Dtype>(param) {}
	  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
	  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);

	  virtual inline const char* type() const { return "Lstm"; }
	  virtual bool IsRecurrent() const { return true; }

	 protected:
	  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
	  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		  const vector<Blob<Dtype>*>& top);
	  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	  int I_; // input dimension
	  int H_; // num of hidden units
	  int T_; // length of sequence
	  int N_; // batch size
	  
	  Dtype clipping_threshold_; // threshold for clipped gradient
	  Blob<Dtype> bias_multiplier_;

	  Blob<Dtype> top_;       // output values
	  Blob<Dtype> cell_;      // memory cell
	  Blob<Dtype> pre_gate_;  // gate values before nonlinearity
	  Blob<Dtype> gate_;      // gate values after nonlinearity

	  Blob<Dtype> c_0_; // previous cell state value
	  Blob<Dtype> h_0_; // previous hidden activation value
	  Blob<Dtype> c_T_; // next cell state value
	  Blob<Dtype> h_T_; // next hidden activation value

	  // intermediate values
	  Blob<Dtype> h_to_gate_;
	  Blob<Dtype> h_to_h_;
	};

}

#endif // CAFFE_LSTM_LAYER_HPP_
