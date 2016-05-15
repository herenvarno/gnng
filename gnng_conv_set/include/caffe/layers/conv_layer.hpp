#ifndef CAFFE_CONV_LAYER_HPP_
#define CAFFE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

//#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class ConvolutionLayer : public Layer<Dtype> {
public:
	explicit ConvolutionLayer(const LayerParameter& param)
		: Layer<Dtype>(param) {}
	
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual inline int MinBottomBlobs() const {
		return 1;
	}
	virtual inline int MinTopBlobs() const {
		return 1;
	}
	virtual inline bool EqualNumBottomTopBlobs() const{
		return true;
	}
	virtual inline const char* type() const {
		return "Convolution";
	}

protected:
	void forward_gpu_gemm(const Dtype* col_input, const Dtype* weights,
		Dtype* output,  const int set_size, bool skip_im2col = false);
	void forward_gpu_bias(Dtype* output, const Dtype* bias);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	inline int input_shape(int i) {
		return (*bottom_shape_)[channel_axis_ + i];
	}
	virtual inline bool reverse_dimensions() { return false; }
	virtual void compute_output_shape();
	
	Blob<int> kernel_shape_;
	Blob<int> stride_;
	Blob<int> pad_;
	Blob<int> dilation_;
	Blob<int> conv_input_shape_;
	vector<int> col_buffer_shape_;
	vector<int> output_shape_;
	const vector<int>* bottom_shape_;
	
	int num_spatial_axes_;
	int bottom_dim_;
	int top_dim_;
	
	int channel_axis_;
	int num_;
	int channels_;
	int group_;
	int out_spatial_dim_;
	int weight_offset_;
	int num_output_;
	bool bias_term_;
	bool is_1x1_;
	bool force_nd_im2col_;
	
private:
	inline void conv_im2col_gpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_gpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_gpu(data, num_spatial_axes_, num_kernels_im2col_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(),
          stride_.gpu_data(), dilation_.gpu_data(), col_buff);
    }
  }
  inline void conv_col2im_gpu(const Dtype* col_buff, Dtype* data) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      col2im_gpu(col_buff, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1],
          dilation_.cpu_data()[0], dilation_.cpu_data()[1], data);
    } else {
      col2im_nd_gpu(col_buff, num_spatial_axes_, num_kernels_col2im_,
          conv_input_shape_.gpu_data(), col_buffer_.gpu_shape(),
          kernel_shape_.gpu_data(), pad_.gpu_data(), stride_.gpu_data(),
          dilation_.gpu_data(), data);
    }
  }
  
	int num_kernels_im2col_;
	int num_kernels_col2im_;
	int conv_out_channels_;
	int conv_in_channels_;
	int conv_out_spatial_dim_;
	int kernel_dim_;
	int col_offset_;
	int output_offset_;

	Blob<Dtype> col_buffer_;
	Blob<Dtype> bias_multiplier_;
  
};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
