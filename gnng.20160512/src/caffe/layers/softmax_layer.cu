#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		  for (int i=0; i<bottom.size(); i++){
	  softmax_axis_ =
		  bottom[i]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
	  top[i]->ReshapeLike(*bottom[i]);
	  vector<int> mult_dims(1, bottom[i]->shape(softmax_axis_));
	  sum_multiplier_.Reshape(mult_dims);
	  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
	  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
	  outer_num_ = bottom[i]->count(i, softmax_axis_);
	  inner_num_ = bottom[i]->count(softmax_axis_ + 1);
	  vector<int> scale_dims = bottom[i]->shape();
	  scale_dims[softmax_axis_] = 1;
	  scale_.Reshape(scale_dims);
	}
}



template <typename Dtype>
__global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template <typename Dtype>
__global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_max, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_exp(const int count, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}

template <typename Dtype>
__global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template <typename Dtype>
__global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const Dtype* channel_sum, Dtype* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template <typename Dtype>
__global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const Dtype* data_1, const Dtype* data_2,
    Dtype* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    for(int i=0;i<bottom.size();i++){
	  const Dtype* bottom_data = bottom[i]->gpu_data();
	  
	  Dtype* top_data = top[i]->mutable_gpu_data();
	  Dtype* scale_data = scale_.mutable_gpu_data();
	  int count = bottom[i]->count();
	  int channels = top[i]->shape(softmax_axis_);
	  caffe_copy(count, bottom_data, top_data);
	  // We need to subtract the max to avoid numerical issues, compute the exp,
	  // and then normalize.
	  // compute max
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  kernel_channel_max<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
		  CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
		  scale_data);
	  // subtract
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  kernel_channel_subtract<Dtype><<<CAFFE_GET_BLOCKS(count),
		  CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
		  scale_data, top_data);
	  // exponentiate
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  kernel_exp<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
		  count, top_data, top_data);
	  // sum after exp
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  kernel_channel_sum<Dtype><<<CAFFE_GET_BLOCKS(outer_num_ * inner_num_),
		  CAFFE_CUDA_NUM_THREADS>>>(outer_num_, channels, inner_num_, top_data,
		  scale_data);
	  // divide
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  kernel_channel_div<Dtype><<<CAFFE_GET_BLOCKS(count),
		  CAFFE_CUDA_NUM_THREADS>>>(count, outer_num_, channels, inner_num_,
		  scale_data, top_data);
	}
}

INSTANTIATE_CLASS(SoftmaxLayer);
REGISTER_LAYER_CLASS(Softmax); 


}  // namespace caffe
