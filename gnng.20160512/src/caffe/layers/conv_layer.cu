#include <vector>
#include "caffe/layers/conv_layer.hpp"

namespace caffe {



template <typename Dtype>
void ConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Configure the kernel size, padding, stride, and inputs.
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  force_nd_im2col_ = conv_param.force_nd_im2col();
  channel_axis_ = bottom[0]->CanonicalAxisIndex(conv_param.axis());
  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = bottom[0]->num_axes();
  num_spatial_axes_ = num_axes - first_spatial_axis;
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1));
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();
  if (conv_param.has_kernel_h() || conv_param.has_kernel_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "kernel_h & kernel_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.kernel_size_size())
        << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = conv_param.kernel_h();
    kernel_shape_data[1] = conv_param.kernel_w();
  } else {
    const int num_kernel_dims = conv_param.kernel_size_size();
    CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
        << "kernel_size must be specified once, or once per spatial dimension "
        << "(kernel_size specified " << num_kernel_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
      for (int i = 0; i < num_spatial_axes_; ++i) {
        kernel_shape_data[i] =
            conv_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
      }
  }
  for (int i = 0; i < num_spatial_axes_; ++i) {
    CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  }
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (conv_param.has_stride_h() || conv_param.has_stride_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "stride_h & stride_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.stride_size())
        << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
  } else {
    const int num_stride_dims = conv_param.stride_size();
    CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
          num_stride_dims == num_spatial_axes_)
        << "stride must be specified once, or once per spatial dimension "
        << "(stride specified " << num_stride_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          conv_param.stride((num_stride_dims == 1) ? 0 : i);
      CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (conv_param.has_pad_h() || conv_param.has_pad_w()) {
    CHECK_EQ(num_spatial_axes_, 2)
        << "pad_h & pad_w can only be used for 2D convolution.";
    CHECK_EQ(0, conv_param.pad_size())
        << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
  } else {
    const int num_pad_dims = conv_param.pad_size();
    CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
          num_pad_dims == num_spatial_axes_)
        << "pad must be specified once, or once per spatial dimension "
        << "(pad specified " << num_pad_dims << " times; "
        << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          conv_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = conv_param.dilation_size();
  CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
        num_dilation_dims == num_spatial_axes_)
      << "dilation must be specified once, or once per spatial dimension "
      << "(dilation specified " << num_dilation_dims << " times; "
      << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       conv_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }
  // Configure output channels and groups.
  channels_ = bottom[0]->shape(channel_axis_);
  num_output_ = this->layer_param_.convolution_param().num_output();
  CHECK_GT(num_output_, 0);
  group_ = this->layer_param_.convolution_param().group();
  CHECK_EQ(channels_ % group_, 0);
  CHECK_EQ(num_output_ % group_, 0)
      << "Number of output should be multiples of group.";
  if (reverse_dimensions()) {
    conv_out_channels_ = channels_;
    conv_in_channels_ = num_output_;
  } else {
    conv_out_channels_ = num_output_;
    conv_in_channels_ = channels_;
  }
  // Handle the parameters: weights and biases.
  // - blobs_[0] holds the filter weights
  // - blobs_[1] holds the biases (optional)
  vector<int> weight_shape(2);
  weight_shape[0] = conv_out_channels_;
  weight_shape[1] = conv_in_channels_ / group_;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    weight_shape.push_back(kernel_shape_data[i]);
  }
  bias_term_ = this->layer_param_.convolution_param().bias_term();
  vector<int> bias_shape(bias_term_, num_output_);
  if (this->blobs_.size() > 0) {
    CHECK_EQ(1 + bias_term_, this->blobs_.size())
        << "Incorrect number of weight blobs.";
    if (weight_shape != this->blobs_[0]->shape()) {
      Blob<Dtype> weight_shaped_blob(weight_shape);
      LOG(FATAL) << "Incorrect weight shape: expected shape "
          << weight_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[0]->shape_string();
    }
    if (bias_term_ && bias_shape != this->blobs_[1]->shape()) {
      Blob<Dtype> bias_shaped_blob(bias_shape);
      LOG(FATAL) << "Incorrect bias shape: expected shape "
          << bias_shaped_blob.shape_string() << "; instead, shape was "
          << this->blobs_[1]->shape_string();
    }
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  kernel_dim_ = this->blobs_[0]->count(1);
  weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int first_spatial_axis = channel_axis_ + 1;
  CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_)
      << "bottom num_axes may not change.";
  num_ = bottom[0]->count(0, channel_axis_);
  CHECK_EQ(bottom[0]->shape(channel_axis_), channels_)
      << "Input size incompatible with convolution kernel.";
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK(bottom[0]->shape() == bottom[bottom_id]->shape())
        << "All inputs must have the same shape.";
  }
  // Shape the tops.
  bottom_shape_ = &bottom[0]->shape();
  compute_output_shape();
  vector<int> top_shape(bottom[0]->shape().begin(),
      bottom[0]->shape().begin() + channel_axis_);
  top_shape.push_back(num_output_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    top_shape.push_back(output_shape_[i]);
  }
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(top_shape);
  }
  if (reverse_dimensions()) {
    conv_out_spatial_dim_ = bottom[0]->count(first_spatial_axis);
  } else {
    conv_out_spatial_dim_ = top[0]->count(first_spatial_axis);
  }
  col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
  output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
  // Setup input dimensions (conv_input_shape_).
  vector<int> bottom_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(bottom_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    if (reverse_dimensions()) {
      conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
    } else {
      conv_input_shape_data[i] = bottom[0]->shape(channel_axis_ + i);
    }
  }
  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(kernel_dim_ * group_);
  for (int i = 0; i < num_spatial_axes_; ++i) {
    if (reverse_dimensions()) {
      col_buffer_shape_.push_back(input_shape(i + 1));
    } else {
      col_buffer_shape_.push_back(output_shape_[i]);
    }
  }
  col_buffer_.Reshape(col_buffer_shape_);
  bottom_dim_ = bottom[0]->count(channel_axis_);
  top_dim_ = top[0]->count(channel_axis_);
  num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
  num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_;
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  out_spatial_dim_ = top[0]->count(first_spatial_axis);
  if (bias_term_) {
    vector<int> bias_multiplier_shape(1, out_spatial_dim_);
    bias_multiplier_.Reshape(bias_multiplier_shape);
    caffe_set(bias_multiplier_.count(), Dtype(1),
        bias_multiplier_.mutable_cpu_data());
  }
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::forward_gpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
    }
    col_buff = col_buffer_.gpu_data();
  }
  
  for (int g = 0; g < group_; ++g) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::forward_gpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}


/*
 * THE KERNEL TO DO ALL THE CONV OPERATION
 */
template <typename Dtype>
__global__ void conv_do_gpu_kernel(const int n,
		const Dtype* I, const int I_cs, const int I_h, const int I_w,
    	const int kernel_h, const int kernel_w,
    	const int pad_h, const int pad_w,
    	const int stride_h, const int stride_w,
    	const int dilation_h, const int dilation_w,
    	const Dtype* W, const int W_h, const int W_w,
    	const Dtype* B, const int B_h, const int B_w,
    	Dtype* O, const int O_h, const int O_w,
    	Dtype* col_buffer) {
	CUDA_KERNEL_LOOP(index, n) {
		Dtype out=0;
		const int width_col = (I_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	  	const int O_x = (index/W_h) / O_w;
	  	const int O_y = (index/W_h) % O_w;
	  	const int W_y = O_y;
	  	const int kernel_dim = kernel_w*kernel_h;
//		for(int i=0; i<W_h; i++){
//		for(int i=0; i<1; i++){
		const int i = index % W_h;
		const int W_x = i;
		const int I_x = ((O_x/width_col)-1)*stride_h + (dilation_h * (kernel_h - 1 ) + 1) - 2*pad_h + (i%kernel_dim)/kernel_w - kernel_h +1;
		const int I_y = ((O_x%width_col)-1)*stride_w + (dilation_w * (kernel_w - 1 ) + 1) - 2*pad_w + (i%kernel_dim)%kernel_w - kernel_w +1;
		//const int I_x = 1;
		//const int I_y = 1;
		const int I_c = i / kernel_dim;
			if(I_x<I_h && I_y<I_w && I_c<I_cs){
				//out = out + 1;
				out = (I[I_c*+I_h*I_w+I_x*I_w+I_y] * W[W_y*W_h+W_x]);
			}
//		}
		if(B && i==0){
			out = out + B[O_y];
		}
//		if(index==0){
		O[O_y*O_h+O_x]+=out;
//	}
	}
}

/* JUST A BACKUP
template <typename Dtype>
__global__ void conv_do_gpu_kernel(const int n,
		const Dtype* I, const int I_cs, const int I_h, const int I_w,
    	const int kernel_h, const int kernel_w,
    	const int pad_h, const int pad_w,
    	const int stride_h, const int stride_w,
    	const int dilation_h, const int dilation_w,
    	const Dtype* W, const int W_h, const int W_w,
    	const Dtype* B, const int B_h, const int B_w,
    	Dtype* O, const int O_h, const int O_w,
    	Dtype* col_buffer) {
	CUDA_KERNEL_LOOP(index, n) {
		Dtype out=0;
		
		const int width_col = (I_w + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	
		const int idx = index / W_h;
	  	const int O_x = idx / O_w;
	  	const int O_y = idx % O_w;
	  	const int W_y = O_y;
	  	const int T_x = O_x;
			int T_y = index % W_h;
			int W_x = index % W_h;
			int I_x = ((T_x/width_col)-1)*stride_h + (dilation_h * (kernel_h - 1 ) + 1) - 2*pad_h + (T_y%(kernel_w*kernel_h))/kernel_w - kernel_h +1;
			int I_y = ((T_x%width_col)-1)*stride_w + (dilation_w * (kernel_w - 1 ) + 1) - 2*pad_w + (T_y%(kernel_w*kernel_h))%kernel_w - kernel_w +1;
	  		int I_c = T_y / (kernel_h * kernel_w);
			if(I_x<I_h && I_y<I_w && I_c<I_cs){
				out = (I[I_c*+I_h*I_w+I_x*I_w+I_y] * W[W_y*W_h+W_x]);
			}
		if(B && index<B_w){
			out = out + B[index];
		}
//		O[O_y*O_h+O_x]=out;
	}
}
*/

template <typename Dtype>
void ConvolutionLayer<Dtype>::forward_gpu_do(const Dtype* input,
    const Dtype* weights, const Dtype* bias, Dtype* output) {
  if (!is_1x1_) {
//		int height = conv_input_shape_.cpu_data()[1];
//		int width = conv_input_shape_.cpu_data()[2];
//		int kernel_h = kernel_shape_.cpu_data()[0];
//		int kernel_w = kernel_shape_.cpu_data()[1];
//		int pad_h = pad_.cpu_data()[0];
//		int pad_w = pad_.cpu_data()[1];
//		int stride_h = stride_.cpu_data()[0];
//		int stride_w = stride_.cpu_data()[1];
//		int dilation_h = dilation_.cpu_data()[0];
//		int dilation_w = dilation_.cpu_data()[1];
//		int num_output = num_output_;
		int channels = conv_in_channels_ * kernel_shape_.cpu_data()[0] * kernel_shape_.cpu_data()[1] ;
	  // We are going to launch channels * height_col * width_col kernels, each
	  // kernel responsible for copying a single-channel grid.
		int height_col = (conv_input_shape_.cpu_data()[1] + 2 * pad_.cpu_data()[0] -
		  (dilation_.cpu_data()[0] * (kernel_shape_.cpu_data()[0] - 1) + 1)) / stride_.cpu_data()[0] + 1;
		int width_col = (conv_input_shape_.cpu_data()[2] + 2 * pad_.cpu_data()[1] -
		  (dilation_.cpu_data()[1] * (kernel_shape_.cpu_data()[1] - 1) + 1)) / stride_.cpu_data()[1] + 1;
		int num_kernels = channels* height_col * width_col * num_output_;
		
		
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  conv_do_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
	                         CAFFE_CUDA_NUM_THREADS>>>(
		  num_kernels, input, conv_in_channels_, conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2], kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1], pad_.cpu_data()[0],
		  pad_.cpu_data()[1], stride_.cpu_data()[0], stride_.cpu_data()[1], dilation_.cpu_data()[0], dilation_.cpu_data()[1], weights, channels, num_output_, bias, height_col*width_col, num_output_, output, height_col * width_col, num_output_, col_buffer_.mutable_gpu_data());
	  CUDA_POST_KERNEL_CHECK;
  }
}



template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();

  for (int i = 0; i < bottom.size(); ++i) {	
     const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    
   for (int n = 0; n < this->num_; ++n) {	// NOTE: For each one in mini batch
   
   /*
    * HERE IS CAFFE'S CODE
    */
 /*        this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
      }*/
    /*
     * HERE IS MY CODE
     */
    if(this->bias_term_){
     	const Dtype* bias = this->blobs_[1]->gpu_data();
      	this->forward_gpu_do(bottom_data + n * this->bottom_dim_, weight, bias, top_data + n * this->top_dim_);
      }else{
      	this->forward_gpu_do(bottom_data + n * this->bottom_dim_, weight, NULL, top_data + n * this->top_dim_);
      }
      cudaDeviceSynchronize();
    }
  }
}

INSTANTIATE_CLASS(ConvolutionLayer);
REGISTER_LAYER_CLASS(Convolution);

}  // namespace caffe
