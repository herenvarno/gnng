#include "test.hpp"
#include "caffe/util/io.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

using namespace caffe;
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
 
  if (argc < 4) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto image ";
    return 1;
  }
/*
  float m[10][12];
  float n[10];
  float w[12]={0};
  for (int i=0; i<10; i++)
  {
  	 for (int j=0; j<12; j++)
  	 {
  	 	m[i][j]=float(i*10+j+1);
  	 }
  }
  for (int x=0; x<10; x++)
  {
  	 for (int y=0; y<12; y++)
  	 {
  	 	cout << m[x][y] << ", ";
  	 }
  	 cout << endl;
  }
  cout << "-------------------------------------"<< endl;
  for (int i=0; i<12; i++)
  {
  	 n[i]=(float)(i*3);
  }
  for (int x=0; x<10; x++)
  {
  	 	cout << n[x] << ", ";
  }
  cout << endl;
  cout << "-------------------------------------"<< endl;
  const float *ptr_m=(float*)m;
  const float *ptr_n=(float*)n;
  float *ptr_w=(float*)w;
	caffe_cpu_gemv<float>(CblasTrans, 10, 12, 1., ptr_m, ptr_n, 0., ptr_w);
   for (int i=0; i<12; i++)
  {
  	 cout << w[i] << ", ";
  }
  cout << endl;
 */
 /*   
  float n[4][9][9]={0};
  float *ptr_m=(float*)m;
  float *ptr_n=(float*)n;
  
 int height=10;
 int width=10;
 int kernel_h=2;
 int kernel_w=2;
 int pad_h=0;
 int pad_w=0;
 int stride_h=1;
 int stride_w=1;
 int dilation_h=1;
 int dilation_w=1;
    
    int output_h1 = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  int output_w1 = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    
	cout << output_h1 << ", " << output_w1 << endl;
	im2col_cpu(ptr_m, 1, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, ptr_n);
	cout << output_h1 << ", " << output_w1 << endl;
	for (int x=0; x<output_h1; x++)
  {
  	 for (int y=0; y<output_w1; y++)
  	 {
  	 	cout << n[1][x][y] << ", ";
  	 }
  	 cout << endl;
  }
  */
 
  Caffe::set_mode(Caffe::GPU);
   Mat image;
//   transpose(imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE), image) ;  // Read the file
	image=imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
	image/=255.0;
    
  //get the net
  Net<float> caffe_test_net(argv[1], TEST);
  //get trained net
  caffe_test_net.CopyTrainedLayersFrom(argv[2]);
	float loss = 0.0;
	vector<cv::Mat> dv;
	dv.push_back(image); // image is a cv::Mat, as I'm using #1416
	vector<int> dvl;
	dvl.push_back(0);
	boost::dynamic_pointer_cast<MemoryDataLayer<float> >(caffe_test_net.layers()[0])->AddMatVector(dv,dvl);
	vector<Blob<float>* > result;
	clock_t tStart = clock();
	for(int w=0;w<100;w++){
		result = caffe_test_net.Forward(&loss);
	}
	printf("Time taken: %.2f ms\n", (double)(clock() - tStart)*1000/CLOCKS_PER_SEC);

  //Here I can use the argmax layer, but for now I do a simple for :)
  float max = 0;
  float max_i = 0;
  
  for(int i=0; i<result.size(); i++){
  	LOG(INFO) << "SIZE"<<i<<" : " << result[i]->shape_string();
	}
  
  for (int i = 0; i < 10; ++i) {
//    float value = result[2]->cpu_data()[i];
		float value = result[1]->cpu_data()[i];
 //   LOG(INFO) << "["<<i<<"] : " << result[2]->cpu_data()[i] << ", " << result[3]->cpu_data()[i];
		LOG(INFO) << "["<<i<<"] : " << result[1]->cpu_data()[i];
    if (max < value){
      max = value;
      max_i = i;
    }
  }
  LOG(INFO) << "max: " << max << " i " << max_i;

  return 0;
}
