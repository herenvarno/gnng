#include "test.hpp"
#include "caffe/util/io.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"
#include <pthread.h>
#include <cuda.h>

using namespace caffe;
using namespace std;
using namespace cv;

pthread_mutex_t mutex1=PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2=PTHREAD_MUTEX_INITIALIZER;
int img_count=5;

void *thread_1_callback(void *ptr){
	Caffe::set_mode(Caffe::GPU);
	Net<float> *caffe_test_net=(Net<float>*)(ptr);
	for(int i=0;i<img_count;i++){
		pthread_mutex_lock(&mutex1);
		Caffe::SetDevice(0);
		caffe_test_net->ForwardFromTo(0, 4);
		pthread_mutex_unlock(&mutex2);
	}
	pthread_exit(NULL);
}
void *thread_2_callback(void *ptr){
	Caffe::set_mode(Caffe::GPU);
	Net<float> *caffe_test_net=(Net<float>*)(ptr);
	for(int i=0;i<img_count;i++){
		pthread_mutex_lock(&mutex2);
		Caffe::SetDevice(1);
		cudaDeviceEnablePeerAccess(0, 0 );
		pthread_mutex_unlock(&mutex1);
		caffe_test_net->ForwardFromTo(5, 8);
	}
	pthread_exit(NULL);
}




int main(int argc, char** argv) {

	int accessible = 0;
	cudaDeviceCanAccessPeer( &accessible, 1, 0);

  if (argc < 4) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto image ";
    return 1;
  }
 /* 
  float m[10][10];
  for (int i=0; i<10; i++)
  {
  	 for (int j=0; j<10; j++)
  	 {
  	 	m[i][j]=i*10+j+1;
  	 }
  }
  
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
	
	clock_t tStart = clock();
	pthread_mutex_lock(&mutex2);
	pthread_t thread1, thread2;
	int ret = 0;
	ret |= pthread_create( &thread1, NULL, thread_1_callback, (void*)&caffe_test_net);
	ret |= pthread_create( &thread2, NULL, thread_2_callback, (void*)&caffe_test_net);
	if(ret){
		LOG(FATAL) << "ERROR WHILE CREATING THREAD!";
	}
	pthread_join(thread1, NULL);
	pthread_join(thread2, NULL);
	printf("Time taken: %.2f ms\n", (double)(clock() - tStart)*1000/CLOCKS_PER_SEC);
	tStart = clock();
	for(int i=0; i<5; i++){
		caffe_test_net.ForwardFromTo(0, 4);
		caffe_test_net.ForwardFromTo(5, 8);
	}
	printf("Time taken: %.2f ms\n", (double)(clock() - tStart)*1000/CLOCKS_PER_SEC);
	vector<Blob<float>* > result = caffe_test_net.output_blobs();

  //Here I can use the argmax layer, but for now I do a simple for :)
  float max = 0;
  float max_i = 0;
  
  for(int i=0; i<result.size(); i++){
  	LOG(INFO) << "SIZE"<<i<<" : " << result[i]->shape_string();
	}
  
  for (int i = 0; i < 10; ++i) {
    float value = result[2]->cpu_data()[i];
    LOG(INFO) << "["<<i<<"] : " << result[2]->cpu_data()[i] << ", " << result[3]->cpu_data()[i];
    if (max < value){
      max = value;
      max_i = i;
    }
  }
  LOG(INFO) << "max: " << max << " i " << max_i;
  
  return 0;
}
