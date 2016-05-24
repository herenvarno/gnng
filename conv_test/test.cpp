#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

using namespace caffe;
using namespace std;
using namespace cv;

int main(int argc, char** argv) {
 
  if (argc < 4) {
    LOG(ERROR) << "test_net net_proto pretrained_net_proto image ";
    return 1;
  }
  
  Caffe::set_mode(Caffe::GPU);
   Mat image;
//   transpose(imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE), image) ;  // Read the file
//	image=imread(argv[3], CV_LOAD_IMAGE_GRAYSCALE);
	image=imread(argv[3]);
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
  
  return 0;
}
