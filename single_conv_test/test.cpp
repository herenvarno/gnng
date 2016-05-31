#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include <time.h>
#include <boost/format.hpp>
#include <sys/wait.h>
#include <fcntl.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <sys/stat.h>

using namespace caffe;
using namespace std;
using namespace cv;

#define INPUT_CHANNEL_NUM 7
#define INPUT_SIZE_NUM 6
#define OUTPUT_CHANNEL_NUM 7
#define KERNEL_SIZE_NUM 5
#define SET_NUM 7

int INPUT_CHANNEL_TABLE[INPUT_CHANNEL_NUM] = {3, 10, 50, 100, 200, 500, 1000};
int INPUT_SIZE_TABLE[INPUT_SIZE_NUM] = {32, 64, 128, 256, 512, 1024};
int OUTPUT_CHANNEL_TABLE[OUTPUT_CHANNEL_NUM] = {1, 10, 50, 100, 200, 500, 1000};
int KERNEL_SIZE_TABLE[KERNEL_SIZE_NUM] = {2, 3, 4, 5, 8};
int SET_TABLE[SET_NUM] = {0, 1, 2, 4, 8, 16, 32};

int main(int argc, char* argv[]) {
  
	Caffe::set_mode(Caffe::GPU);
	
	unlink("time.log");
	ofstream logfile;
	
	// CREATE DATA PATH
	char PREFIX[1024]={0};
	pid_t pid;
	
	ssize_t count = readlink("/proc/self/exe", PREFIX, 1024);
	int i;
	for(i=strlen(PREFIX); i>0; i--)
	{
		if(PREFIX[i]=='/')
		{
			PREFIX[i]='\0';
			break;
		}
	}
	
	char DATAPATH[1024];
	snprintf(DATAPATH, sizeof(DATAPATH), "%s/data", PREFIX);
	mkdir(DATAPATH, 0755);
	
	// PRE MEASURE IDLE STATE
	pid = fork();
	if(pid==0)
	{
		// Child: measure the power
		//nvidia-smi -i 0 -q -d POWER,PERFORMANCE,MEMORY -l 1 &> $LOGDIR/$1
		char path[1024]={0};
		snprintf(path, 1024, "%s/data/0-normal-0.log", PREFIX);
		char* child_argv[]={"/usr/bin/nvidia-smi", "-i", "0", "--format=csv,noheader,nounits", "--query-gpu=pstate,memory.used,power.draw", "-lms", "2", "-f", path, NULL};
		execv("/usr/bin/nvidia-smi", child_argv);
		exit(127);
	}
	else if(pid>0)
	{
		// Parent: run program
		sleep(5);
		kill(pid, SIGKILL);
		int status;
		waitpid(pid, &status, 0);
	}
	else
	{
		// Error
		printf("ERROR !!!\n");
		return pid;
	}
	
	
	for(int input_channel=0; input_channel<INPUT_CHANNEL_NUM; input_channel++){
		for(int input_size=0; input_size<INPUT_SIZE_NUM; input_size++){
			for(int output_channel=0; output_channel<OUTPUT_CHANNEL_NUM; output_channel++){
				for(int kernel_size=0; kernel_size<KERNEL_SIZE_NUM; kernel_size++){
					for(int set=0; set < SET_NUM; set++){
					
						if(INPUT_SIZE_TABLE[input_size]>256 and INPUT_CHANNEL_TABLE[input_channel] > 100)
							continue;
						
						int LOOP_TIMES;
						if(INPUT_SIZE_TABLE[input_size]>512)
							LOOP_TIMES = 1;
						else if(INPUT_SIZE_TABLE[input_size]>64)
							LOOP_TIMES = 10;
						else
							LOOP_TIMES = 100;
						
						sleep(10);
						printf("TEST [input_channel=%d, input_size=%d, output_channel=%d, kernel_size=%d, set=%d]\n", INPUT_CHANNEL_TABLE[input_channel], INPUT_SIZE_TABLE[input_size], OUTPUT_CHANNEL_TABLE[output_channel], KERNEL_SIZE_TABLE[kernel_size], SET_TABLE[set]);
						pid = fork();
						if(pid==0)
						{
							// Child: measure the power
							//nvidia-smi -i 0 --format=csv,noheader --query-gpu=timestamp,pstate,memory.used,power.draw -lms 2
							char path[1024]={0};
							snprintf(path, 1024, "%s/data/1-%d-%d-%d-%d-%d.log", PREFIX, INPUT_CHANNEL_TABLE[input_channel], INPUT_SIZE_TABLE[input_size], OUTPUT_CHANNEL_TABLE[output_channel], KERNEL_SIZE_TABLE[kernel_size], SET_TABLE[set], NULL);
							char* child_argv[]={"/usr/bin/nvidia-smi", "-i", "0", "--format=csv,noheader,nounits", "--query-gpu=pstate,memory.used,power.draw", "-lms", "2", "-f", path, NULL};
							execv("/usr/bin/nvidia-smi", child_argv);
							exit(127);
						}
						else if(pid>0)
						{
							// Parent: run program
							sleep(2);
							logfile.open("time.log", std::ofstream::out | std::ofstream::app);
							// CREATE THE PROTOTXT
							ifstream ifs("template.prototxt");
							string format=string((std::istreambuf_iterator<char>(ifs)),
								(std::istreambuf_iterator<char>()));
							ifs.close();
							ofstream ofs("net.prototxt");
							ofs << boost::format(format) % INPUT_CHANNEL_TABLE[input_channel] % INPUT_SIZE_TABLE[input_size] % INPUT_SIZE_TABLE[input_size] % SET_TABLE[set] % OUTPUT_CHANNEL_TABLE[output_channel] % KERNEL_SIZE_TABLE[kernel_size]; 
							ofs.close();

							// CREATE THE INPUT IMAGE
							Mat I(INPUT_CHANNEL_TABLE[input_channel], INPUT_SIZE_TABLE[input_size]*INPUT_SIZE_TABLE[input_size], DataType<unsigned char>::type);
							theRNG().state = time(NULL);
							theRNG().fill(I, RNG::UNIFORM, 0, 255);
							I=I.reshape(INPUT_CHANNEL_TABLE[input_channel], INPUT_SIZE_TABLE[input_size]);

							// CREATE THE NET
							Net<float> caffe_test_net("net.prototxt", TEST);
					
							// FEED THE INPUT LAYER WITH INPUT IMAGE
							vector<cv::Mat> dv;
							dv.push_back(I);
							vector<int> dvl;
							dvl.push_back(0);
							boost::dynamic_pointer_cast<MemoryDataLayer<float> >(caffe_test_net.layers()[0])->AddMatVector(dv,dvl);
					
							// FORWARD 100 TIMES
							vector<Blob<float>* > result;
							clock_t tStart = clock();
							for(int i=0; i<LOOP_TIMES; i++){
								result = caffe_test_net.Forward();
							}
							logfile << boost::format("[input_channel=%d, input_size=%d, output_channel=%d, kernel_size=%d, set=%d] : %.2lf\n") % INPUT_CHANNEL_TABLE[input_channel] % INPUT_SIZE_TABLE[input_size] % OUTPUT_CHANNEL_TABLE[output_channel] % KERNEL_SIZE_TABLE[kernel_size] % SET_TABLE[set] % double((clock() - tStart)/(CLOCKS_PER_SEC*0.001));
							logfile.close();
							kill(pid, SIGKILL);
							int status;
							waitpid(pid, &status, 0);
						}
						else
						{
							// Error
							printf("ERROR !!!\n");
							return pid;
						}
					}
				}
			}
		}
	}
	
	// POST MEASURE STATE
	pid = fork();
	if(pid==0)
	{
		// Child: measure the power
		char path[1024]={0};
		snprintf(path, 1024, "%s/data/0-normal-1.log", PREFIX);
		char* child_argv[]={"/usr/bin/nvidia-smi", "-i", "0", "--format=csv,noheader,nounits", "--query-gpu=pstate,memory.used,power.draw", "-lms", "2", "-f", path, NULL};
		execv("/usr/bin/nvidia-smi", child_argv);
		exit(127);
	}
	else if(pid>0)
	{
		// Parent: run program
		sleep(5);
		kill(pid, SIGKILL);
		int status;
		waitpid(pid, &status, 0);
	}
	else
	{
		// Error
		printf("ERROR !!!\n");
		return pid;
	}
	return 0;
}
