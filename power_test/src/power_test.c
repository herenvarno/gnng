#include<stdio.h>
#include<stdlib.h>
#include <limits.h>
#include <unistd.h>
#include <sys/types.h>
#include <signal.h>
#include <string.h>
#include <sys/wait.h>
#include <fcntl.h>

#define STEPNUM 23
#define STEPFIELD 3

int main(int argc, char* argv[])
{
	const char* STEPS[STEPNUM][STEPFIELD]={
		{"STEP 0-0. Measure steady power before test!\n", NULL, "0-normal-0.log"},
		{"STEP 1-0. Measure LeNet : CAFFE\n", "LeNet 1 0", "1-LeNet-0.log"},
		{"STEP 1-1. Measure LeNet : SET=1\n", "LeNet 1 1", "1-LeNet-1.log"},
		{"STEP 1-2. Measure LeNet : SET=2\n", "LeNet 1 2", "1-LeNet-2.log"},
		{"STEP 1-4. Measure LeNet : SET=4\n", "LeNet 1 4", "1-LeNet-4.log"},
		{"STEP 1-8. Measure LeNet : SET=8\n", "LeNet 1 8", "1-LeNet-8.log"},
		{"STEP 1-16. Measure LeNet : SET=16\n", "LeNet 1 16", "1-LeNet-16.log"},
		{"STEP 1-32. Measure LeNet : SET=32\n", "LeNet 1 32", "1-LeNet-32.log"},
		{"STEP 2-0. Measure CaffeNet : CAFFE\n", "CaffeNet 3 0", "2-CaffeNet-0.log"},
		{"STEP 2-1. Measure CaffeNet : SET=1\n", "CaffeNet 3 1", "2-CaffeNet-1.log"},
		{"STEP 2-2. Measure CaffeNet : SET=2\n", "CaffeNet 3 2", "2-CaffeNet-2.log"},
		{"STEP 2-4. Measure CaffeNet : SET=4\n", "CaffeNet 3 4", "2-CaffeNet-4.log"},
		{"STEP 2-8. Measure CaffeNet : SET=8\n", "CaffeNet 3 8", "2-CaffeNet-8.log"},
		{"STEP 2-16. Measure CaffeNet : SET=16\n", "CaffeNet 3 16", "2-CaffeNet-16.log"},
		{"STEP 2-32. Measure CaffeNet : SET=32\n", "CaffeNet 3 32", "2-CaffeNet-32.log"},
		{"STEP 3-0. Measure GoogleNet : CAFFE\n", "GoogleNet 3 0", "3-GoogleNet-0.log"},
		{"STEP 3-1. Measure GoogleNet : SET=1\n", "GoogleNet 3 1", "3-GoogleNet-1.log"},
		{"STEP 3-2. Measure GoogleNet : SET=2\n", "GoogleNet 3 2", "3-GoogleNet-2.log"},
		{"STEP 3-4. Measure GoogleNet : SET=4\n", "GoogleNet 3 4", "3-GoogleeNet-4.log"},
		{"STEP 3-8. Measure GoogleNet : SET=8\n", "GoogleNet 3 8", "3-GoogleNet-8.log"},
		{"STEP 3-16. Measure GoogleNet : SET=16\n", "GoogleNet 3 16", "3-GoogleNet-16.log"},
		{"STEP 3-32. Measure GoogleNet : SET=32\n", "GoogleNet 3 32", "3-GoogleNet-32.log"},
		{"STEP 0-1. Measure steady power after test!\n", NULL, "0-normal-1.log"}
	};
		
	char CMD[1024]={0};
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
	printf("%s\n", PREFIX);
	
	// INIT
	snprintf(CMD, sizeof(CMD), "%s/script/init.sh", PREFIX);
	system(CMD);
	
	// START TEST STEPS LOOP
	for(i=0;i<STEPNUM;i++)
	{
		sleep(60);
		printf("%s", STEPS[i][0]);
		pid = fork();
		if(pid==0)
		{
			// Child: measure the power
			//nvidia-smi -i 0 -q -d POWER -l 1 &> $LOGDIR/$1
			//FILE *f=fopen(STEPS[i][2], "w+");
			//dup2(fileno(f), STDOUT_FILENO);
			char path[1024]={0};
			snprintf(path, 1024, "%s/data/%s", PREFIX, STEPS[i][2]);
			int f=open(path, O_WRONLY|O_CREAT|O_TRUNC, 0666);
			dup2(f, STDOUT_FILENO);
			char *child_argv[]={"/usr/bin/nvidia-smi", "-i", "0", "-q", "-d", "POWER,PERFORMANCE", "-lms", "2", NULL};
			execv("/usr/bin/nvidia-smi", child_argv);
			exit(127);
		}
		else if(pid>0)
		{
			// Parent: run program
			sleep(2);
			if(STEPS[i][1]==NULL)
			{
				// for normal power test.
				sleep(5);
			}
			else
			{
				snprintf(CMD, sizeof(CMD), "%s/script/lancher.sh %s", PREFIX, STEPS[i][1]);
				system(CMD);
			}
			kill(pid, SIGKILL);
			int status;
			waitpid(pid, &status, 0);
			printf("%x\n", status);
		}
		else
		{
			// Error
			printf("ERROR !!!\n");
			return pid;
		}
	}
	printf("TEST COMPLETE!\n");
	return 0;
}
