#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utils.h"
#define MAXGPU 1
#define MAXN 16777216
#define GPULOCAL 256
#define BLK 256

// A lot thanks to Morris
cl_context clCtx;
cl_program clPrg;
cl_kernel clKrn;
cl_command_queue clQue;
cl_mem clMemOut;

uint32_t hostC[BLK];

int release() {
    fprintf(stderr, "Starting Cleanup ...\n\n");
    if (clMemOut) clReleaseMemObject(clMemOut);
    if (clKrn) clReleaseKernel(clKrn);
    if (clPrg) clReleaseProgram(clPrg);
    if (clQue) clReleaseCommandQueue(clQue);
    if (clCtx) clReleaseContext(clCtx);
    exit(0);
}

#define MAX_PROGRAM_LENGTH 1500 + 5

int N;
uint32_t keyA, keyB;
char program_chars[MAX_PROGRAM_LENGTH];
int init(const char* filename) {
    FILE *fp;
    char ch;
    int length = 0;
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("open file error!\n");
        return 0;
    }
    
    while((ch = getc(fp)) != EOF) {
        program_chars[length++] = ch;
    }

    fclose(fp);
    // Finish reading files

    // Platform
    cl_int err;
    cl_uint num;
    err = clGetPlatformIDs(0, 0, &num);
    if(err != CL_SUCCESS) {
        printf("Unable to get platforms\n");
        return 0;
    }

    cl_platform_id platforms[num];
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if(err != CL_SUCCESS) {
        printf("Unable to get platform ID\n");
        return 0;
    }

    // Device
    cl_device_id device_id[MAXGPU];
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, MAXGPU, device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to get device id");
        return 0;
    }

    // Context
    clCtx = clCreateContext(NULL, 1, device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS) {
        printf("Unable to create context\n");
        return 0;
    }
    
    // Command queue
    clQue = clCreateCommandQueue(clCtx, device_id[0], 0, &err);
    if(err != CL_SUCCESS) {
        printf("Unable to create command queue\n");
        return 0;
    }
    
    // Program
    program_chars[length] = 0;
    const char* source = &program_chars[0];
    clPrg = clCreateProgramWithSource(clCtx, 1, &source, 0, &err);
    if(err != CL_SUCCESS) {
        printf("Unable to create program\n");
        return 0;
    }

    err = clBuildProgram(clPrg, 1, device_id, NULL, NULL, NULL);
    if(err != CL_SUCCESS) {
		size_t log_size;
		clGetProgramBuildInfo(clPrg, device_id[0],
				CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char *program_log = (char *) calloc(log_size+1, sizeof(char));
		clGetProgramBuildInfo(clPrg, device_id[0],
				CL_PROGRAM_BUILD_LOG, log_size+1, program_log, NULL);
		fprintf(stderr, "%s", program_log);
		free(program_log);
        printf("Unable to build program\n");
        return 0;
    }
    
    // Kernel
    clKrn = clCreateKernel(clPrg, "vecdot", &err);
    if(err != CL_SUCCESS) {
        printf("Unable to create kernel\n");
        return 0;
    }

    // Buffers
    cl_mem_flags clOutBuffFlag = CL_MEM_WRITE_ONLY;
    clMemOut = clCreateBuffer(clCtx, clOutBuffFlag,
        sizeof(uint32_t)*BLK, hostC, &err);
    if(err != CL_SUCCESS) {
        printf("Unable to create buffer\n");
        return 0;
    }

    return 1;
}

int execute() {
    cl_int err;
	err = clSetKernelArg(clKrn, 0, sizeof(cl_uint), (void *) &N);
    if(err != CL_SUCCESS) {
        printf("Unable to set kernel arg 0\n");
        return 0;
    }
    err = clSetKernelArg(clKrn, 1, sizeof(cl_uint), (void *) &keyA);
    if(err != CL_SUCCESS) {
        printf("Unable to set kernel arg 1\n");
        return 0;
    }
    err = clSetKernelArg(clKrn, 2, sizeof(cl_uint), (void *) &keyB);
    if(err != CL_SUCCESS) {
        printf("Unable to set kernel arg 2\n");
        return 0;
    }
    err = clSetKernelArg(clKrn, 3, sizeof(cl_mem), (void *) &clMemOut);
    if(err != CL_SUCCESS) {
        printf("Unable to set kernel arg 3\n");
        return 0;
    }

    // Partition to blocks, each size 256
    N = (N+GPULOCAL*BLK-1)/(GPULOCAL*BLK)*GPULOCAL;
    size_t globalOffset[] = {0};
    size_t globalSize[] = {N};
    size_t localSize[] = {GPULOCAL};

    err = clEnqueueNDRangeKernel(clQue, clKrn, 1, globalOffset,
            globalSize, localSize, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        printf("Unable to enqueue\n");
        return 0;
    }

	// -- read back
	uint32_t sum = 0;
	clEnqueueReadBuffer(clQue, clMemOut, CL_TRUE, 0, sizeof(uint32_t), &sum, 0, NULL, NULL);
    printf("%u\n", sum);

    return 1;
}

int main() {
    init("vecdot.cl");
    while(scanf("%d %u %u", &N, &keyA, &keyB) == 3) {
        execute();
    }
    release();
}