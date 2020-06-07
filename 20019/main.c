#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "utils.h"
#define MAXGPU 2
#define MAXN 16777216
#define GPULOCAL 256
#define BLK 256
#define MAXCASE 100000

// A lot thanks to Morris
cl_context clCtx[MAXGPU];
cl_program clPrg[MAXGPU];
cl_kernel clKrn[MAXGPU];
cl_command_queue clQue[MAXGPU];
cl_mem clMemOut[MAXGPU];

int release() {
    fprintf(stderr, "Starting Cleanup ...\n\n");
    for (int i=0; i<MAXGPU; i++) {
        if (clMemOut[i]) clReleaseMemObject(clMemOut[i]);
        if (clKrn[i]) clReleaseKernel(clKrn[i]);
        if (clPrg[i]) clReleaseProgram(clPrg[i]);
        if (clQue[i]) clReleaseCommandQueue(clQue[i]);
        if (clCtx[i]) clReleaseContext(clCtx[i]);
    }
    exit(0);
}

#define MAX_PROGRAM_LENGTH 1500 + 5

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

    program_chars[length] = 0;
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
	
    // #pragma omp parallel for
    for (int device = 0; device < MAXGPU; device++) {
        // Context
        clCtx[device] = clCreateContext(NULL, 1, &device_id[device], NULL, NULL, &err);
        if(err != CL_SUCCESS) {
            printf("Unable to create context\n");
            return 0;
        }
        
        // Command queue
        clQue[device] = clCreateCommandQueue(clCtx[device], device_id[device], 0, &err);
        if(err != CL_SUCCESS) {
            printf("Unable to create command queue\n");
            return 0;
        }
        
        // Program
        const char* source = program_chars;
        clPrg[device] = clCreateProgramWithSource(clCtx[device], 1, &source, (size_t *) &length, &err);
        if(err != CL_SUCCESS) {
            printf("Unable to create program\n");
            return 0;
        }

        err = clBuildProgram(clPrg[device], 1, &device_id[device], NULL, NULL, NULL);
        if(err != CL_SUCCESS) {
            size_t log_size;
            clGetProgramBuildInfo(clPrg[device], device_id[device],
                    CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
            char *program_log = (char *) calloc(log_size+1, sizeof(char));
            clGetProgramBuildInfo(clPrg[device], device_id[device],
                    CL_PROGRAM_BUILD_LOG, log_size+1, program_log, NULL);
            fprintf(stderr, "%s", program_log);
            free(program_log);
            printf("Unable to build program\n");
            return 0;
        }
        
        // Kernel
        clKrn[device] = clCreateKernel(clPrg[device], "vecdot", &err);
        if(err != CL_SUCCESS) {
            printf("Unable to create kernel\n");
            return 0;
        }

        // Buffers
        cl_mem_flags clOutBuffFlag = CL_MEM_READ_WRITE;
        clMemOut[device] = clCreateBuffer(clCtx[device], clOutBuffFlag,
            sizeof(uint32_t)*BLK, NULL, &err);
        if(err != CL_SUCCESS) {
            printf("Unable to create buffer\n");
            return 0;
        }
    }

    return 1;
}

int execute(int GPUID, int N, uint32_t keyA, uint32_t keyB) {
    cl_int err;
	err = clSetKernelArg(clKrn[GPUID], 0, sizeof(cl_uint), (void *) &N);
    if(err != CL_SUCCESS) {
        printf("Unable to set kernel arg 0\n");
        return 0;
    }
    err = clSetKernelArg(clKrn[GPUID], 1, sizeof(cl_uint), (void *) &keyA);
    if(err != CL_SUCCESS) {
        printf("Unable to set kernel arg 1\n");
        return 0;
    }
    err = clSetKernelArg(clKrn[GPUID], 2, sizeof(cl_uint), (void *) &keyB);
    if(err != CL_SUCCESS) {
        printf("Unable to set kernel arg 2\n");
        return 0;
    }
    err = clSetKernelArg(clKrn[GPUID], 3, sizeof(cl_mem), (void *) &clMemOut[GPUID]);
    if(err != CL_SUCCESS) {
        printf("Unable to set kernel arg 3 GPU:%d\n", GPUID);
        return 0;
    }

    // Partition to blocks, each size 256
    N = (N+GPULOCAL*BLK-1)/(GPULOCAL*BLK)*GPULOCAL;
    size_t globalOffset[] = {0};
    size_t globalSize[] = {N};
    size_t localSize[] = {GPULOCAL};

    uint32_t ZERO = 0;
    err = clEnqueueWriteBuffer(clQue[GPUID], clMemOut[GPUID], CL_TRUE, 0, sizeof(uint32_t), (void *) &ZERO, 0, NULL, NULL);
    err = clEnqueueNDRangeKernel(clQue[GPUID], clKrn[GPUID], 1, globalOffset,
            globalSize, localSize, 0, NULL, NULL);
    if(err != CL_SUCCESS) {
        printf("Unable to enqueue\n");
        return 0;
    }

	// -- read back
	uint32_t sum = 0;
	clEnqueueReadBuffer(clQue[GPUID], clMemOut[GPUID], CL_TRUE, 0, sizeof(uint32_t), &sum, 0, NULL, NULL);
    // printf("%u\n", sum);

    return sum;
}

int tc;
int N[MAXCASE];
uint32_t ans[MAXCASE];
uint32_t keyA[MAXCASE], keyB[MAXCASE];
void flush() {
    if (tc == 0) return;
	omp_set_num_threads(MAXGPU);
#pragma omp parallel for schedule(dynamic)
    for (int i=0; i<tc; i++) {
        ans[i] = execute(omp_get_thread_num(), N[i], keyA[i], keyB[i]);
    }
    for (int i=0; i<tc; i++) {
        printf("%u\n", ans[i]);
    }
}

int main() {
    init("vecdot.cl");
    while(scanf("%d %u %u", &N[tc], &keyA[tc], &keyB[tc]) == 3) {
        tc++;
        if (tc == MAXCASE) {
            flush();
            tc = 0;
        }
    }
    flush();
    release();
}