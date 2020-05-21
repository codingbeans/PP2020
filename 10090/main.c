#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define CheckFailAndExit(status) \
    if (status != CL_SUCCESS) { \
        fprintf(stderr, "Error %d: Line %u in file %s\n\n", status, __LINE__, __FILE__), \
        release(); \
    }

// Thanks to Morris Dada
cl_context				clCtx;
cl_program 				clPrg;
cl_kernel				clKrn;
cl_command_queue		clQue;
cl_mem         			clMemOut;
#define MAXGPU 1
#define MAXN 16777216
#define GPULOCAL 256
uint32_t	hostC[MAXN/GPULOCAL];

int release() {
    fprintf(stderr, "Starting Cleanup ...\n\n");
    if (clMemOut) clReleaseMemObject(clMemOut);
    if (clKrn) clReleaseKernel(clKrn);
    if (clPrg) clReleaseProgram(clPrg);
    if (clQue) clReleaseCommandQueue(clQue);
    if (clCtx) clReleaseContext(clCtx);
    exit(0);
}

#define MAX_PROGRAM_LENGTH 3000 + 5

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
    cl_device_id device_id;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to get device id");
        return 0;
    }

    // Context
    clCtx = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    CheckFailAndExit(err);
    
    // Command queue
    clQue = clCreateCommandQueue(clCtx, device_id, 0, &err);
    CheckFailAndExit(err);
    
    // Program
    program_chars[length] = 0;
    const char* source = &program_chars[0];
    clPrg = clCreateProgramWithSource(clCtx, 1, &source, 0, 0);
    CheckFailAndExit(err);
    err = clBuildProgram(clPrg, 1, &device_id, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Line %u in file %s\n\n", __LINE__, __FILE__);
        size_t log_size;
        clGetProgramBuildInfo(clPrg, device_id,
                CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *program_log = (char *) calloc(log_size+1, sizeof(char));
        clGetProgramBuildInfo(clPrg, device_id,
                CL_PROGRAM_BUILD_LOG, log_size+1, program_log, NULL);
        program_log[log_size] = '\0';
        fprintf(stderr, "log= %s\n", program_log);
        free(program_log);
        CheckFailAndExit(err);
    }
    clKrn = clCreateKernel(clPrg, "vecdot", &err);
    CheckFailAndExit(err);

    // Buffers
    cl_mem_flags clOutBuffFlag = CL_MEM_WRITE_ONLY;
    clMemOut = clCreateBuffer(clCtx, clOutBuffFlag,
        sizeof(uint32_t)*MAXN/GPULOCAL, hostC, &err);
    CheckFailAndExit(err);

    return 1;
}

int execute() {
    uint32_t padding = 0;
    while (N%GPULOCAL) {
        padding += encrypt(N, keyA) * encrypt(N, keyB);
        N++;
    }

    cl_int err;
    err = clSetKernelArg(clKrn, 0, sizeof(cl_uint), (void *) &keyA);
    CheckFailAndExit(err);
    err = clSetKernelArg(clKrn, 1, sizeof(cl_uint), (void *) &keyB);
    CheckFailAndExit(err);
    err = clSetKernelArg(clKrn, 2, sizeof(cl_mem), (void *) &clMemOut);
    CheckFailAndExit(err);

    // Execute and get result
    size_t globalOffset[] = {0};
    size_t globalSize[] = {N};
    size_t localSize[] = {GPULOCAL};
    err = clEnqueueNDRangeKernel(clQue, clKrn, 1, globalOffset, globalSize, localSize, 0 ,NULL, NULL);
    CheckFailAndExit(err);

    clEnqueueReadBuffer(clQue, clMemOut, CL_TRUE, 0, sizeof(uint32_t)*N/GPULOCAL, 
            hostC, 0, NULL, NULL);

    uint32_t sum = 0;
    for (int i=0; i< N/GPULOCAL; i++) sum+= hostC[i];
    printf("%u\n", sum - padding);

    return 1;
}

int main() {
    init("vecdot.cl");
    while(scanf("%d %u %u", &N, &keyA, &keyB) == 3) {
        execute();
    }
    release();
}