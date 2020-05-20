#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <stdio.h>
#include <stdlib.h>

#define MAX_PROGRAM_LENGTH 300 + 5

char program_chars[MAX_PROGRAM_LENGTH];

cl_program load_program(cl_context context, const char* filename)
{
    FILE *fp;
    char ch;
    int length = 0;
    if ((fp = fopen(filename, "r")) == NULL) {
        printf("open file error!\n");
        return NULL;
    }
    
    while((ch = getc(fp)) != EOF) {
        program_chars[length++] = ch;
    }

    fclose(fp);
    program_chars[length] = 0;
    const char* source = &program_chars[0];
    cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
    if(program == 0) {
        return 0;
    }

    if(clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
        return 0;
    }

    return program;
}

int main() {
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

    // Need reinterpret_cast?
    cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0], 0 };
    cl_context context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
    if(context == 0) {
        printf("Can't create OpenCL context\n");
        return 0;
    }
    // clReleaseContext(context);
    // return 0;

    size_t cb;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    cl_device_id devices[cb / sizeof(cl_device_id)];
    clGetContextInfo(context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);

    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, 0, NULL, &cb);
    char devname[cb];
    clGetDeviceInfo(devices[0], CL_DEVICE_NAME, cb, &devname[0], 0);
    printf("Device: %s \n", devname);

    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, 0);
    if(queue == 0) {
        printf("Can't create command queue\n");
        clReleaseContext(context);
        return 0;
    }

    const int DATA_SIZE = 1048576;
    float a[DATA_SIZE], b[DATA_SIZE], res[DATA_SIZE];
    for(int i = 0; i < DATA_SIZE; i++) {
        a[i] = rand();
        b[i] = rand();
    }

    cl_mem cl_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &a[0], NULL);
    cl_mem cl_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * DATA_SIZE, &b[0], NULL);
    cl_mem cl_res = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * DATA_SIZE, NULL, NULL);
    if(cl_a == 0 || cl_b == 0 || cl_res == 0) {
        printf("Can't create OpenCL buffer\n");
        clReleaseMemObject(cl_a);
        clReleaseMemObject(cl_b);
        clReleaseMemObject(cl_res);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }

    cl_program program = load_program(context, "shader.cl");
    if(program == 0) {
        printf("Can't load or build program\n");
        clReleaseMemObject(cl_a);
        clReleaseMemObject(cl_b);
        clReleaseMemObject(cl_res);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }

    cl_kernel adder = clCreateKernel(program, "adder", 0);
    if(adder == 0) {
        printf("Can't load kernel\n");
        clReleaseProgram(program);
        clReleaseMemObject(cl_a);
        clReleaseMemObject(cl_b);
        clReleaseMemObject(cl_res);
        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        return 0;
    }

    clSetKernelArg(adder, 0, sizeof(cl_mem), &cl_a);
    clSetKernelArg(adder, 1, sizeof(cl_mem), &cl_b);
    clSetKernelArg(adder, 2, sizeof(cl_mem), &cl_res);

    size_t work_size = DATA_SIZE;
    err = clEnqueueNDRangeKernel(queue, adder, 1, 0, &work_size, 0, 0, 0, 0);
    if(err == CL_SUCCESS) {
        err = clEnqueueReadBuffer(queue, cl_res, CL_TRUE, 0, sizeof(float) * DATA_SIZE, &res[0], 0, 0, 0);
    }

    if(err == CL_SUCCESS) {
        int correct = 1;
        for(int i = 0; i < DATA_SIZE; i++) {
            if(a[i] + b[i] != res[i]) {
                correct = 0;
                break;
            }
        }

        if(correct) {
            printf("Data is correct\n");
        }
        else {
            printf("Data is incorrect\n");
        }
    }
    else {
        printf("Can't run kernel or read back data\n");
    }


    clReleaseKernel(adder);
    clReleaseProgram(program);
    clReleaseMemObject(cl_a);
    clReleaseMemObject(cl_b);
    clReleaseMemObject(cl_res);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}