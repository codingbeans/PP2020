#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX_FILENAME_LENGTH 30 + 5
#define MAX_PROGRAM_LENGTH 3000 + 5
#define MAX_LOG_LENGTH 3000 + 5

char program_chars[MAX_PROGRAM_LENGTH];
char program_log[MAX_LOG_LENGTH];
char filename[MAX_FILENAME_LENGTH];
int main() {
    scanf("%s", filename);
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

    cl_device_id device_id;
    err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Unable to get device id");
        return 0;
    }

    cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platforms[0], 0 };
    cl_context context = clCreateContext(prop, 1, &device_id, NULL, NULL, NULL);
    if(context == 0) {
        printf("Can't create OpenCL context\n");
        return 0;
    }

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
    program_chars[length] = 0;
    const char* source = &program_chars[0];
    cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
    if(program == 0) {
        return 0;
    }

    cl_int ret = clBuildProgram(program, 0, 0, 0, 0, 0);
    size_t len = 0;
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, len, program_log, NULL);
    program_log[len] = '\0';
    if(ret != CL_SUCCESS) {
        printf("%s", program_log);
        return 0;
    }
}