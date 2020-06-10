#include "labeling.h"
#include <thrust/tabulate.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

// Negate struct defined in thrust as an example
// template<typename T>
// struct negate
//  {
//    typedef T argument_type;
 
//    typedef T result_type;
 
//    __thrust_exec_check_disable__
//    __host__ __device__ T operator()(const T &x) const {return -x;}
//  }; // end negate
 
template<typename T>
struct str2tag {
    typedef T argument_type;
    typedef T result_type;
    const char* cuStr;
    str2tag(const char* cuStr): cuStr(cuStr) {}
//    __thrust_exec_check_disable__
   __host__ __device__ T operator()(const T &index) const { return (cuStr[index] != ' ') ? -1 : index;}
};

 
template<typename T>
struct tag2ans {
    typedef T argument_type;
    typedef T result_type;
    const int32_t* cuPos;
    tag2ans(const char* cuPos): cuPos(cuPos) {}
//    __thrust_exec_check_disable__
   __host__ __device__ T operator()(const T &index) const { return index - cuPos[index];}
};

void labeling(const char *cuStr, int *cuPos, int strLen) {
	thrust::tabulate(thrust::device, cuPos, cuPos+strLen, str2tag<int32_t>(cuStr));
	thrust::inclusive_scan(thrust::device, cuPos, cuPos+strLen, cuPos, thrust::maximum<int>());
	thrust::tabulate(thrust::device, cuPos, cuPos+strLen, tag2ans<int32_t>(cuPos));
}