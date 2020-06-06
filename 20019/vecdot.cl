#ifndef _UTILS_H
#define _UTILS_H
#define uint32_t unsigned int
static inline uint32_t rotate_left(uint32_t x, uint32_t n) {
    return  (x << n) | (x >> (32-n));
}
static inline uint32_t encrypt(uint32_t m, uint32_t key) {
    return (rotate_left(m, key&31) + key)^key;
}
#endif

#define GPULOCAL 256
#define BLK 256
// Thanks to Morris
__kernel void vecdot(int L, int R, uint32_t keyA, uint32_t keyB, __global int* C) {
    __local int buf[BLK];
    int globalId = get_global_id(0);
    int groupId = get_group_id(0);
    int localId = get_local_id(0);
    int localSz = get_local_size(0);
    int l = globalId * BLK + L;
    int r = l + BLK;
    if (r >= R - 1) r = R - 1;
    int sum = 0;
    for (int i = l; i < r; i++)
        sum += encrypt(i, keyA) * encrypt(i, keyB);
    buf[localId] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = localSz>>1; i; i >>= 1) {
        if (localId < i)
            buf[localId] += buf[localId + i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0) {
        uint32_t t = buf[0];
        atomic_add(&C[0], t);
    }
}