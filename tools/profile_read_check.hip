// A known byte count to calibrate rocprof's derived FETCH_SIZE on this GPU.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#define CK(x) do { auto e=(x); if(e!=hipSuccess){fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e));return 1;} } while(0)
__global__ void profile_read(const unsigned * p, unsigned * out, size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; unsigned acc=0;
    for(;i<n;i+=(size_t)gridDim.x*blockDim.x) acc ^= p[i];
    out[blockIdx.x*blockDim.x+threadIdx.x]=acc;
}
int main() {
    for(size_t bytes:{size_t(1)<<20,size_t(256)<<20,size_t(1)<<30}) {
        unsigned * p,*out; CK(hipMalloc(&p,bytes)); CK(hipMalloc(&out,320*256*4)); CK(hipMemset(p,1,bytes));
        for(int k=0;k<3;++k) {
            hipEvent_t a,b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b)); CK(hipEventRecord(a));
            profile_read<<<320,256>>>(p,out,bytes/4); CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
            float ms; CK(hipEventElapsedTime(&ms,a,b)); printf("bytes=%zu repeat=%d ms=%.6f\n",bytes,k,ms);
            CK(hipEventDestroy(a)); CK(hipEventDestroy(b));
        }
        CK(hipFree(p)); CK(hipFree(out));
    }
    CK(hipDeviceReset()); return 0;
}
