#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x) do { auto e=(x); if(e!=hipSuccess){ fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e)); exit(1); } } while(0)

// The source is one Q8_0 wave's eight groups. Odd 272-byte group phases put
// scales first, even phases put codes first; there is no padding or extra bank.
__global__ void init(unsigned char *inter, unsigned char *wave, int groups) {
    int g=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x&31;
    if(g>=groups) return;
    int c=(g&1)?16:0, s=(g&1)?0:256;
    for(int b=0;b<8;++b) {
        unsigned char v=(unsigned char)((g*17+b*53+lane*29)&255);
        inter[g*272+b*34+2+lane]=v;
        wave[g*272+c+b*32+lane]=v;
        if(lane==b) {
            unsigned short h=(unsigned short)(0x3c00+((g*13+b*3)&63));
            inter[g*272+b*34]=(unsigned char)h;
            inter[g*272+b*34+1]=(unsigned char)(h>>8);
            wave[g*272+s+b*2]=(unsigned char)h;
            wave[g*272+s+b*2+1]=(unsigned char)(h>>8);
        }
    }
}

template<bool TRANSPOSED>
__global__ void consume(const unsigned char *image, float *out, int rows) {
    int row=(blockIdx.x*blockDim.x+threadIdx.x)/32, lane=threadIdx.x&31;
    if(row>=rows) return;
    float sum=0.0f;
    int a=(lane&15)-7;
    for(int k=0;k<32;++k) {
        int g=row*32+k, c=(g&1)?16:0, s=(g&1)?0:256;
        for(int b=0;b<8;++b) {
            unsigned short bits=0;
            if(lane==b) {
                auto p=image+g*272+(TRANSPOSED?s+b*2:b*34);
                bits=(unsigned short)p[0]|((unsigned short)p[1]<<8);
            }
            bits=__shfl(bits,b);
            union { unsigned short bits; __half value; } scale={bits};
            int code=(int)(signed char)image[g*272+(TRANSPOSED?c+b*32+lane:b*34+2+lane)];
            sum=fmaf((float)(code*a),__half2float(scale.value),sum);
        }
    }
    for(int offset=16;offset;offset>>=1) sum+=__shfl_down(sum,offset);
    if(!lane) out[row]=sum;
}

static double run(bool transposed,const unsigned char *inter,const unsigned char *wave,float *out,int rows) {
    hipEvent_t start,stop;
    CHECK(hipEventCreate(&start)); CHECK(hipEventCreate(&stop));
    CHECK(hipEventRecord(start));
    if(transposed) hipLaunchKernelGGL((consume<true>),dim3((rows+3)/4),dim3(128),0,0,wave,out,rows);
    else hipLaunchKernelGGL((consume<false>),dim3((rows+3)/4),dim3(128),0,0,inter,out,rows);
    CHECK(hipGetLastError()); CHECK(hipEventRecord(stop)); CHECK(hipEventSynchronize(stop));
    float ms; CHECK(hipEventElapsedTime(&ms,start,stop));
    CHECK(hipEventDestroy(start)); CHECK(hipEventDestroy(stop));
    return ms;
}

int main(int argc,char **argv) {
    int rows=argc>1?atoi(argv[1]):171520;
    if(rows<=0) return 2;
    size_t groups=(size_t)rows*32, bytes=groups*272;
    unsigned char *inter,*wave; float *out;
    CHECK(hipMalloc(&inter,bytes)); CHECK(hipMalloc(&wave,bytes)); CHECK(hipMalloc(&out,rows*sizeof(float)));
    hipLaunchKernelGGL(init,dim3((groups+127)/128),dim3(128),0,0,inter,wave,(int)groups);
    CHECK(hipGetLastError()); CHECK(hipDeviceSynchronize());
    std::vector<float> old(rows),now(rows);
    run(false,inter,wave,out,rows); CHECK(hipMemcpy(old.data(),out,rows*sizeof(float),hipMemcpyDeviceToHost));
    run(true,inter,wave,out,rows); CHECK(hipMemcpy(now.data(),out,rows*sizeof(float),hipMemcpyDeviceToHost));
    size_t mismatch=0;
    for(int i=0;i<rows;++i) mismatch+=memcmp(&old[i],&now[i],sizeof(float))!=0;
    printf("rows=%d groups=%zu image_bytes=%zu differing_f32=%zu\n",rows,groups,bytes,mismatch);
    if(mismatch) return 3;
    for(int r=0;r<7;++r) {
        if(r&1) printf("round=%d wave_ms=%.6f inter_ms=%.6f\n",r,run(true,inter,wave,out,rows),run(false,inter,wave,out,rows));
        else printf("round=%d inter_ms=%.6f wave_ms=%.6f\n",r,run(false,inter,wave,out,rows),run(true,inter,wave,out,rows));
    }
    CHECK(hipFree(inter)); CHECK(hipFree(wave)); CHECK(hipFree(out));
}
