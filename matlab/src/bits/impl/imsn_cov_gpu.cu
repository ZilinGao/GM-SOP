// @file  mpn_cov_gpu.cu
// @brief MPN-COV implementation (GPU)
// @author Jiangtao Xie
// @author Peihua Li

/*
Copyright (C) 2017 Peihua Li and Jiangtao Xie

All rights reserved.
*/

#include "nnimsn_cov_blas.hpp"
#include "../data.hpp"
#include <math.h>
#include <memory>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <cassert>
#include "blashelper_gpu.hpp"


#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n); \
         i += blockDim.x * gridDim.x)   // We import this Macro function  from our Caffe Implementation 

inline int
GET_BLOCKS(const int N)
{
    return (N + VL_CUDA_NUM_THREADS - 1) / VL_CUDA_NUM_THREADS; // We import this function  from our Caffe Implementation 
}

template<typename T> __global__ void set_kernel(const ptrdiff_t n, const T alpha, T* y) 
{
    CUDA_KERNEL_LOOP(index, n) {
        y[index] = alpha;
    }
}

template<typename T> void gpuMemset(const ptrdiff_t n, const T alpha, T* y)
{
    if(alpha == 0){
        cudaMemset(y, 0, sizeof(T)*n);
    }
    set_kernel<T><<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(n , alpha, y);
}
template<typename T> __global__ void init_II_kernel(T* a,
                                                   T aux_value,
                                                   const ptrdiff_t n)
{
    CUDA_KERNEL_LOOP(index,n){
        a[index*(n+1)]  =  a[index*(n+1)] + aux_value;
    }
}
template<typename T> __global__ void init_I_kernel(T* a,
												   T* b,
                                                   const ptrdiff_t n)
{
    CUDA_KERNEL_LOOP(index,n){
        a[index*(n+1)]  =  3;
		b[index*(n+1)]  =  1;
    }
}

template<typename T> __global__ void processEig_kernel(T* diagS,
                                                  T* diagSroot,
                                                  T* diagSderi,
                                                  T* dev_s,
                                                  ptrdiff_t n,
                                                  int Dmin,
                                                  T epsilon,
                                                  T alpha)
{
    CUDA_KERNEL_LOOP(i,Dmin){
        diagS[i] = dev_s[i];
        diagSroot[i] = std::pow(dev_s[i] + epsilon,alpha);
        diagSderi[i] = alpha * std::pow(dev_s[i] + epsilon,alpha - 1);
    }
}
template<typename T> inline void
processEigResults_gpu(T *diagS,
                      T *diagSroot,
                      T *diagSderi,
                      T *dev_s,
                      T *aux_D,
                      ptrdiff_t n,
                      T epsilon,
                      T alpha)
{
    int Dmin = (int)(*aux_D);
    processEig_kernel<T>
        <<<GET_BLOCKS(Dmin),VL_CUDA_NUM_THREADS>>>(diagS,diagSroot,diagSderi,dev_s,n,Dmin,epsilon,alpha);
}
template<typename T> __global__ void getOutput_kernel(T* a,
                                                      T* b,
                                                      ptrdiff_t n)
{
    int lda = n,offset,idx = 0;
    CUDA_KERNEL_LOOP(i,n){
        offset = i;
        for(int j = 0;j < offset + 1;j ++) {
            idx = i * (i + 1) / 2 + j;
            a[idx] = b[i * lda + j];
        }
    }
}
template<typename T>  __host__ void
getOutput_gpu(T* output,
              T* result,
              ptrdiff_t n)
{
    getOutput_kernel<T><<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(output,result,n);
}
template<typename T> __global__ void getdLdCfromderOutput_kernel(T* a,
                                                                 T const* b,
                                                                 ptrdiff_t n)
{
    int lda = n,offset,idx = 0;
    CUDA_KERNEL_LOOP(i,n){
        offset = i;
        for(int j = 0;j < offset + 1;j ++){
            idx = i * (i + 1) / 2 + j;
            a[i * lda + j] = b[idx];
        }
    }
}
template<typename T>  __host__ void
getdLdCfromderOutput_gpu(T* dLdC,
                         T const* derOutput,
                         ptrdiff_t n)
{
    getdLdCfromderOutput_kernel<T><<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(dLdC,derOutput,n);
}

template<typename T> __global__ void symmetric_kernel(T* a,
                                                      int n)
{
    int lda = n,offset;
    CUDA_KERNEL_LOOP(i,n){
        offset = i;
        for(int j = offset;j < n;j ++) {
            a[i * lda + j] = (a[i * lda + j] + a[j * lda + i]) / 2;
            a[j * lda + i] = a[i * lda + j];
        }
    }
}

template<typename T>  __host__ void
symmetric_gpu(T* a,
              ptrdiff_t n)
{
    symmetric_kernel<T>
        <<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(a,n);
}

template<typename T> __global__ void traceNormBackward_kernel(T* a,
                                                              T const alpha,
					                                          T beta,
					                                          ptrdiff_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        a[i*(n+1)] = a[i*(n+1)] - beta/alpha;
    }
}

template<typename T>  __host__ void
traceNormBackward_gpu(T* a,
                      T const alpha,
					  T beta,
					  ptrdiff_t n)
{
    traceNormBackward_kernel<T>
		<<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(a,alpha,beta,n);
}
template<typename T> __global__ void matrixAdd_kernel(T* x,
                                                      T* y,
		                                              T* z,
		                                              T alpha,
		                                              T beta,
		                                              T sigma,
		                                              ptrdiff_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
        x[i] = alpha * x[i] + beta * y[i] + sigma * z[i];
    }
}
template<typename T> __host__ void
matrixAdd_gpu(T* x,
              T* y,
		      T* z,
		      T alpha,
		      T beta,
		      T sigma,
		      ptrdiff_t n)
{
    matrixAdd_kernel<T>
        <<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(x,y,z,alpha,beta,sigma,n);
}
template<typename T> __global__ void copy_kernel(T* a,
                                                 T* b,
												 ptrdiff_t n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		a[i] = b[i];
	}
}
namespace vl { namespace impl {
    template<typename T,vl::DataType dataType>
    struct imsn_cov<vl::VLDT_GPU,T,dataType>
    {
        static vl::ErrorCode
            forward(Context& context,
                    T* output,
                    T const* data,
                    T* aux_Y,
                    T* aux_Z,
					T* aux_T,
                    size_t height, size_t width, size_t depth, size_t num,
                    int iterNum)
        {
           vl::ErrorCode error;
            ptrdiff_t m = height,n = width,L = depth,d,i;
			ptrdiff_t dataOffset,ypOffset;
			ptrdiff_t aux_YOffset,aux_ZOffset,aux_TOffset;
			ptrdiff_t aux_YOffset_1,aux_ZOffset_1,outputOffset;
			unsigned int workspaceSize =  (unsigned int)(n*n*(3) + m*m + m*n);
			T* workspace = (T*)context.getWorkspace(vl::VLDT_GPU , workspaceSize*sizeof(T));
			T* II        = workspace;
            T* cov_work  = II + m*m;
			T* I3        = cov_work + m*n;
			T* result    = I3 + n*n;
			T* I1         = result + n*n;
			T aux_I_value= -(T)1 / m / m;

			T* ZY        = NULL;
            gpuMemset(m*m, aux_I_value, II);
			gpuMemset(n*n, T(0), I3);
			gpuMemset(n*n, T(0), I1);
			init_II_kernel<T><<<GET_BLOCKS(m),VL_CUDA_NUM_THREADS>>>(II,(T)1/m,m);
			init_I_kernel<T><<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(I3,I1,n);
			for(d = 0;d < L; d++){ //Covariance
                dataOffset    = d*m*n;
				ypOffset      = d*(n*n*iterNum);
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::cov(context,
                                                                    data + dataOffset,
                                                                    aux_Y + ypOffset,II,cov_work,
                                                                    m,n);
                symmetric_gpu(aux_Y + ypOffset,n);
                if(error != VLE_Success) {goto done;}        
            }
			for(d = 0;d < L; d++){ // Trace Norm
				aux_TOffset = d;
				ypOffset = d*(n*n*iterNum);
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::dot(context,
					                                                n*n,
															        aux_Y + ypOffset,ptrdiff_t(1),
															        I1,ptrdiff_t(1),
															        aux_T + aux_TOffset);
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::scal(context,
					                                                 n*n,
					                                                (T)1/aux_T[aux_TOffset],
															         aux_Y + ypOffset,ptrdiff_t(1));
				if(error != vl::VLE_Success) {goto done ;}
			}
			for(d = 0;d < L; d++){
				ZY    = aux_Z + n*n*iterNum*d + n*n; // Z1
				aux_YOffset = n*n*iterNum*d + n*n; //Y1
				ypOffset = d*(n*n*iterNum);
				outputOffset = d*n*(n+1)/2;
				copy_kernel<T><<<GET_BLOCKS(n*n),VL_CUDA_NUM_THREADS>>>(ZY,aux_Y + ypOffset,n*n);
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::axpy(context,
					                                                 n*n,(T)(-1),
															         I3,ptrdiff_t(1),
															         ZY,ptrdiff_t(1));
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
					                                                 'n','n',
															         n,n,n,
															         (T)(-0.5),aux_Y + ypOffset,n,
															         ZY,n,
																	 (T)0,aux_Y + aux_YOffset,n);
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::scal(context,
					                                                 n*n,
					                                                (T)(-0.5),
															         ZY,ptrdiff_t(1));
				if(error != vl::VLE_Success) {goto done ;}
				for(i = 2;i < iterNum;i++){
					ZY = aux_Z + n*n*iterNum*d;
					aux_YOffset = n*n*iterNum*d + i*n*n;
					aux_ZOffset = n*n*iterNum*d + i*n*n;
					aux_YOffset_1 = n*n*iterNum*d + (i-1)*n*n;
					aux_ZOffset_1 = n*n*iterNum*d + (i-1)*n*n;
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																          n,n,n,
																         (T)1,aux_Z + aux_ZOffset_1,n,
																          aux_Y + aux_YOffset_1,n,
																         (T)0,ZY,n);
					if(error != vl::VLE_Success) {goto done ;}
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::axpy(context,
						                                                 n*n,(T)(-1),
																         I3,ptrdiff_t(1),
															             ZY,ptrdiff_t(1));
					if(error != vl::VLE_Success) {goto done ;}
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																          n,n,n,
																         (T)(-0.5),aux_Y + aux_YOffset_1,n,
																          ZY,n,
																         (T)0,aux_Y + aux_YOffset,n);
					if(error != vl::VLE_Success) {goto done ;}
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																         n,n,n,
																        (T)(-0.5),ZY,n,
																         aux_Z + aux_ZOffset_1,n,
																         (T)0,aux_Z + aux_ZOffset,n);
					if(error != vl::VLE_Success) {goto done ;}
				}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                             'n','n',
														              n,n,n,
														             (T)1,aux_Z + aux_ZOffset,n,
															         aux_Y + aux_YOffset,n,
															         (T)0,ZY,n);
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::axpy(context,
						                                             n*n,(T)(-1),
														             I3,ptrdiff_t(1),
															         ZY,ptrdiff_t(1));
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                             'n','n',
														             n,n,n,
														             (T)(-0.5),aux_Y + aux_YOffset,n,
														             ZY,n,
														             (T)0,result,n);
				if(error != vl::VLE_Success) {goto done ;}
				getOutput_gpu(output + outputOffset,result,n);
			}
            
        done:
            return context.passError(error, __func__);
        }
        static vl::ErrorCode
            backward(Context& context,
                     T* derData,
                     T const* data,
                     T const* derOutput,
                     T const* aux_Y,
                     T const* aux_Z,
					 T const* aux_T,
                     size_t height, size_t width, size_t depth, size_t num,
                     int iterNum)
        {
            vl::ErrorCode error;
			ptrdiff_t m = height*width,n = depth,L = num,d,i;
			ptrdiff_t derOutputOffset,dLdCOffset,dataOffset;
			ptrdiff_t aux_YOffset,aux_ZOffset,aux_TOffset;
			ptrdiff_t aux_YOffset_1,aux_ZOffset_1,ypOffset;
			ptrdiff_t dLdYOffset,dLdZOffset;
			ptrdiff_t dLdYOffset_1,dLdZOffset_1;
			ptrdiff_t derDataOffset;
			unsigned int workspaceSize =  (unsigned int)(n*n*(iterNum)*2 + m*n + n*n*L + n*n*8 +m*m);
			T* workspace = (T*)context.getWorkspace(vl::VLDT_GPU , workspaceSize*sizeof(T));
			T* dLdY      = workspace;
			T* dLdZ      = workspace + n*n*(iterNum);
			T* I3        = dLdZ + n*n*(iterNum);
			T* dLdC      = I3 + n*n;
			T* iterMemA  = dLdC + n*n*L;
			T* iterMemB  = iterMemA + n*n;
			T* iterMemC  = iterMemB + n*n;
			T* gradiMem  = iterMemC + n*n;
			T* I_X       = gradiMem + n*n;
			T* II        = I_X + m*n;
			T* ZY        = II + m*m;
			T* YZ        = ZY + n*n;
			T* I1        = YZ + n*n;
			T* ZY_dLdY   = NULL;T* dLdZ_ZY   = NULL;
			T* Z_dLdZ    = NULL;T* Y_dLdY    = NULL;
			T* Z_dLdZ_Z  = NULL;T* Y_dLdY_Y  = NULL;
			T* dLdX      = NULL;T* dLdP      = NULL;
			T  const* P  = NULL;
			T aux_I_value= -(T)1 / m / m;
			T  P_dot_dLdP;
			gpuMemset(n*n*L,(T)0,dLdC);
			gpuMemset(m*m, aux_I_value, II);
			gpuMemset(n*n, T(0), I3);
			init_II_kernel<T><<<GET_BLOCKS(m),VL_CUDA_NUM_THREADS>>>(II,(T)1/m,m);
			init_I_kernel<T><<<GET_BLOCKS(n),VL_CUDA_NUM_THREADS>>>(I3,I1,n);

			for(d = 0;d < L;d++){
				derOutputOffset = d*n*(n+1)/2;
				dLdCOffset      = d*n*n;
				getdLdCfromderOutput_gpu(dLdC + dLdCOffset, derOutput + derOutputOffset, n);
				//symmetric_cpu(dLdC + derOutputOffset,n);
			}
			for(d = 0;d < L;d++){
				derDataOffset = d*m*n;
				aux_TOffset = d;
				aux_YOffset = n*n*iterNum*d + n*n*(iterNum - 1);
				aux_ZOffset = n*n*iterNum*d + n*n*(iterNum - 1);
				dLdYOffset  = n*n*(iterNum - 1);
				dLdZOffset  = n*n*(iterNum - 1);
				dataOffset  = d*m*n;
				dLdCOffset  = d*n*n;
				ypOffset    = n*n*iterNum*d;
				ZY_dLdY     = iterMemA;
				Y_dLdY      = iterMemC;
				P           = aux_Y + ypOffset;
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
					                                                 'n','n',
															          n,n,n,
															         (T)1,aux_Y + aux_YOffset,n,
															          aux_Z + aux_ZOffset,n,
															         (T)0,YZ,n);
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::axpy(context,
					                                                 n*n,(T)(-1),
															         I3,ptrdiff_t(1),
															         YZ,ptrdiff_t(1));
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
					                                                 'n','n',
															         n,n,n,
															         (T)1,aux_Z + aux_ZOffset,n,
															         aux_Y + aux_YOffset,n,
															         (T)0,ZY,n);
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
					                                                 'n','n',
															         n,n,n,
															         (T)(-0.5),ZY,n,
															         dLdC + dLdCOffset,n,
															         (T)0,ZY_dLdY,n);
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
					                                                 'n','n',
															         n,n,n,
															         (T)(-0.5),dLdC + dLdCOffset,n,
															          YZ,n,
															         (T)0,dLdY + dLdYOffset,n);
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::axpy(context,
					                                                 n*n,(T)1,
															         ZY_dLdY,ptrdiff_t(1),
															         dLdY + dLdYOffset,ptrdiff_t(1));
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
					                                                 'n','n',
																	 n,n,n,
															         (T)1,aux_Y + aux_YOffset,n,
															         dLdC + dLdCOffset,n,
															         (T)0,Y_dLdY,n);
				if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
					                                                 'n','n',
															          n,n,n,
															          (T)(-0.5),Y_dLdY,n,
															          aux_Y + aux_YOffset,n,
															          (T)0,dLdZ + dLdZOffset,n);
				if(error != vl::VLE_Success) {goto done ;}
				for(i = iterNum - 1;i > 1;i--){
					dLdYOffset = n*n*(i);
					dLdZOffset = n*n*(i);
					dLdYOffset_1 = n*n*(i-1);
					dLdZOffset_1 = n*n*(i-1);
					aux_YOffset_1  = n*n*iterNum*d + n*n*(i-1);
					aux_ZOffset_1  = n*n*iterNum*d + n*n*(i-1);
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																          n,n,n,
																          (T)1,aux_Y + aux_YOffset_1,n,
																          aux_Z + aux_ZOffset_1,n,
																          (T)0,YZ,n);
					if(error != vl::VLE_Success) {goto done ;}
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::axpy(context,
					                                                     n*n,(T)(-1),
															             I3,ptrdiff_t(1),
															             YZ,ptrdiff_t(1));
					if(error != vl::VLE_Success) {goto done ;}
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																         n,n,n,
																         (T)1,aux_Z+ aux_ZOffset_1,n,
																         aux_Y + aux_YOffset_1,n,
																         (T)0,ZY,n);
					if(error != vl::VLE_Success) {goto done ;}
					Z_dLdZ = iterMemC;Z_dLdZ_Z = iterMemB;ZY_dLdY = iterMemA;
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																         n,n,n,
																         (T)1,aux_Z + aux_ZOffset_1,n,
																         dLdZ + dLdZOffset,n,
																         (T)0,Z_dLdZ,n);
					if(error != vl::VLE_Success) {goto done ;}
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																         n,n,n,
																         (T)(-0.5),Z_dLdZ,n,
																         aux_Z + aux_ZOffset_1,n,
																         (T)0,Z_dLdZ_Z,n);
					if(error != vl::VLE_Success) {goto done ;}
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																         n,n,n,
																         (T)(-0.5),ZY,n,
																         dLdY + dLdYOffset,n,
																         (T)0,ZY_dLdY,n);
					if(error != vl::VLE_Success) {goto done ;}
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																         n,n,n,
																         (T)(-0.5),dLdY + dLdYOffset,n,
																         YZ,n,
																         (T)0,dLdY + dLdYOffset_1,n);
					if(error != vl::VLE_Success) {goto done ;}
					matrixAdd_gpu(dLdY + dLdYOffset_1,Z_dLdZ_Z,ZY_dLdY,T(1),T(1),T(1),n*n);
					Y_dLdY = iterMemC;Y_dLdY_Y = iterMemB;dLdZ_ZY = iterMemA;
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																         n,n,n,
																         (T)1,aux_Y + aux_YOffset_1,n,
																         dLdY + dLdYOffset,n,
																         (T)0,Y_dLdY,n);
					if(error != vl::VLE_Success) {goto done ;}
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																         n,n,n,
																         (T)(-0.5),Y_dLdY,n,
																         aux_Y + aux_YOffset_1,n,
																         (T)0,Y_dLdY_Y,n);
					if(error != vl::VLE_Success) {goto done ;}
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																         n,n,n,
																         (T)(-0.5),dLdZ + dLdZOffset,n,
																         ZY,n,
																         (T)0,dLdZ_ZY,n);
					if(error != vl::VLE_Success) {goto done ;}
					error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                                 'n','n',
																         n,n,n,
											 					         (T)(-0.5),YZ,n,
																         dLdZ + dLdZOffset,n,
																         (T)0,dLdZ + dLdZOffset_1,n);
					if(error != vl::VLE_Success) {goto done ;}
					matrixAdd_gpu(dLdZ + dLdZOffset_1,Y_dLdY_Y,dLdZ_ZY,T(1),T(1),T(1),n*n);
				}
				dLdP  = gradiMem;
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                             'n','n',
														              n,n,n,
															         (T)(-0.5),dLdY + dLdYOffset_1,n,
															          P,n,
															         (T)0,dLdP,n);
				if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
						                                             'n','n',
														             n,n,n,
															         (T)(-0.5),P,n,
															         dLdY + dLdYOffset_1,n,
															         (T)(1),dLdP,n);
				if(error != vl::VLE_Success) {goto done ;}
				matrixAdd_gpu(dLdP,dLdY + dLdYOffset_1,dLdZ + dLdZOffset_1,T(1),T(1.5),T(-0.5),n*n);
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::dot(context,
					                                                n*n,
															        P,ptrdiff_t(1),
															        dLdP,ptrdiff_t(1),
															        &P_dot_dLdP);
			    if(error != vl::VLE_Success) {goto done ;}
				error = vl::impl::blas<vl::VLDT_GPU, dataType>::scal(context,
					                                                 n*n,
					                                                (T)1/aux_T[aux_TOffset],
															         dLdP,ptrdiff_t(1));
				if(error != vl::VLE_Success) {goto done ;}
				traceNormBackward_gpu(dLdP,aux_T[aux_TOffset],P_dot_dLdP,n);
				symmetric_gpu(dLdP,n);
				if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     m,n,m,
                                                                     T(1),II,m,
                                                                     data + dataOffset,m,
                                                                     T(0),I_X,m);
				if(error != vl::VLE_Success) {goto done ;}
                error = vl::impl::blas<vl::VLDT_GPU, dataType>::gemm(context,
                                                                     'n','n',
                                                                     m,n,n,
                                                                     T(2),I_X,m,
                                                                     dLdP,n,
                                                                     T(0),derData + derDataOffset,m);
				if(error != vl::VLE_Success) {goto done ;}

			}

            done:
            return context.passError(error, __func__);
           
        }


    };
} }
template struct vl::impl::imsn_cov<vl::VLDT_GPU, float,vl::VLDT_Float> ;

#ifdef ENABLE_DOUBLE
template struct vl::impl::imsn_cov<vl::VLDT_GPU, double, vl::VLDT_Double> ;
#endif
