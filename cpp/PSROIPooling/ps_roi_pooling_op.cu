// kernel_example.cu.cc
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "ps_roi_pooling_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include <cstdint>
#include <math.h>
#include <stdlib.h>
#include <float.h>

using namespace tensorflow;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// // Define the CUDA kernel.
// template <typename T>
// __global__ void FocalLossCudaKernel(const int size, const T* in, T* out) {
//   for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
//        i += blockDim.x * gridDim.x) {
//     out[i] = 2 * ldg(in + i);
//   }
// }

template <typename T>
__global__ void PSROIPoolingNotNormalized(T* probs, const T* lables, const T * alphas, const T gamma, const int num_rows, const int num_cols) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const int row = tid / num_cols;
  const int col = tid % num_cols;

  if (row < num_rows && col < num_cols) {
    probs[tid] = - ldg(alphas + row) * pow((1-probs[tid]), gamma) * log(probs[tid]) * lables[tid] * 2.;
  }
}

template <typename T>
__global__ void GenerateNormalizedProb(const T* logits, const T* sum_logits,  T* output,
                                       const int num_rows, const int num_cols) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const int row = tid / num_cols;
  const int col = tid % num_cols;

  if (row < num_rows && col < num_cols) {
    output[tid] = logits[tid] / ldg(sum_logits + row);
  }
}

template <typename T>
__global__ void SubtractAndExp(const T* logits, const T* max_logits, T* output, const int num_rows, const int num_cols) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  const int row = tid / num_cols;
  const int col = tid % num_cols;

  if (row < num_rows && col < num_cols) {
    output[tid] = exp(logits[tid] - ldg(max_logits + row));
  }
}

/*
    This version uses n/2 threads --
    it performs the first level of reduction when reading from global memory.
*/
template <class T>
__global__ void sum_reduce(const T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockDim.x < n) mySum += g_idata[i+blockDim.x];

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = mySum;
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void sum_reduce_wrapper(int size, int threads, int blocks, const T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    sum_reduce<T><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size);
}

template void sum_reduce_wrapper<float>(int size, int threads, int blocks, const float *d_idata, float *d_odata);


__device__ float atomicMaxf(float* address, float val)
{
    int *address_as_int =(int*)address;
    int old = *address_as_int, assumed;
    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
        }
    return __int_as_float(old);
}

// As Pavan indicated, you need to initialize your shared memory array. The last block launched may not be a "full" block, if gridDim.x*blockDim.x is greater than elements.
// Note that in this line, even though we are checking that the thread operating (gid) is less than elements, when we add s to gid for indexing into the shared memory we can still index outside of the legitimate values copied into shared memory, in the last block. Therefore we need the shared memory initialization indicated in note 1.
// As you already discovered, your last line was not correct. Each block produces it's own result, and we must combine them somehow. One method you might consider if the number of blocks launched is small (more on this later) is to use atomics. Normally we steer people away from using atomics since they are "costly" in terms of execution time. However, the other option we are faced with is saving the block result in global memory, finishing the kernel, and then possibly launching another kernel to combine the individual block results. If I have launched a large number of blocks initially (say more than 1024) then if I follow this methodology I might end up launching two additional kernels. Thus the consideration of atomics. As indicated, there is no native atomicMax function for floats, but as indicated in the documentation, you can use atomicCAS to generate any arbitrary atomic function, and I have provided an example of that in atomicMaxf which provides an atomic max for float.

// __global__ void max_reduce(const float* const d_array, float* d_max,
//                                               const size_t elements)
// {
//     extern __shared__ float shared[];

//     int tid = threadIdx.x;
//     int gid = (blockDim.x * blockIdx.x) + tid;
//     shared[tid] = -FLOAT_MAX;

//     if (gid < elements)
//         shared[tid] = d_array[gid];
//     __syncthreads();

//     for (unsigned int s=blockDim.x/2; s>0; s>>=1)
//     {
//         if (tid < s && gid < elements)
//             shared[tid] = max(shared[tid], shared[tid + s]);  // 2
//         __syncthreads();
//     }
//     // what to do now?
//     // option 1: save block result and launch another kernel
//     if (tid == 0)
//         d_max[blockIdx.x] = shared[tid]; // 3
//     // option 2: use atomics
//     if (tid == 0)
//       atomicMaxf(d_max, shared[tid]);
// }

// But is running 1024 or more atomic functions (one per block) the best way? Probably not.

// When launching kernels of threadblocks, we really only need to launch enough threadblocks to keep the machine busy. As a rule of thumb we want at least 4-8 warps operating per SM, and somewhat more is probably a good idea. But there's no particular benefit from a machine utilization standpoint to launch thousands of threadblocks initially. If we pick a number like 8 threadblocks per SM, and we have at most, say, 14-16 SMs in our GPU, this gives us a relatively small number of 8*14 = 112 threadblocks. Let's choose 128 (8*16) for a nice round number. There's nothing magical about this, it's just enough to keep the GPU busy. If we make each of these 128 threadblocks do additional work to solve the whole problem, we can then leverage our use of atomics without (perhaps) paying too much of a penalty for doing so, and avoid multiple kernel launches. So how would this look?:

// With this modified kernel, when creating the kernel launch, we are not deciding how many threadblocks to launch based on the overall data size (elements). Instead we are launching a fixed number of blocks (say, 128, you can modify this number to find out what runs fastest), and letting each threadblock (and thus the entire grid) loop through memory, computing partial max operations on each element in shared memory. Then, in the line marked with comment 1, we must re-set the gid variable to it's initial value. This is actually unnecessary and the block reduction loop code can be further simplified if we guarantee that the size of the grid (gridDim.x*blockDim.x) is less than elements, which is not difficult to do at kernel launch.

// Note that when using this atomic method, it's necessary to initialize the result (*d_max in this case) to an appropriate value, like -FLOAT_MAX.

// Again, we normally steer people way from atomic usage, but in this case, it's worth considering if we carefully manage it, and it allows us to save the overhead of an additional kernel launch.

// For a ninja-level analysis of how to do fast parallel reductions, take a look at Mark Harris' excellent whitepaper which is available with the relevant CUDA sample.

__global__ void max_reduce(const float* const d_array, float* d_max,
                                              const size_t elements)
{
  extern __shared__ float shared[];

  int tid = threadIdx.x;
  int gid = (blockDim.x * blockIdx.x) + tid;
  shared[tid] = -FLT_MAX;

  while (gid < elements) {
    shared[tid] = max(shared[tid], d_array[gid]);
    gid += gridDim.x*blockDim.x;
  }
  __syncthreads();
  gid = (blockDim.x * blockIdx.x) + tid;  // 1
  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
    if (tid < s && gid < elements)
      shared[tid] = max(shared[tid], shared[tid + s]);
    __syncthreads();
  }

  if (tid == 0)
    atomicMaxf(d_max, shared[0]);
}

template <class T>
void max_reduce_wrapper(int col_size, int threads, int blocks, const T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    max_reduce<<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, col_size);
}

template void max_reduce_wrapper<float>(int size, int threads, int blocks, const float *d_idata, float *d_odata);


template <typename T>
void PSROIPoolingFunctorGPU<T>::operator()(OpKernelContext* context, const GPUDevice& d, typename TTypes<T>::ConstMatrix logits, \
        typename TTypes<T>::ConstMatrix labels, typename TTypes<T>::Matrix softmax, typename TTypes<T>::Vec alpha, \
        typename TTypes<T>::ConstScalar gamma, typename TTypes<T>::Vec focal_loss) {
  // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.

    // CudaLaunchConfig config = GetCudaLaunchConfig(updates_size, d);
    // ScatterOpCustomKernel<T, Index, op>
    //       <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
    //           params.data(), updates.data(), indices.data(),
    //           first_dim_size, updates_size, indices_size);
    //   return -1;

    const int rows = logits.dimension(0);
    const int cols = logits.dimension(1);

    if (logits.size() > 0) {
      const int numBlocks = 128;
      const int numThreads = Eigen::divup(rows * cols, numBlocks);

      Tensor max_logits;
      Tensor temp_along_class;
      Tensor sum_logits;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            TensorShape({rows}), &max_logits));

      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            TensorShape({numBlocks}), &temp_along_class));

      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            TensorShape({rows}), &sum_logits));

      for(int row_index = 0;row_index < rows;++row_index)
        max_reduce_wrapper(cols, numThreads, numBlocks, \
          reinterpret_cast<const T*>(logits.data()) + row_index*cols, \
          reinterpret_cast<T*>(max_logits.flat<T>().data()) + row_index);

      SubtractAndExp<<<numBlocks, numThreads, 0, d.stream()>>>(reinterpret_cast<const T*>(logits.data()), \
                                                reinterpret_cast<const T*>(max_logits.flat<T>().data()), \
                                                reinterpret_cast<T*>(softmax.data()), rows, cols);

      for(int row_index = 0;row_index < rows;++row_index){
        sum_reduce_wrapper(cols, numThreads, numBlocks, \
          reinterpret_cast<const T*>(softmax.data()) + row_index*cols, \
          reinterpret_cast<T*>(temp_along_class.flat<T>().data()));
        sum_reduce_wrapper(cols, numBlocks, 1, \
          reinterpret_cast<const T*>(temp_along_class.flat<T>().data()), \
          reinterpret_cast<T*>(sum_logits.flat<T>().data()) + row_index);
      }


      GenerateNormalizedProb<<<numBlocks, numThreads, 0, d.stream()>>>(reinterpret_cast<const T*>(softmax.data()), \
                                        reinterpret_cast<const T*>(sum_logits.flat<T>().data()), \
                                        reinterpret_cast<T*>(softmax.data()), \
                                        rows, cols);
      PSROIPoolingNotNormalized<<<numBlocks, numThreads, 0, d.stream()>>>(reinterpret_cast<T*>(softmax.data()), \
                                        reinterpret_cast<const T*>(labels.data()), \
                                        reinterpret_cast<const T*>(alpha.data()), gamma(0), rows, cols);

      for(int row_index = 0;row_index < rows;++row_index){
        sum_reduce_wrapper(cols, numThreads, numBlocks, \
          reinterpret_cast<const T*>(softmax.data()) + row_index*cols, \
          reinterpret_cast<T*>(temp_along_class.flat<T>().data()));
        sum_reduce_wrapper(cols, numBlocks, 1, \
          reinterpret_cast<const T*>(temp_along_class.flat<T>().data()), \
          reinterpret_cast<T*>(focal_loss.data()) + row_index);
      }
  }
}

template struct PSROIPoolingFunctorGPU<float>;
// #define DEFINE_GPU_SPECS(T)   \
//   template struct PSROIPoolingFunctorGPU<T>;

// TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#endif  // GOOGLE_CUDA
