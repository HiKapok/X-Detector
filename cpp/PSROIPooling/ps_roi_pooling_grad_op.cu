// kernel_example.cu.cc
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "ps_roi_pooling_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"

using namespace tensorflow;

// Define the CUDA kernel.
template <typename T>
__global__ void PSROIPoolingGradCudaKernel(const int size, const T* in, T* out) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += blockDim.x * gridDim.x) {
    out[i] = static_cast<T>(2) * ldg(in + i);
  }
}


template <typename T>
void PSROIPoolingGradFunctorGPU<T>::operator()(const GPUDevice& d, typename TTypes<T>::ConstMatrix logits, typename TTypes<T>::ConstVec labels,
  typename TTypes<T>::Matrix softmax, typename TTypes<T>::Vec alpha, typename TTypes<T>::ConstScalar gamma, typename TTypes<T>::ConstVec focal_loss,
  typename TTypes<T>::Matrix grads) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  int block_count = 1024;
  int thread_per_block = 20;
  PSROIPoolingGradCudaKernel<T>
      <<<block_count, thread_per_block, 0, d.stream()>>>(1, nullptr, nullptr);
}

template struct PSROIPoolingGradFunctorGPU<float>;

// #define DEFINE_GPU_SPECS(T)   \
//   template struct PSROIPoolingGradFunctorGPU<T>;

// TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#endif  // GOOGLE_CUDA
