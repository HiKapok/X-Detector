// MIT License

// Copyright (c) 2018 Changan Wang

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef KERNEL_ROTATED_PSROI_POOLING_H_
#define KERNEL_ROTATED_PSROI_POOLING_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <cstdint>
#include <tuple>
#include <limits>
#include <iostream>

using tensorflow::TTypes;
using tensorflow::OpKernelContext;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

using KDimSize = std::tuple<int, int, int, int, int, bool>;

#define PI 3.14159265359

template <typename Device, typename T>
struct RotatedPSROIAlignFunctor {
  void operator()(OpKernelContext* context, const Device& d, typename TTypes<T>::ConstFlat inputs, typename TTypes<T>::ConstFlat rois, typename TTypes<int32_t>::ConstFlat orders, const int32_t grid_dim_width, const int32_t grid_dim_height, typename TTypes<T>::Flat pooled_features, typename TTypes<int32_t>::Flat pooled_index, KDimSize dim_info);
};

template <typename Device, typename T>
struct RotatedPSROIAlignGradFunctor {
  void operator()(OpKernelContext* context, const Device& d, typename TTypes<T>::ConstFlat inputs, typename TTypes<T>::ConstFlat rois, typename TTypes<int32_t>::ConstFlat orders, const int32_t grid_dim_width, const int32_t grid_dim_height, typename TTypes<T>::ConstFlat pooled_features_grad, typename TTypes<int32_t>::ConstFlat pooled_index, typename TTypes<T>::Flat grad_output, KDimSize dim_info);
};

#if GOOGLE_CUDA == 1
template <typename T>
struct RotatedPSROIAlignFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const GPUDevice& d, typename TTypes<T>::ConstFlat inputs, typename TTypes<T>::ConstFlat rois, typename TTypes<int32_t>::ConstFlat orders, const int32_t grid_dim_width, const int32_t grid_dim_height, typename TTypes<T>::Flat pooled_features, typename TTypes<int32_t>::Flat pooled_index, KDimSize dim_info);
};
#endif

#if GOOGLE_CUDA == 1
template <typename T>
struct RotatedPSROIAlignGradFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const GPUDevice& d, typename TTypes<T>::ConstFlat inputs, typename TTypes<T>::ConstFlat rois, typename TTypes<int32_t>::ConstFlat orders, const int32_t grid_dim_width, const int32_t grid_dim_height, typename TTypes<T>::ConstFlat pooled_features_grad, typename TTypes<int32_t>::ConstFlat pooled_index, typename TTypes<T>::Flat grad_output, KDimSize dim_info);
};
#endif

#endif // KERNEL_ROTATED_PSROI_POOLING_H_

