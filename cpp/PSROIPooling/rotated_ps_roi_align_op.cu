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
#if GOOGLE_CUDA == 1
#define EIGEN_USE_GPU
#include "rotated_ps_roi_align_op.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;

#include <cstdint>
#include <cmath>
#include <cfloat>

// Define the CUDA kernel.
template <typename T>
__global__ void RotatedPSROIAlignCudaKernel(CudaLaunchConfig config, const T * inputs, const T * rois, const int32_t * orders, T * pooled_features, int32_t * pooled_index, const int32_t grid_dim_width, const int32_t grid_dim_height, const int batch_size, const int num_channals, const int map_height, const int map_width, const int num_rois, const bool using_max_pool) {

  const int32_t grid_size = grid_dim_width * grid_dim_height;
  const int32_t bank_size = num_channals / grid_size;

  CUDA_1D_KERNEL_LOOP(worker_index, config.virtual_thread_count) {
    // image_index * roi_index * channal_pos_remainder * row_index * col_index
    const int32_t position_index = (worker_index % num_channals) / bank_size;
    const int32_t row_index = position_index / grid_dim_width;
    const int32_t col_index = position_index % grid_dim_width;
    // position of the channal of pooled feature
    // position of the channal in the bank of feature map
    const int32_t channal_pos_remainder = worker_index % bank_size;
    const int32_t pool_index = worker_index / num_channals;
    const int32_t image_index = pool_index / num_rois;
    const int32_t roi_index = pool_index % num_rois;

    const T * roi_to_pool = rois + (image_index * num_rois + roi_index) * 8;
    const int32_t * roi_order = orders + image_index * num_rois + roi_index;

    const T * feature_map_to_pool = inputs + (image_index * num_channals + (position_index % grid_size) * bank_size + channal_pos_remainder) * map_height * map_width;
    T * pooled_features_start = pooled_features + image_index * (num_rois * num_channals) + roi_index * num_channals + (position_index % grid_size) * bank_size + channal_pos_remainder;
    int32_t * pooled_index_start = pooled_index + image_index * (num_rois * num_channals) + roi_index * num_channals + (position_index % grid_size) * bank_size + channal_pos_remainder;

    int32_t order = ldg(roi_order) < 0 ? 0 : ldg(roi_order) * 2;

    T roi_y0 = static_cast<T>(ldg(roi_to_pool + (order++) % 8) * map_height);
    T roi_x0 = static_cast<T>(ldg(roi_to_pool + (order++) % 8) * map_width);
    T roi_y1 = static_cast<T>(ldg(roi_to_pool + (order++) % 8) * map_height);
    T roi_x1 = static_cast<T>(ldg(roi_to_pool + (order++) % 8) * map_width);
    T roi_y2 = static_cast<T>(ldg(roi_to_pool + (order++) % 8) * map_height);
    T roi_x2 = static_cast<T>(ldg(roi_to_pool + (order++) % 8) * map_width);
    T roi_y3 = static_cast<T>(ldg(roi_to_pool + (order++) % 8) * map_height);
    T roi_x3 = static_cast<T>(ldg(roi_to_pool + (order++) % 8) * map_width);


    double len0 = static_cast<double>((roi_y1 - roi_y0) * (roi_y1 - roi_y0) + (roi_x1 - roi_x0) * (roi_x1 - roi_x0));
    double len1 = static_cast<double>((roi_y2 - roi_y1) * (roi_y2 - roi_y1) + (roi_x2 - roi_x1) * (roi_x2 - roi_x1));
    double len2 = static_cast<double>((roi_y3 - roi_y2) * (roi_y3 - roi_y2) + (roi_x3 - roi_x2) * (roi_x3 - roi_x2));
    double len3 = static_cast<double>((roi_y0 - roi_y3) * (roi_y0 - roi_y3) + (roi_x0 - roi_x3) * (roi_x0 - roi_x3));
    double cross_len0 = static_cast<double>((roi_y0 - roi_y2) * (roi_y0 - roi_y2) + (roi_x0 - roi_x2) * (roi_x0 - roi_x2));
    double cross_len1 = static_cast<double>((roi_y3 - roi_y1) * (roi_y3 - roi_y1) + (roi_x3 - roi_x1) * (roi_x3 - roi_x1));

    order = ldg(roi_order) < 0 ? (len0 + len2 > len1 + len3 ? 1 : 0) : 0;
    // fix ROI
    if(len0 < std::numeric_limits<T>::min() || len1 < std::numeric_limits<T>::min() || len2 < std::numeric_limits<T>::min() || len3 < std::numeric_limits<T>::min()){
    // not check convex for faster speed
    //if(is_convex(roi_to_pool)){
      *pooled_features_start = static_cast<T>(0);
      *pooled_index_start = static_cast<T>(0);
      continue;
    }

    T roi_y0_order = (order == 0) ? roi_y0 : roi_y1;
    T roi_x0_order = (order == 0) ? roi_x0 : roi_x1;
    T roi_y1_order = (order == 0) ? roi_y1 : roi_y2;
    T roi_x1_order = (order == 0) ? roi_x1 : roi_x2;
    T roi_y2_order = (order == 0) ? roi_y2 : roi_y3;
    T roi_x2_order = (order == 0) ? roi_x2 : roi_x3;
    T roi_y3_order = (order == 0) ? roi_y3 : roi_y0;
    T roi_x3_order = (order == 0) ? roi_x3 : roi_x0;

    T y_step_left = (roi_y3_order - roi_y0_order)/(grid_dim_height * 1.);
    T y_step_right = (roi_y2_order - roi_y1_order)/(grid_dim_height * 1.);
    T x_step_top = (roi_x1_order - roi_x0_order)/(grid_dim_width * 1.);
    T x_step_bottom = (roi_x2_order - roi_x3_order)/(grid_dim_width * 1.);

    T left_y1 = (roi_y0_order + row_index * y_step_left);
    T right_y1 = (roi_y1_order + row_index * y_step_right);
    T left_y2 = (roi_y0_order + (row_index + 1.) * y_step_left);
    T right_y2 = (roi_y1_order + (row_index + 1.) * y_step_right);

    T left_top_y = left_y1 + col_index * (right_y1 - left_y1)/(grid_dim_width);
    T right_top_y = left_y1 + (col_index + 1.) * (right_y1 - left_y1)/(grid_dim_width);
    T left_bottom_y = left_y2 + col_index * (right_y2 - left_y2)/(grid_dim_width);
    T right_bottom_y = left_y2 + (col_index + 1.) * (right_y2 - left_y2)/(grid_dim_width);

    T top_x1 = (roi_x0_order + col_index * x_step_top);
    T bottom_x1 = (roi_x3_order + col_index * x_step_bottom);
    T top_x2 = (roi_x0_order + (col_index + 1.) * x_step_top);
    T bottom_x2 = (roi_x3_order + (col_index + 1.) * x_step_bottom);

    T left_top_x = top_x1 + row_index * (bottom_x1 - top_x1)/(grid_dim_height);
    T left_bottom_x = top_x1 + (row_index + 1.) * (bottom_x1 - top_x1)/(grid_dim_height);
    T right_top_x = top_x2 + row_index * (bottom_x2 - top_x2)/(grid_dim_height);
    T right_bottom_x = top_x2 + (row_index + 1.) * (bottom_x2 - top_x2)/(grid_dim_height);

    float pool_bin_width = static_cast<float>(tf_max(tf_min(fabsf(right_top_x - left_top_x), fabsf(right_top_y - left_top_y)), tf_min(fabsf(right_bottom_x - left_bottom_x), fabsf(right_bottom_y - left_bottom_y))));
    float pool_bin_height = static_cast<float>(tf_max(tf_min(fabsf(left_bottom_x - left_top_x), fabsf(left_bottom_y - left_top_y)), tf_min(fabsf(right_bottom_x - right_top_x), fabsf(right_bottom_y - right_top_y))));
    int32_t num_elem_width = static_cast<int32_t>(pool_bin_width) + 1;
    int32_t num_elem_height = static_cast<int32_t>(pool_bin_height) + 1;

    T grid_y_step_left = (left_bottom_y - left_top_y)/(num_elem_height + 1.);
    T grid_y_step_right = (right_bottom_y - right_top_y)/(num_elem_height + 1.);
    T grid_x_step_top = (right_top_x - left_top_x)/(num_elem_width + 1.);
    T grid_x_step_bottom = (right_bottom_x - left_bottom_x)/(num_elem_width + 1.);

    int32_t max_pool_ind = 0;
    //T max_elem = std::numeric_limits<T>::lowest();
    T max_or_acc_elem = using_max_pool ? std::numeric_limits<T>::lowest() : static_cast<T>(0);

    for(int32_t pool_h = 0; pool_h < num_elem_height; ++pool_h){
      for(int32_t pool_w = 0; pool_w < num_elem_width; ++pool_w){
        //std::cout << "col_to_pool: " << col_to_pool << " row_to_pool: " << row_to_pool << std::endl;
        T col_to_pool = (left_top_x + (pool_w + 1.) * grid_x_step_top + left_bottom_x + (pool_w + 1.) * grid_x_step_bottom) / 2.;
        T row_to_pool = (left_top_y + (pool_h + 1.) * grid_y_step_left + right_top_y + (pool_h + 1.) * grid_y_step_right) / 2.;

        int32_t int_col_to_pool = static_cast<int32_t>(col_to_pool);
        int32_t int_row_to_pool = static_cast<int32_t>(row_to_pool);
        float float_col_to_pool = col_to_pool - int_col_to_pool;
        float float_row_to_pool = row_to_pool - int_row_to_pool;

        int32_t current_switch_ind = num_elem_width * pool_h + pool_w;
        //std::cout << "current_switch_ind: " << current_switch_ind << std::endl;
        T temp_value = static_cast<T>((1. - float_col_to_pool) * (1. - float_row_to_pool) * ldg(feature_map_to_pool + int_row_to_pool * map_width + int_col_to_pool) +
                                  (1. - float_col_to_pool) * float_row_to_pool * ldg(feature_map_to_pool + tf_min(int_row_to_pool + 1, map_height - 1) * map_width + int_col_to_pool) +
                                  float_col_to_pool * (1. - float_row_to_pool) * ldg(feature_map_to_pool + int_row_to_pool * map_width + tf_min(int_col_to_pool + 1, map_width - 1)) +
                                  float_col_to_pool * float_row_to_pool * ldg(feature_map_to_pool + tf_min(int_row_to_pool + 1, map_height - 1) * map_width + tf_min(int_col_to_pool + 1, map_width - 1)));
        if(using_max_pool){
          if(max_or_acc_elem < temp_value){
            max_or_acc_elem = temp_value;
            max_pool_ind = current_switch_ind;
          }
        }else{
          max_or_acc_elem += temp_value;
        }
      }
    }

    if(!using_max_pool) max_or_acc_elem /= static_cast<T>(num_elem_height * num_elem_width);
    *pooled_features_start = max_or_acc_elem;
    *pooled_index_start = using_max_pool ? max_pool_ind : static_cast<T>(0);
  }
}

template <typename T>
void RotatedPSROIAlignFunctor<GPUDevice, T>::operator()(OpKernelContext* context, const GPUDevice& d, typename TTypes<T>::ConstFlat inputs, typename TTypes<T>::ConstFlat rois, typename TTypes<int32_t>::ConstFlat orders, const int32_t grid_dim_width, const int32_t grid_dim_height, typename TTypes<T>::Flat pooled_features, typename TTypes<int32_t>::Flat pooled_index, KDimSize dim_info) {

    int batch_size = 0;
    int num_channals = 0;
    int map_height = 0;
    int map_width = 0;
    int num_rois = 0;
    bool using_max_pool = false;

    std::tie(batch_size, num_channals, map_height, map_width, num_rois, using_max_pool) = dim_info;

    CudaLaunchConfig config = GetCudaLaunchConfig(batch_size * num_rois * num_channals, d);
    RotatedPSROIAlignCudaKernel <<<config.block_count,
                        config.thread_per_block, 0, d.stream()>>> (config, inputs.data(), rois.data(), orders.data(), pooled_features.data(), pooled_index.data(), grid_dim_width, grid_dim_height, batch_size, num_channals, map_height, map_width, num_rois, using_max_pool);

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
      fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
      exit( -1 );
    }
}

template struct RotatedPSROIAlignFunctor<GPUDevice, float>;
// #define DEFINE_GPU_SPECS(T)   \
//   template struct RotatedPSROIAlignFunctorGPU<T>;

// TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#endif  // GOOGLE_CUDA
