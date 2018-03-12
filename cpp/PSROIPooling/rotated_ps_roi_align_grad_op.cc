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
#include "rotated_ps_roi_align_op.h"
#include "common.h"
#include "work_sharder.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>

using namespace tensorflow;

// the inputs should have format NCHW, which is faster on GPUs
REGISTER_OP("RotatedPsRoiAlignGrad")
    .Attr("T: {float}")
    .Attr("grid_dim_width: int")
    .Attr("grid_dim_height: int")
    .Attr("pool_method: string")
    .Input("inputs: T")
    .Input("rois: T")
    .Input("orders: int32")
    .Input("pooled_features_grad: T")
    .Input("pooled_index: int32")
    .Output("grad_output: T")
    .Doc(R"doc(
        RotatedPsRoiAlignGrad is the Gradient op of RotatedPsRoiAlign.
        The input rois to be pooled must in format [y0, x0, y1, x1, y2, x2, y3, x3] which is four vertexes defining quadrilateral in clockwise order and each element must be in range [0, 1.].
        The input orders define which point is the first one, each element must be in range [-1, 4). The order will be determined to be the first vertex of the shorter side if given -1.
        The caller must make sure that all rois is valid (has a intersect region (one pixel at least) with the window [0., 0., 0., 1., 1., 1., 1., 0.]).
        )doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });

// CPU specialization of actual computation.
// template <typename T>
// struct RotatedPSROIAlignGradFunctor<CPUDevice, T> {
//   void operator()(OpKernelContext* context, const CPUDevice& d, typename TTypes<T>::ConstFlat inputs, typename TTypes<T>::ConstFlat rois, typename TTypes<int32_t>::ConstFlat orders, const int32_t grid_dim_width, const int32_t grid_dim_height, typename TTypes<T>::ConstFlat pooled_features_grad, typename TTypes<int32_t>::ConstFlat pooled_index, typename TTypes<T>::Flat grad_output, KDimSize dim_info) {

//     int batch_size = 0;
//     int num_channals = 0;
//     int map_height = 0;
//     int map_width = 0;
//     int num_rois = 0;
//     bool using_max_pool = false;

//     std::tie(batch_size, num_channals, map_height, map_width, num_rois, using_max_pool) = dim_info;
//     grad_output = grad_output.setZero();

//     auto pooling_grad_routine = [&rois, &orders, &pooled_features_grad, &pooled_index, &grad_output, grid_dim_width, grid_dim_height, batch_size, num_channals, map_height, map_width, num_rois, using_max_pool](int64_t start, int64_t limit){
//       const int32_t grid_size = grid_dim_width * grid_dim_height;
//       const int32_t bank_size = num_channals / grid_size;
//       for (int64_t worker_index = start; worker_index < limit; ++worker_index){
//         // image_index * roi_index * channal_pos_remainder * row_index * col_index
//         const int32_t position_index = (worker_index % num_channals) / bank_size;
//         const int32_t row_index = position_index / grid_dim_width;
//         const int32_t col_index = position_index % grid_dim_width;
//         // position of the channal of pooled feature
//         // position of the channal in the bank of feature map
//         const int32_t channal_pos_remainder = worker_index % bank_size;
//         const int32_t pool_index = worker_index / num_channals;
//         const int32_t image_index = pool_index / num_rois;
//         const int32_t roi_index = pool_index % num_rois;

//         const T * roi_to_pool = rois.data() + (image_index * num_rois + roi_index) * 8;
//         const int32_t * roi_order = orders.data() + image_index * num_rois + roi_index;

//         volatile T * grad_output_start = reinterpret_cast<volatile T*>(grad_output.data() + (image_index * num_channals + position_index * bank_size + channal_pos_remainder) * map_height * map_width);
//         const T * pooled_features_start = pooled_features_grad.data() + worker_index;
//         const int32_t * pooled_index_start = pooled_index.data() + worker_index;

//         int32_t order = *roi_order < 0 ? 0 : *roi_order * 2;

//         T roi_y0 = static_cast<T>(roi_to_pool[(order++) % 8] * map_height);
//         T roi_x0 = static_cast<T>(roi_to_pool[(order++) % 8] * map_width);
//         T roi_y1 = static_cast<T>(roi_to_pool[(order++) % 8] * map_height);
//         T roi_x1 = static_cast<T>(roi_to_pool[(order++) % 8] * map_width);
//         T roi_y2 = static_cast<T>(roi_to_pool[(order++) % 8] * map_height);
//         T roi_x2 = static_cast<T>(roi_to_pool[(order++) % 8] * map_width);
//         T roi_y3 = static_cast<T>(roi_to_pool[(order++) % 8] * map_height);
//         T roi_x3 = static_cast<T>(roi_to_pool[(order++) % 8] * map_width);

//         double len0 = static_cast<double>((roi_y1 - roi_y0) * (roi_y1 - roi_y0) + (roi_x1 - roi_x0) * (roi_x1 - roi_x0));
//         double len1 = static_cast<double>((roi_y2 - roi_y1) * (roi_y2 - roi_y1) + (roi_x2 - roi_x1) * (roi_x2 - roi_x1));
//         double len2 = static_cast<double>((roi_y3 - roi_y2) * (roi_y3 - roi_y2) + (roi_x3 - roi_x2) * (roi_x3 - roi_x2));
//         double len3 = static_cast<double>((roi_y0 - roi_y3) * (roi_y0 - roi_y3) + (roi_x0 - roi_x3) * (roi_x0 - roi_x3));
//         double cross_len0 = static_cast<double>((roi_y0 - roi_y2) * (roi_y0 - roi_y2) + (roi_x0 - roi_x2) * (roi_x0 - roi_x2));
//         double cross_len1 = static_cast<double>((roi_y3 - roi_y1) * (roi_y3 - roi_y1) + (roi_x3 - roi_x1) * (roi_x3 - roi_x1));

//         order = *roi_order < 0 ? (len0 + len2 > len1 + len3 ? 1 : 0) : 0;
//         // fix ROI
//         if(len0 < std::numeric_limits<T>::min() || len1 < std::numeric_limits<T>::min() || len2 < std::numeric_limits<T>::min() || len3 < std::numeric_limits<T>::min()){
//         // not check convex for faster speed
//         //if(is_convex(roi_to_pool)){
//           continue;
//         }

//         T roi_y0_order = (order == 0) ? roi_y0 : roi_y1;
//         T roi_x0_order = (order == 0) ? roi_x0 : roi_x1;
//         T roi_y1_order = (order == 0) ? roi_y1 : roi_y2;
//         T roi_x1_order = (order == 0) ? roi_x1 : roi_x2;
//         T roi_y2_order = (order == 0) ? roi_y2 : roi_y3;
//         T roi_x2_order = (order == 0) ? roi_x2 : roi_x3;
//         T roi_y3_order = (order == 0) ? roi_y3 : roi_y0;
//         T roi_x3_order = (order == 0) ? roi_x3 : roi_x0;

//         T y_step_left = (roi_y3_order - roi_y0_order)/(grid_dim_height * 1.);
//         T y_step_right = (roi_y2_order - roi_y1_order)/(grid_dim_height * 1.);
//         T x_step_top = (roi_x1_order - roi_x0_order)/(grid_dim_width * 1.);
//         T x_step_bottom = (roi_x2_order - roi_x3_order)/(grid_dim_width * 1.);

//         T left_y1 = (roi_y0_order + row_index * y_step_left);
//         T right_y1 = (roi_y1_order + row_index * y_step_right);
//         T left_y2 = (roi_y0_order + (row_index + 1.) * y_step_left);
//         T right_y2 = (roi_y1_order + (row_index + 1.) * y_step_right);

//         T left_top_y = left_y1 + col_index * (right_y1 - left_y1)/(grid_dim_width);
//         T right_top_y = left_y1 + (col_index + 1.) * (right_y1 - left_y1)/(grid_dim_width);
//         T left_bottom_y = left_y2 + col_index * (right_y2 - left_y2)/(grid_dim_width);
//         T right_bottom_y = left_y2 + (col_index + 1.) * (right_y2 - left_y2)/(grid_dim_width);

//         T top_x1 = (roi_x0_order + col_index * x_step_top);
//         T bottom_x1 = (roi_x3_order + col_index * x_step_bottom);
//         T top_x2 = (roi_x0_order + (col_index + 1.) * x_step_top);
//         T bottom_x2 = (roi_x3_order + (col_index + 1.) * x_step_bottom);

//         T left_top_x = top_x1 + row_index * (bottom_x1 - top_x1)/(grid_dim_height);
//         T left_bottom_x = top_x1 + (row_index + 1.) * (bottom_x1 - top_x1)/(grid_dim_height);
//         T right_top_x = top_x2 + row_index * (bottom_x2 - top_x2)/(grid_dim_height);
//         T right_bottom_x = top_x2 + (row_index + 1.) * (bottom_x2 - top_x2)/(grid_dim_height);

//         float pool_bin_width = static_cast<float>(std::max(std::min(std::abs(right_top_x - left_top_x), std::abs(right_top_y - left_top_y)), std::min(std::abs(right_bottom_x - left_bottom_x), std::abs(right_bottom_y - left_bottom_y))));
//         float pool_bin_height = static_cast<float>(std::max(std::min(std::abs(left_bottom_x - left_top_x), std::abs(left_bottom_y - left_top_y)), std::min(std::abs(right_bottom_x - right_top_x), std::abs(right_bottom_y - right_top_y))));
//         int32_t num_elem_width = static_cast<int32_t>(pool_bin_width) + 1;
//         int32_t num_elem_height = static_cast<int32_t>(pool_bin_height) + 1;

//         T grid_y_step_left = (left_bottom_y - left_top_y)/(num_elem_height + 1.);
//         T grid_y_step_right = (right_bottom_y - right_top_y)/(num_elem_height + 1.);
//         T grid_x_step_top = (right_top_x - left_top_x)/(num_elem_width + 1.);
//         T grid_x_step_bottom = (right_bottom_x - left_bottom_x)/(num_elem_width + 1.);

//         if(using_max_pool){
//           const int32_t pool_h = *pooled_index_start / num_elem_width;
//           const int32_t pool_w = *pooled_index_start % num_elem_width;

//             T col_to_pool = (left_top_x + (pool_w + 1.) * grid_x_step_top + left_bottom_x + (pool_w + 1.) * grid_x_step_bottom) / 2.;
//             T row_to_pool = (left_top_y + (pool_h + 1.) * grid_y_step_left + right_top_y + (pool_h + 1.) * grid_y_step_right) / 2.;

//           int32_t int_col_to_pool = static_cast<int32_t>(col_to_pool);
//           int32_t int_row_to_pool = static_cast<int32_t>(row_to_pool);
//           float float_col_to_pool = col_to_pool - int_col_to_pool;
//           float float_row_to_pool = row_to_pool - int_row_to_pool;

//           const T grad_in = *pooled_features_start;
//           atomic_float_add(grad_output_start + int_row_to_pool * map_width + int_col_to_pool, static_cast<T>((1. - float_col_to_pool) * (1. - float_row_to_pool) * grad_in));
//           atomic_float_add(grad_output_start + std::min(int_row_to_pool + 1, map_height - 1) * map_width + int_col_to_pool, static_cast<T>((1. - float_col_to_pool) * float_row_to_pool * grad_in));
//           atomic_float_add(grad_output_start + int_row_to_pool * map_width + std::min(int_col_to_pool + 1, map_width - 1), static_cast<T>(float_col_to_pool * (1. - float_row_to_pool) * grad_in));
//           atomic_float_add(grad_output_start + std::min(int_row_to_pool + 1, map_height - 1) * map_width + std::min(int_col_to_pool + 1, map_width - 1), static_cast<T>(float_col_to_pool * float_row_to_pool * grad_in));
//         }else{
//           const T grad_in = *pooled_features_start / static_cast<T>(num_elem_width * num_elem_height);
//           for(int32_t pool_h = 0; pool_h < num_elem_height; ++pool_h){
//             for(int32_t pool_w = 0; pool_w < num_elem_width; ++pool_w){
//                 T col_to_pool = (left_top_x + (pool_w + 1.) * grid_x_step_top + left_bottom_x + (pool_w + 1.) * grid_x_step_bottom) / 2.;
//                 T row_to_pool = (left_top_y + (pool_h + 1.) * grid_y_step_left + right_top_y + (pool_h + 1.) * grid_y_step_right) / 2.;

//               int32_t int_col_to_pool = static_cast<int32_t>(col_to_pool);
//               int32_t int_row_to_pool = static_cast<int32_t>(row_to_pool);
//               float float_col_to_pool = col_to_pool - int_col_to_pool;
//               float float_row_to_pool = row_to_pool - int_row_to_pool;

//               atomic_float_add(grad_output_start + int_row_to_pool * map_width + int_col_to_pool, static_cast<T>((1. - float_col_to_pool) * (1. - float_row_to_pool) * grad_in));
//               atomic_float_add(grad_output_start + std::min(int_row_to_pool + 1, map_height - 1) * map_width + int_col_to_pool, static_cast<T>((1. - float_col_to_pool) * float_row_to_pool * grad_in));
//               atomic_float_add(grad_output_start + int_row_to_pool * map_width + std::min(int_col_to_pool + 1, map_width - 1), static_cast<T>(float_col_to_pool * (1. - float_row_to_pool) * grad_in));
//               atomic_float_add(grad_output_start + std::min(int_row_to_pool + 1, map_height - 1) * map_width + std::min(int_col_to_pool + 1, map_width - 1), static_cast<T>(float_col_to_pool * float_row_to_pool * grad_in));
//             }
//         }
//         }
//       }
//     };

//     const DeviceBase::CpuWorkerThreads& worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
//     // one worker for one position in each ROI
//     const int64_t shard_cost = 4 * map_height * map_width / grid_dim_width / grid_dim_height / 4;
//     Shard(worker_threads.num_threads, worker_threads.workers,
//           pooled_features_grad.size(), shard_cost, pooling_grad_routine);
//   }
// };

// // calculate gradients from input side
// // the result of this kernel is same as the above kernel which is calculate gradients from the output side
// // the different is that this kernel don't need synchronous gradients of the same input cell
// // but the drawback of this kernel is that more threads scheduling may be occurred due to the larger input feature map size compared with output feature map
// // you can choose any one to use depends on the relative overhead between the scheduling and atomic sync operation
template <typename T>
struct RotatedPSROIAlignGradFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const CPUDevice& d, typename TTypes<T>::ConstFlat inputs, typename TTypes<T>::ConstFlat rois, typename TTypes<int32_t>::ConstFlat orders, const int32_t grid_dim_width, const int32_t grid_dim_height, typename TTypes<T>::ConstFlat pooled_features_grad, typename TTypes<int32_t>::ConstFlat pooled_index, typename TTypes<T>::Flat grad_output, KDimSize dim_info) {
    int batch_size = 0;
    int num_channals = 0;
    int map_height = 0;
    int map_width = 0;
    int num_rois = 0;
    bool using_max_pool = false;

    std::tie(batch_size, num_channals, map_height, map_width, num_rois, using_max_pool) = dim_info;

    grad_output = grad_output.setZero();

    auto pooling_grad_routine = [&rois, &orders, &pooled_features_grad, &pooled_index, &grad_output, grid_dim_width, grid_dim_height, batch_size, num_channals, map_height, map_width, num_rois, using_max_pool](int64_t start, int64_t limit){
      const int32_t grid_size = grid_dim_width * grid_dim_height;
      const int32_t bank_size = num_channals/grid_size;
      for (int64_t worker_index = start; worker_index < limit; ++worker_index){
        const int32_t cur_image_index = worker_index / (num_channals * map_height * map_width);
        const int32_t cur_channal_index = (worker_index % (num_channals * map_height * map_width)) / (map_height * map_width);
        const int32_t offset_on_map = worker_index % (map_height * map_width);
        const int32_t col_on_map = offset_on_map % map_width;
        const int32_t row_on_map = offset_on_map / map_width;

        T * grad_to_fill = reinterpret_cast<T*>(grad_output.data() + worker_index);

        for(int roi_index = 0;roi_index < num_rois;++roi_index){
          const T * roi_to_pool = rois.data() + (cur_image_index * num_rois + roi_index) * 8;
          const int32_t * roi_order = orders.data() + cur_image_index * num_rois + roi_index;

          const T pooled_features_grad_in = *(pooled_features_grad.data() + cur_image_index * (num_rois * num_channals) + roi_index * num_channals + cur_channal_index);
          const int32_t pooled_max_index = *(pooled_index.data() + cur_image_index * (num_rois * num_channals) + roi_index * num_channals + cur_channal_index);

          const int32_t row_index = (cur_channal_index / bank_size) / grid_dim_width;
          const int32_t col_index = (cur_channal_index / bank_size) % grid_dim_width;

          int32_t order = *roi_order < 0 ? 0 : *roi_order * 2;

          T roi_y0 = static_cast<T>(roi_to_pool[(order++) % 8] * map_height);
          T roi_x0 = static_cast<T>(roi_to_pool[(order++) % 8] * map_width);
          T roi_y1 = static_cast<T>(roi_to_pool[(order++) % 8] * map_height);
          T roi_x1 = static_cast<T>(roi_to_pool[(order++) % 8] * map_width);
          T roi_y2 = static_cast<T>(roi_to_pool[(order++) % 8] * map_height);
          T roi_x2 = static_cast<T>(roi_to_pool[(order++) % 8] * map_width);
          T roi_y3 = static_cast<T>(roi_to_pool[(order++) % 8] * map_height);
          T roi_x3 = static_cast<T>(roi_to_pool[(order++) % 8] * map_width);

          double len0 = static_cast<double>((roi_y1 - roi_y0) * (roi_y1 - roi_y0) + (roi_x1 - roi_x0) * (roi_x1 - roi_x0));
          double len1 = static_cast<double>((roi_y2 - roi_y1) * (roi_y2 - roi_y1) + (roi_x2 - roi_x1) * (roi_x2 - roi_x1));
          double len2 = static_cast<double>((roi_y3 - roi_y2) * (roi_y3 - roi_y2) + (roi_x3 - roi_x2) * (roi_x3 - roi_x2));
          double len3 = static_cast<double>((roi_y0 - roi_y3) * (roi_y0 - roi_y3) + (roi_x0 - roi_x3) * (roi_x0 - roi_x3));
          double cross_len0 = static_cast<double>((roi_y0 - roi_y2) * (roi_y0 - roi_y2) + (roi_x0 - roi_x2) * (roi_x0 - roi_x2));
          double cross_len1 = static_cast<double>((roi_y3 - roi_y1) * (roi_y3 - roi_y1) + (roi_x3 - roi_x1) * (roi_x3 - roi_x1));

          order = *roi_order < 0 ? (len0 + len2 > len1 + len3 ? 1 : 0) : 0;
          // fix ROI
          if(len0 < std::numeric_limits<T>::min() || len1 < std::numeric_limits<T>::min() || len2 < std::numeric_limits<T>::min() || len3 < std::numeric_limits<T>::min()){
          // not check convex for faster speed
          //if(is_convex(roi_to_pool)){
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

          float pool_bin_width = static_cast<float>(std::max(std::min(std::abs(right_top_x - left_top_x), std::abs(right_top_y - left_top_y)), std::min(std::abs(right_bottom_x - left_bottom_x), std::abs(right_bottom_y - left_bottom_y))));
          float pool_bin_height = static_cast<float>(std::max(std::min(std::abs(left_bottom_x - left_top_x), std::abs(left_bottom_y - left_top_y)), std::min(std::abs(right_bottom_x - right_top_x), std::abs(right_bottom_y - right_top_y))));
          int32_t num_elem_width = static_cast<int32_t>(pool_bin_width) + 1;
          int32_t num_elem_height = static_cast<int32_t>(pool_bin_height) + 1;

          T grid_y_step_left = (left_bottom_y - left_top_y)/(num_elem_height + 1.);
          T grid_y_step_right = (right_bottom_y - right_top_y)/(num_elem_height + 1.);
          T grid_x_step_top = (right_top_x - left_top_x)/(num_elem_width + 1.);
          T grid_x_step_bottom = (right_bottom_x - left_bottom_x)/(num_elem_width + 1.);

          if(using_max_pool){
            const int32_t pool_h = pooled_max_index / num_elem_width;
            const int32_t pool_w = pooled_max_index % num_elem_width;

            T col_to_pool = (left_top_x + (pool_w + 1.) * grid_x_step_top + left_bottom_x + (pool_w + 1.) * grid_x_step_bottom) / 2.;
            T row_to_pool = (left_top_y + (pool_h + 1.) * grid_y_step_left + right_top_y + (pool_h + 1.) * grid_y_step_right) / 2.;

            int32_t int_col_to_pool = static_cast<int32_t>(col_to_pool);
            int32_t int_row_to_pool = static_cast<int32_t>(row_to_pool);
            float float_col_to_pool = col_to_pool - int_col_to_pool;
            float float_row_to_pool = row_to_pool - int_row_to_pool;

            // not 'if else' here for there may be collapsing in pooling operation when the ROI is small enough
            if(col_on_map == int_col_to_pool && row_on_map == int_row_to_pool){
              *grad_to_fill += static_cast<T>((1. - float_col_to_pool) * (1. - float_row_to_pool) * pooled_features_grad_in);
            }
            if(col_on_map == int_col_to_pool && row_on_map == std::min(int_row_to_pool + 1, map_height - 1)){
              *grad_to_fill += static_cast<T>((1. - float_col_to_pool) * float_row_to_pool * pooled_features_grad_in);
            }
            if(col_on_map == std::min(int_col_to_pool + 1, map_width - 1) && row_on_map == int_row_to_pool){
              *grad_to_fill += static_cast<T>(float_col_to_pool * (1. - float_row_to_pool) * pooled_features_grad_in);
            }
            if(col_on_map == std::min(int_col_to_pool + 1, map_width - 1) && row_on_map == std::min(int_row_to_pool + 1, map_height - 1)){
              *grad_to_fill += static_cast<T>(float_col_to_pool * float_row_to_pool * pooled_features_grad_in);
            }
          }else{
            T acc_back_grad = static_cast<T>(0);
            for(int32_t pool_h = 0; pool_h < num_elem_height; ++pool_h){
              for(int32_t pool_w = 0; pool_w < num_elem_width; ++pool_w){
                T col_to_pool = (left_top_x + (pool_w + 1.) * grid_x_step_top + left_bottom_x + (pool_w + 1.) * grid_x_step_bottom) / 2.;
                T row_to_pool = (left_top_y + (pool_h + 1.) * grid_y_step_left + right_top_y + (pool_h + 1.) * grid_y_step_right) / 2.;

                int32_t int_col_to_pool = static_cast<int32_t>(col_to_pool);
                int32_t int_row_to_pool = static_cast<int32_t>(row_to_pool);
                float float_col_to_pool = col_to_pool - int_col_to_pool;
                float float_row_to_pool = row_to_pool - int_row_to_pool;

                if(col_on_map == int_col_to_pool && row_on_map == int_row_to_pool){
                  acc_back_grad += static_cast<T>((1. - float_col_to_pool) * (1. - float_row_to_pool) * pooled_features_grad_in);
                }
                if(col_on_map == int_col_to_pool && row_on_map == std::min(int_row_to_pool + 1, map_height - 1)){
                  acc_back_grad += static_cast<T>((1. - float_col_to_pool) * float_row_to_pool * pooled_features_grad_in);
                }
                if(col_on_map == std::min(int_col_to_pool + 1, map_width - 1) && row_on_map == int_row_to_pool){
                  acc_back_grad += static_cast<T>(float_col_to_pool * (1. - float_row_to_pool) * pooled_features_grad_in);
                }
                if(col_on_map == std::min(int_col_to_pool + 1, map_width - 1) && row_on_map == std::min(int_row_to_pool + 1, map_height - 1)){
                  acc_back_grad += static_cast<T>(float_col_to_pool * float_row_to_pool * pooled_features_grad_in);
                }
              }
            }
            *grad_to_fill += acc_back_grad / static_cast<T>(num_elem_width * num_elem_height);
          }
        }

      }
    };

    const DeviceBase::CpuWorkerThreads& worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    // one worker for one position in each ROI
    const int64_t shard_cost = num_rois * 4;
    Shard(worker_threads.num_threads, worker_threads.workers,
          grad_output.size(), shard_cost, pooling_grad_routine);
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class RotatedPSROIAlignGradOp : public OpKernel {
 public:
  explicit RotatedPSROIAlignGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("grid_dim_width", &grid_dim_width_in));
    OP_REQUIRES(context, grid_dim_width_in >= 0, errors::InvalidArgument("Need Attr grid_dim_width >= 0, got ", grid_dim_width_in));

    OP_REQUIRES_OK(context, context->GetAttr("grid_dim_height", &grid_dim_height_in));
    OP_REQUIRES(context, grid_dim_height_in >= 0, errors::InvalidArgument("Need Attr grid_dim_height >= 0, got ", grid_dim_height_in));

    OP_REQUIRES_OK(context, context->GetAttr("pool_method", &pool_method));
    OP_REQUIRES(context, StringPiece(pool_method).contains(StringPiece("mean")) || StringPiece(pool_method).contains(StringPiece("max")), errors::InvalidArgument("Need Attr pool_method to be either 'mean' or 'max', got ", pool_method));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& inputs_in = context->input(0);
    const Tensor& rois_in = context->input(1);
    const Tensor& orders_in = context->input(2);
    const Tensor& pooled_features_grad = context->input(3);
    const Tensor& pooled_index = context->input(4);

    OP_REQUIRES(context, inputs_in.shape().dims() == 4, errors::InvalidArgument("inputs must be in 'NCHW' format."));
    OP_REQUIRES(context, pooled_features_grad.shape() == pooled_index.shape(), errors::InvalidArgument("pooled_index and pooled_features_grad must have the same shape"));
    OP_REQUIRES(context, rois_in.shape().dims() == 3 && rois_in.shape().dim_size(2) == 8, errors::InvalidArgument("rois must be in 'batch_size x num_rois x 8' format."));
    OP_REQUIRES(context, inputs_in.dim_size(0) == rois_in.dim_size(0), errors::InvalidArgument("'batch_size' in inputs and rois don't match."));
    OP_REQUIRES(context, orders_in.shape().dims() == 2, errors::InvalidArgument("orders must be in 'batch_size x num_rois' format."));
    OP_REQUIRES(context, (orders_in.dim_size(0) == rois_in.dim_size(0)) && (orders_in.dim_size(1) == rois_in.dim_size(1)), errors::InvalidArgument("'batch_size' or 'num_rois' in orders and rois don't match."));

    const int batch_size = inputs_in.dim_size(0);
    const int num_channals = inputs_in.dim_size(1);
    const int map_height = inputs_in.dim_size(2);
    const int map_width = inputs_in.dim_size(3);
    const int num_rois = rois_in.dim_size(1);

    const int32_t grid_size = grid_dim_width_in * grid_dim_height_in;
    auto bank_size = static_cast<int>(num_channals / grid_size);

    OP_REQUIRES(context, pooled_features_grad.shape() == TensorShape({batch_size, num_rois, grid_size, bank_size}), errors::InvalidArgument("both pooled_index and pooled_features_grad must have the shape 'batch_size x num_rois x grid_size x bank_size'"));

    Tensor* grad_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, inputs_in.shape(), &grad_output));

    RotatedPSROIAlignGradFunctor<Device, T>()(context, context->eigen_device<Device>(), inputs_in.template flat<T>(), rois_in.template flat<T>(), orders_in.template flat<int32_t>(), grid_dim_width_in, grid_dim_height_in, pooled_features_grad.template flat<T>(), pooled_index.template flat<int32_t>(), grad_output->template flat<T>(), std::make_tuple(batch_size, num_channals, map_height, map_width, num_rois, StringPiece(pool_method).contains(StringPiece("max"))));
    // PSROIPoolingFunctor<Device, T>()(context, context->eigen_device<Device>(), inputs_in.tensor<T, 4>(), rois_in.tensor<T, 3>(), grid_dim_buffer[0], pooled_features->tensor<T, 4>());
  }

private:
  int32_t grid_dim_width_in{-1};
  int32_t grid_dim_height_in{-1};
  std::string pool_method{"max"};
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("RotatedPsRoiAlignGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RotatedPSROIAlignGradOp<CPUDevice, T>);
REGISTER_CPU(float);

// TF_CALL_NUMBER_TYPES(REGISTER_CPU);
// #undef REGISTER_CPU

// Register the GPU kernels.
#if GOOGLE_CUDA == 1
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("RotatedPsRoiAlignGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      RotatedPSROIAlignGradOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
