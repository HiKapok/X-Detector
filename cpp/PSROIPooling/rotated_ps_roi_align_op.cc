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
REGISTER_OP("RotatedPsRoiAlign")
    .Attr("T: {float}")
    .Attr("grid_dim_width: int")
    .Attr("grid_dim_height: int")
    .Attr("pool_method: string")
    .Input("inputs: T")
    .Input("rois: T")
    .Input("orders: int32")
    .Output("pooled_features: T")
    .Output("pooled_index: int32")
    .Doc(R"doc(
        RotatedPsRoiAlign is a new PsRoiPooling method without align problems.
        The input rois to be pooled must in format [y0, x0, y1, x1, y2, x2, y3, x3] which is four vertexes defining quadrilateral in clockwise order and each element must be in range [0, 1.].
        The input orders define which point is the first one, each element must be in range [-1, 4). The order will be determined to be the first vertex of the shorter side if given -1.
        The caller must make sure that all rois is valid (has a intersect region (one pixel at least) with the window [0., 0., 0., 1., 1., 1., 1., 0.]).
        )doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle inputs_shape = c->input(0);
      shape_inference::DimensionHandle num_per_batch = c->Dim(inputs_shape, 0);
      shape_inference::DimensionHandle num_channals = c->Dim(inputs_shape, 1);
      shape_inference::DimensionHandle num_rois = c->Dim(c->input(1), 1);
      //TF_RETURN_IF_ERROR(c->MakeDimGetAttrForScalarInput(3, &grid_dim_height));
      int32_t grid_dim_width(0);
      TF_RETURN_IF_ERROR(c->GetAttr("grid_dim_width", &grid_dim_width));
      int32_t grid_dim_height(0);
      TF_RETURN_IF_ERROR(c->GetAttr("grid_dim_height", &grid_dim_height));
      // one can use following function to make more check on input shape
      // use WithValue check DimensionHandle, and use WithRank check ShapeHandle
      // TF_RETURN_IF_ERROR(c->WithRank(logits_shape, 2, &logits_shape));
      // TF_RETURN_IF_ERROR(c->WithValue(num_per_batch, 128, &num_per_batch));
      const int32_t grid_size(grid_dim_width * grid_dim_height);
      shape_inference::DimensionHandle bank_size;
      TF_RETURN_IF_ERROR(c->Divide(num_channals, grid_size, true, &bank_size));
      // use MakeShape to create a ShapeHandle from one DimensionHandle
      c->set_output(0, c->MakeShape({num_per_batch, num_rois, grid_size, bank_size}));
      c->set_output(1, c->MakeShape({num_per_batch, num_rois, grid_size, bank_size}));
      //c->set_output(1, c->MakeShape({num_per_batch, num_classes}));
      return Status::OK();
    });


// CPU specialization of actual computation.
//template <typename T>
template <typename T>
struct RotatedPSROIAlignFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const CPUDevice& d, typename TTypes<T>::ConstFlat inputs, typename TTypes<T>::ConstFlat rois, typename TTypes<int32_t>::ConstFlat orders, const int32_t grid_dim_width, const int32_t grid_dim_height, typename TTypes<T>::Flat pooled_features, typename TTypes<int32_t>::Flat pooled_index, KDimSize dim_info) {

    int batch_size = 0;
    int num_channals = 0;
    int map_height = 0;
    int map_width = 0;
    int num_rois = 0;
    bool using_max_pool = false;

    std::tie(batch_size, num_channals, map_height, map_width, num_rois, using_max_pool) = dim_info;

    auto pooling_routine = [&inputs, &rois, &orders, &pooled_features, &pooled_index, grid_dim_width, grid_dim_height, batch_size, num_channals, map_height, map_width, num_rois, using_max_pool](int64_t start, int64_t limit){
      const int32_t grid_size = grid_dim_width * grid_dim_height;
      const int32_t bank_size = num_channals/grid_size;

      // auto fn_get_nth_dividing_points = [](T y1, T x1, T y2, T x2, int32_t nth /* from 0 to total - 1 */, int32_t total) -> std::make_tuple<T, T> {
      //   T nth_x = (x1 + x2) / 2.;
      //   double x_step = (x1 - x2) / (total + 1.);
      //   double y_step = (y1 - y2) / (total + 1.);
      //   if(std::abs(x1 - x2) > std::numeric_limits<T>::min()){
      //     nth_x = x2 + nth * x_step;
      //   }
      //   T nth_y = y2 + nth * y_step;
      //   return std::make_tuple(nth_y, nth_x);
      // };

      // https://stackoverflow.com/questions/471962/how-do-determine-if-a-polygon-is-complex-convex-nonconvex
      // Return True if the polynomial defined by the sequence of 2D
      // points is 'strictly convex': points are valid, side lengths non-
      // zero, interior angles are strictly between zero and a straight
      // angle, and the polygon does not intersect itself.

      // NOTES:  1.  Algorithm: the signed changes of the direction angles
      //             from one side to the next side must be all positive or
      //             all negative, and their sum must equal plus-or-minus
      //             one full turn (2 pi radians). Also check for too few,
      //             invalid, or repeated points.
      //         2.  No check is explicitly done for zero internal angles
      //             (180 degree direction-change angle) as this is covered
      //             in other ways, including the `n < 3` check.
      auto is_convex = [](const T * points){
        double TWO_PI = 2 * PI;
        // Get starting information
        T old_x = points[5];
        T old_y = points[4];
        T new_x = points[7];
        T new_y = points[6];
        if(std::abs(new_y - old_y) < std::numeric_limits<T>::min() && std::abs(new_x - old_x) < std::numeric_limits<T>::min()) return false;
        T new_direction = std::atan2(new_y - old_y, new_x - old_x);
        T old_direction = 0.;
        double angle_sum = 0.;
        double orientation = 1.;
        // Check each point (the side ending there, its angle) and accum. angles
        for(uint16_t index = 0; index < 4; index++){
            // Update point coordinates and side directions, check side length
            old_x = new_x;
            old_y = new_y;
            old_direction = new_direction;
            new_y = points[2 * index];
            new_x = points[2 * index + 1];
            if(std::abs(old_x - new_x) < std::numeric_limits<T>::min() && std::abs(old_y - new_y) < std::numeric_limits<T>::min()) return false;  // repeated consecutive points
            new_direction = std::atan2(new_y - old_y, new_x - old_x);
            // Calculate & check the normalized direction-change angle
            double angle = new_direction - old_direction;
            if(angle <= -PI) angle += TWO_PI;  // make it in half-open interval (-Pi, Pi]
            else if(angle > PI) angle -= TWO_PI;

            if(index == 0){  // if first time through loop, initialize orientation
                if(angle == 0.0) return false;
                orientation = angle > 0.0 ? 1.0 : -1.0;
            }else{  // if other time through loop, check orientation is stable
                if(orientation * angle <= 0.0) return false;// not both pos. or both neg.
            }
            // Accumulate the direction-change angle
            angle_sum += angle;
        }
        // Check that the total number of full turns is plus-or-minus 1
        return std::abs(std::round(angle_sum / TWO_PI)) == 1;
      };

      for (int64_t worker_index = start; worker_index < limit; ++worker_index){
        // worker_index / bank_size / grid_size * num_channals + worker_index / bank_size / grid_size % num_rois * num_channals + (worker_index % grid_size) + worker_index % bank_size;
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

        const T * roi_to_pool = rois.data() + (image_index * num_rois + roi_index) * 8;
        const int32_t * roi_order = orders.data() + image_index * num_rois + roi_index;

        const T * feature_map_to_pool = inputs.data() + (image_index * num_channals + position_index * bank_size + channal_pos_remainder) * map_height * map_width;
        // T * pooled_features_start = pooled_features.data() + image_index * (num_rois * num_channals) + roi_index * num_channals + (position_index % grid_size) * bank_size + channal_pos_remainder;
        // int32_t * pooled_index_start = pooled_index.data() + image_index * (num_rois * num_channals) + roi_index * num_channals + (position_index % grid_size) * bank_size + channal_pos_remainder;
        T * pooled_features_start = pooled_features.data() + worker_index;
        int32_t * pooled_index_start = pooled_index.data() + worker_index;

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

        float pool_bin_width = static_cast<float>(std::max(std::min(std::abs(right_top_x - left_top_x), std::abs(right_top_y - left_top_y)), std::min(std::abs(right_bottom_x - left_bottom_x), std::abs(right_bottom_y - left_bottom_y))));
        float pool_bin_height = static_cast<float>(std::max(std::min(std::abs(left_bottom_x - left_top_x), std::abs(left_bottom_y - left_top_y)), std::min(std::abs(right_bottom_x - right_top_x), std::abs(right_bottom_y - right_top_y))));
        int32_t num_elem_width = static_cast<int32_t>(pool_bin_width) + 1;
        int32_t num_elem_height = static_cast<int32_t>(pool_bin_height) + 1;

        T grid_y_step_left = (left_bottom_y - left_top_y)/(num_elem_height + 1.);
        T grid_y_step_right = (right_bottom_y - right_top_y)/(num_elem_height + 1.);
        T grid_x_step_top = (right_top_x - left_top_x)/(num_elem_width + 1.);
        T grid_x_step_bottom = (right_bottom_x - left_bottom_x)/(num_elem_width + 1.);

        int32_t max_pool_ind = 0;
        T max_or_acc_elem = using_max_pool ? std::numeric_limits<T>::lowest() : static_cast<T>(0);
        //std::cout << "num_elem_height: " << num_elem_height << " num_elem_width:" << num_elem_width << std::endl;
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
            T temp_value = static_cast<T>((1. - float_col_to_pool) * (1. - float_row_to_pool) * feature_map_to_pool[int_row_to_pool * map_width + int_col_to_pool] +
                                      (1. - float_col_to_pool) * float_row_to_pool * feature_map_to_pool[std::min(int_row_to_pool + 1, map_height - 1) * map_width + int_col_to_pool] +
                                      float_col_to_pool * (1. - float_row_to_pool) * feature_map_to_pool[int_row_to_pool * map_width + std::min(int_col_to_pool + 1, map_width - 1)] +
                                      float_col_to_pool * float_row_to_pool * feature_map_to_pool[std::min(int_row_to_pool + 1, map_height - 1) * map_width + std::min(int_col_to_pool + 1, map_width - 1)]);
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
    };

    const DeviceBase::CpuWorkerThreads& worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    // one worker for one position in each ROI
    const int64_t shard_cost = 4 * map_height * map_width / grid_dim_width / grid_dim_height / 4;
    Shard(worker_threads.num_threads, worker_threads.workers,
          pooled_features.size(), shard_cost, pooling_routine);
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class RotatedPSROIAlignOp : public OpKernel {
 public:
  explicit RotatedPSROIAlignOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("grid_dim_width", &grid_dim_width_in));
    OP_REQUIRES(context, grid_dim_width_in >= 0, errors::InvalidArgument("Need Attr grid_dim_width >= 0, got ", grid_dim_width_in));

    OP_REQUIRES_OK(context, context->GetAttr("grid_dim_height", &grid_dim_height_in));
    OP_REQUIRES(context, grid_dim_height_in >= 0, errors::InvalidArgument("Need Attr grid_dim_height >= 0, got ", grid_dim_height_in));

    OP_REQUIRES_OK(context, context->GetAttr("pool_method", &pool_method));
    OP_REQUIRES(context, StringPiece(pool_method).contains(StringPiece("mean")) || StringPiece(pool_method).contains(StringPiece("max")), errors::InvalidArgument("Need Attr pool_method to be either 'mean' or 'max', got ", pool_method));
    // std::cout << (StringPiece(pool_method).contains(StringPiece("mean")) || StringPiece(pool_method).contains(StringPiece("max"))) << std::endl;
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& inputs_in = context->input(0);
    const Tensor& rois_in = context->input(1);
    const Tensor& orders_in = context->input(2);

    OP_REQUIRES(context, inputs_in.shape().dims() == 4, errors::InvalidArgument("inputs must be in 'NCHW' format."));
    OP_REQUIRES(context, rois_in.shape().dims() == 3 && rois_in.shape().dim_size(2) == 8, errors::InvalidArgument("rois must be in 'batch_size x num_rois x 8' format."));
    OP_REQUIRES(context, orders_in.shape().dims() == 2, errors::InvalidArgument("orders must be in 'batch_size x num_rois' format."));
    OP_REQUIRES(context, inputs_in.dim_size(0) == rois_in.dim_size(0), errors::InvalidArgument("'batch_size' in inputs and rois don't match."));
    OP_REQUIRES(context, (orders_in.dim_size(0) == rois_in.dim_size(0)) && (orders_in.dim_size(1) == rois_in.dim_size(1)), errors::InvalidArgument("'batch_size' or 'num_rois' in orders and rois don't match."));

    const int batch_size = inputs_in.dim_size(0);
    const int num_channals = inputs_in.dim_size(1);
    const int map_height = inputs_in.dim_size(2);
    const int map_width = inputs_in.dim_size(3);
    const int num_rois = rois_in.dim_size(1);

    const int32_t grid_size = grid_dim_width_in * grid_dim_height_in;

    auto bank_size = static_cast<int>(num_channals / grid_size);
    Tensor* pooled_features = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {batch_size, num_rois, grid_size, bank_size}, &pooled_features));
    Tensor* pooled_index = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {batch_size, num_rois, grid_size, bank_size}, &pooled_index));

    RotatedPSROIAlignFunctor<Device, T>()(context, context->eigen_device<Device>(), inputs_in.template flat<T>(), rois_in.template flat<T>(), orders_in.template flat<int32_t>(), grid_dim_width_in, grid_dim_height_in, pooled_features->template flat<T>(), pooled_index->template flat<int32_t>(), std::make_tuple(batch_size, num_channals, map_height, map_width, num_rois, StringPiece(pool_method).contains(StringPiece("max"))));
    // RotatedPSROIPoolingFunctor<Device, T>()(context, context->eigen_device<Device>(), inputs_in.tensor<T, 4>(), rois_in.tensor<T, 3>(), grid_dim_buffer[0], pooled_features->tensor<T, 4>());
  }

private:
  int32_t grid_dim_width_in{-1};
  int32_t grid_dim_height_in{-1};
  std::string pool_method{"max"};
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("RotatedPsRoiAlign").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      RotatedPSROIAlignOp<CPUDevice, T>);
REGISTER_CPU(float);

// TF_CALL_NUMBER_TYPES(REGISTER_CPU);
// #undef REGISTER_CPU

// Register the GPU kernels.
#if GOOGLE_CUDA == 1
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("RotatedPsRoiAlign").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      RotatedPSROIAlignOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
