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
#include "ps_roi_align_op.h"
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
REGISTER_OP("PsRoiAlign")
    .Attr("T: {float}")
    .Attr("grid_dim_width: int")
    .Attr("grid_dim_height: int")
    .Attr("pool_method: string")
    .Input("inputs: T")
    .Input("rois: T")
    // .Input("grid_dim_width: int32")
    // .Input("grid_dim_height: int32")
    .Output("pooled_features: T")
    .Output("pooled_index: int32")
    .Doc(R"doc(
        PsRoiAlign is a new PsRoiPooling method without align problems.
        The input rois to be pooled must in format [center_y, center_x, h, w] and each element must be in range [0, 1.].
        The caller must make sure that all rois is valid (has a intersect region (one pixel at least) with the window [0.5, 0.5, 1., 1.]).
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
struct PSROIAlignFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const CPUDevice& d, typename TTypes<T>::ConstFlat inputs, typename TTypes<T>::ConstFlat rois, const int32_t grid_dim_width, const int32_t grid_dim_height, typename TTypes<T>::Flat pooled_features, typename TTypes<int32_t>::Flat pooled_index, KDimSize dim_info) {

    int batch_size = 0;
    int num_channals = 0;
    int map_height = 0;
    int map_width = 0;
    int num_rois = 0;
    bool using_max_pool = false;

    std::tie(batch_size, num_channals, map_height, map_width, num_rois, using_max_pool) = dim_info;

    auto pooling_routine = [&inputs, &rois, &pooled_features, &pooled_index, grid_dim_width, grid_dim_height, batch_size, num_channals, map_height, map_width, num_rois, using_max_pool](int64_t start, int64_t limit){
      const int32_t grid_size = grid_dim_width * grid_dim_height;
      const int32_t bank_size = num_channals/grid_size;
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

        const T * roi_to_pool = rois.data() + (image_index * num_rois + roi_index) * 4;

        const T * feature_map_to_pool = inputs.data() + (image_index * num_channals + position_index * bank_size + channal_pos_remainder) * map_height * map_width;
        // T * pooled_features_start = pooled_features.data() + image_index * (num_rois * num_channals) + roi_index * num_channals + (position_index % grid_size) * bank_size + channal_pos_remainder;
        // int32_t * pooled_index_start = pooled_index.data() + image_index * (num_rois * num_channals) + roi_index * num_channals + (position_index % grid_size) * bank_size + channal_pos_remainder;
        T * pooled_features_start = pooled_features.data() + worker_index;
        int32_t * pooled_index_start = pooled_index.data() + worker_index;

        T roi_ymin = static_cast<T>(0);
        T roi_xmin = static_cast<T>(0);
        T roi_ymax = static_cast<T>(0);
        T roi_xmax = static_cast<T>(0);
        // fix ROI
        if(roi_to_pool[2] < std::numeric_limits<T>::min() || roi_to_pool[3] < std::numeric_limits<T>::min()){
          *pooled_features_start = static_cast<T>(0);
          continue;
        }

        std::tie(roi_ymin, roi_xmin, roi_ymax, roi_xmax) = [roi_to_pool, map_height, map_width](){
          T roi_y_center = static_cast<T>(roi_to_pool[0] * map_height);
          T roi_x_center = static_cast<T>(roi_to_pool[1] * map_width);
          T roi_h = std::max(roi_to_pool[2] * map_height, static_cast<T>(1));
          T roi_w = std::max(roi_to_pool[3] * map_width, static_cast<T>(1));

          T roi_ymin = std::max(roi_y_center - static_cast<T>(roi_h / 2.), static_cast<T>(0));
          T roi_xmin = std::max(roi_x_center - static_cast<T>(roi_w / 2.), static_cast<T>(0));
          T roi_ymax = std::min(roi_y_center + static_cast<T>(roi_h / 2.), static_cast<T>(map_height) - std::numeric_limits<T>::min());
          T roi_xmax = std::min(roi_x_center + static_cast<T>(roi_w / 2.), static_cast<T>(map_width) - std::numeric_limits<T>::min());
          return std::make_tuple(roi_ymin, roi_xmin, roi_ymax, roi_xmax);
        }();
        // T roi_center_y = roi_to_pool[0];
        // T roi_center_x = roi_to_pool[1];
        T roi_h = roi_ymax - roi_ymin;
        T roi_w = roi_xmax - roi_xmin;
        float pool_bin_width = static_cast<float>(roi_w) / grid_dim_width;
        float pool_bin_height = static_cast<float>(roi_h) / grid_dim_height;
        int32_t num_elem_width = static_cast<int32_t>(pool_bin_width) + 1;
        int32_t num_elem_height = static_cast<int32_t>(pool_bin_height) + 1;

        // std::cout << "pool_bin_width: " << pool_bin_width << " pool_bin_height: " << pool_bin_height << " num_elem_width: " << num_elem_width << " num_elem_height: " << num_elem_height << std::endl;

        // std::cout << "worker_index: " << worker_index << " roi_index: " << roi_index
        // << " roi_ymin: " << roi_ymin << " roi_xmin: " << roi_xmin << " roi_ymax: " << roi_ymax << " roi_xmax: " << roi_xmax << " image_index: " << image_index << " position_index: " << (position_index % grid_size) << " channal_pos_remainder: " << channal_pos_remainder << std::endl;

        float step_width_each_bin = pool_bin_width / num_elem_width;
        float step_height_each_bin = pool_bin_height / num_elem_height;

        float pool_width_start = roi_xmin + pool_bin_width * col_index;
        float pool_height_start = roi_ymin + pool_bin_height * row_index;
        int32_t max_pool_ind = 0;
        T max_or_acc_elem = using_max_pool ? std::numeric_limits<T>::lowest() : static_cast<T>(0);
        for (int32_t h_ind = 0; h_ind < num_elem_height; ++h_ind) {
          for (int32_t w_ind = 0; w_ind < num_elem_width; ++w_ind) {
            float col_to_pool = pool_width_start + step_width_each_bin * w_ind + step_width_each_bin / 2.;
            float row_to_pool = pool_height_start + step_height_each_bin * h_ind + step_height_each_bin / 2.;
            //std::cout << "col_to_pool: " << col_to_pool << " row_to_pool: " << row_to_pool << std::endl;
            int32_t int_col_to_pool = static_cast<int32_t>(col_to_pool);
            int32_t int_row_to_pool = static_cast<int32_t>(row_to_pool);
            float float_col_to_pool = col_to_pool - int_col_to_pool;
            float float_row_to_pool = row_to_pool - int_row_to_pool;

            int32_t current_switch_ind = num_elem_width * h_ind + w_ind;
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
class PSROIAlignOp : public OpKernel {
 public:
  explicit PSROIAlignOp(OpKernelConstruction* context) : OpKernel(context) {
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

    OP_REQUIRES(context, inputs_in.shape().dims() == 4, errors::InvalidArgument("inputs must be in 'NCHW' format."));
    OP_REQUIRES(context, rois_in.shape().dims() == 3 && rois_in.shape().dim_size(2) == 4, errors::InvalidArgument("rois must be in 'batch_size x num_rois x 4' format."));
    OP_REQUIRES(context, inputs_in.dim_size(0) == rois_in.dim_size(0), errors::InvalidArgument("'batch_size' in inputs and rois don't match."));

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

    PSROIAlignFunctor<Device, T>()(context, context->eigen_device<Device>(), inputs_in.template flat<T>(), rois_in.template flat<T>(), grid_dim_width_in, grid_dim_height_in, pooled_features->template flat<T>(), pooled_index->template flat<int32_t>(), std::make_tuple(batch_size, num_channals, map_height, map_width, num_rois, StringPiece(pool_method).contains(StringPiece("max"))));
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
      Name("PsRoiAlign").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      PSROIAlignOp<CPUDevice, T>);
REGISTER_CPU(float);

// TF_CALL_NUMBER_TYPES(REGISTER_CPU);
// #undef REGISTER_CPU

// Register the GPU kernels.
#if GOOGLE_CUDA == 1
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("PsRoiAlign").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      PSROIAlignOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
