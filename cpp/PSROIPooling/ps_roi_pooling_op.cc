// kernel_example.cc
#include "ps_roi_pooling_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>

using namespace tensorflow;

// no need for int32, just for test
REGISTER_OP("PSROIPooling")
    .Attr("T: {float, int32}")
    .Input("logits: T")
    .Input("labels: T")
    .Input("alphas: T")
    .Input("gamma: T")
    .Output("loss_sum: T")
    //.Output("loss: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle logits_shape = c->input(0);
      shape_inference::DimensionHandle num_per_batch = c->Dim(logits_shape, 0);
      //shape_inference::DimensionHandle num_classes = c->Dim(logits_shape, 1);
      // use MakeShape to create a ShapeHandle from one DimensionHandle
      c->set_output(0, c->MakeShape({num_per_batch}));
      //c->set_output(1, c->MakeShape({num_per_batch, num_classes}));

      // one can use following function to make more check on input shape
      // use WithValue check DimensionHandle, and use WithRank check ShapeHandle
      // TF_RETURN_IF_ERROR(c->WithRank(logits_shape, 2, &logits_shape));
      // TF_RETURN_IF_ERROR(c->WithValue(num_per_batch, 128, &num_per_batch));
      return Status::OK();
    });


// CPU specialization of actual computation.
//template <typename T>
template <typename T>
struct PSROIPoolingFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const CPUDevice& d, typename TTypes<T>::ConstMatrix logits, typename TTypes<T>::ConstMatrix labels, typename TTypes<T>::Matrix softmax,
    typename TTypes<T>::Vec alpha, typename TTypes<T>::ConstScalar gamma, typename TTypes<T>::Vec focal_loss) {
    const int batch_size = logits.dimension(0);
    const int num_classes = logits.dimension(1);

    Eigen::DSizes<int, 1> along_class(1);
    Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
    Eigen::DSizes<int, 2> one_by_class(1, num_classes);

    // shifted_logits = logits - max(logits along classes);
    auto shifted_logits = (logits -
                           logits.maximum(along_class)
                               .eval()
                               .reshape(batch_by_one)
                               .broadcast(one_by_class));

    // Calculate the log of the softmax
    // softmax = logits - max(logits along classes);
    softmax = shifted_logits;
    // softmax = softmax - log(sum(exp(softmax along classes)));
    softmax = (softmax - softmax.exp()
                           .sum(along_class)
                           .eval()
                           .reshape(batch_by_one)
                           .log()
                           .broadcast(one_by_class));

    auto sub_by_one_prob = static_cast<T>(0) - softmax.unaryExpr([](float x){ return 1.-std::exp(x); }).pow(gamma(0));

    focal_loss = (alpha.reshape(Eigen::DSizes<int, 2>(batch_size, 1)).broadcast(one_by_class) * labels * sub_by_one_prob * softmax).sum(along_class).reshape(Eigen::DSizes<int, 1>(batch_size));
    // focal_loss = (alpha.reshape(Eigen::DSizes<int, 2>(1, num_classes)).broadcast(Eigen::DSizes<int, 2>(batch_size, 1)) * labels * sub_by_one_prob * softmax.log()).sum(along_class).reshape(Eigen::DSizes<int, 1>(batch_size));
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class PSROIPoolingOp : public OpKernel {
 public:
  explicit PSROIPoolingOp(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    const Tensor& labels_in = context->input(1);
    const Tensor& alphas_in = context->input(2);
    const Tensor& gamma_in = context->input(3);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(labels_in.shape()),
                errors::InvalidArgument("labels must be 2-dimensional"));

    const int batch_size = logits_in.dim_size(0);
    const int num_classes = logits_in.dim_size(1);

    OP_REQUIRES(context, (labels_in.dim_size(0) == logits_in.dim_size(0)) && (labels_in.dim_size(1) == logits_in.dim_size(1)), errors::InvalidArgument("labels must be matrix of size batch * classes"));

    OP_REQUIRES(context, alphas_in.dims() < 2, errors::InvalidArgument("expected alphas be 1-dimensional vector or scalar, got dimension ", alphas_in.dims()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(gamma_in.shape()), errors::InvalidArgument("gamma must be scalar"));


    Tensor alpha_per_class;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({batch_size}), &alpha_per_class));

    if(TensorShapeUtils::IsScalar(alphas_in.shape())){
      alpha_per_class.vec<T>().setConstant(alphas_in.scalar<T>()());
    }else{
      OP_REQUIRES(context, (alphas_in.dim_size(0) == batch_size), errors::InvalidArgument("alphas must be scalar or is specified for each example, got size ",  alphas_in.dim_size(0)));
      CHECK(alpha_per_class.CopyFrom(alphas_in, TensorShape({batch_size})));
    }

    typename TTypes<T>::Vec alpha_vec = alpha_per_class.vec<T>();

    Tensor softmax_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({batch_size, num_classes}), &softmax_tensor));

    Tensor* loss_sum_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {batch_size}, &loss_sum_out));

    //loss_out->CopyFrom(alpha_per_class, TensorShape({num_classes}));
    //loss_out->vec<T>().setConstant(alphas_in.scalar<T>()());

    PSROIPoolingFunctor<Device, T>()(context, context->eigen_device<Device>(), logits_in.matrix<T>(), labels_in.matrix<T>(), softmax_tensor.matrix<T>(), alpha_vec, gamma_in.scalar<T>(), loss_sum_out->vec<T>());

  }

private:

};

#if GOOGLE_CUDA
template <typename T>
class PSROIPoolingOp<GPUDevice, T> : public OpKernel {
 public:
  explicit PSROIPoolingOp(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    const Tensor& labels_in = context->input(1);
    const Tensor& alphas_in = context->input(2);
    const Tensor& gamma_in = context->input(3);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(labels_in.shape()),
                errors::InvalidArgument("labels must be 2-dimensional"));

    const int batch_size = logits_in.dim_size(0);
    const int num_classes = logits_in.dim_size(1);

    OP_REQUIRES(context, (labels_in.dim_size(0) == logits_in.dim_size(0)) && (labels_in.dim_size(1) == logits_in.dim_size(1)), errors::InvalidArgument("labels must be matrix of size batch * classes"));

    OP_REQUIRES(context, alphas_in.dims() < 2, errors::InvalidArgument("expected alphas be 1-dimensional vector or scalar, got dimension ", alphas_in.dims()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(gamma_in.shape()), errors::InvalidArgument("gamma must be scalar"));


    Tensor alpha_per_class;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({batch_size}), &alpha_per_class));

    if(TensorShapeUtils::IsScalar(alphas_in.shape())){
      alpha_per_class.vec<T>().setConstant(alphas_in.scalar<T>()());
    }else{
      OP_REQUIRES(context, (alphas_in.dim_size(0) == batch_size), errors::InvalidArgument("alphas must be scalar or is specified for each example, got size ",  alphas_in.dim_size(0)));
      CHECK(alpha_per_class.CopyFrom(alphas_in, TensorShape({batch_size})));
    }

    typename TTypes<T>::Vec alpha_vec = alpha_per_class.vec<T>();

    Tensor softmax_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({batch_size, num_classes}), &softmax_tensor));

    Tensor* loss_sum_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {batch_size}, &loss_sum_out));

    //loss_out->CopyFrom(alpha_per_class, TensorShape({num_classes}));
    //loss_out->vec<T>().setConstant(alphas_in.scalar<T>()());

    PSROIPoolingFunctorGPU<T>()(context, context->eigen_device<GPUDevice>(), logits_in.matrix<T>(), labels_in.matrix<T>(), softmax_tensor.matrix<T>(), alpha_vec, gamma_in.scalar<T>(), loss_sum_out->vec<T>());

  }

private:

};
#endif


// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("PSROIPooling").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      PSROIPoolingOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// TF_CALL_NUMBER_TYPES(REGISTER_CPU);
// #undef REGISTER_CPU

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("PSROIPooling").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      PSROIPoolingOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
