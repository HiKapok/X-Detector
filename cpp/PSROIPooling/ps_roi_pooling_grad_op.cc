// kernel_example.cc
#include "ps_roi_pooling_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

// no need for int32, just for test
REGISTER_OP("PSROIPoolingGrad")
    .Attr("T: {float, int32}")
    .Input("logits: T")
    .Input("labels: T")
    .Input("alphas: T")
    .Input("gamma: T")
    .Input("loss: T")
    .Output("grads: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));

      return Status::OK();
    });


// CPU specialization of actual computation.
template <typename T>
struct PSROIPoolingGradFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstMatrix logits, typename TTypes<T>::ConstVec labels, typename TTypes<T>::Matrix softmax,
    typename TTypes<T>::Vec alpha, typename TTypes<T>::ConstScalar gamma, typename TTypes<T>::ConstVec focal_loss, typename TTypes<T>::Matrix grads) {
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

    softmax = shifted_logits.exp();
    // softmax = softmax * (1 / sum(softmax along classes));
    softmax = (softmax * softmax.sum(along_class)
                         .inverse()
                         .eval()
                         .reshape(batch_by_one)
                         .broadcast(one_by_class));


    for(int batch_index = 0;batch_index < batch_size;++batch_index){
      for(int index = 0;index < num_classes;++index){
        float pt = softmax(batch_index, labels(batch_index));
        if(index == labels(batch_index)){
          grads(batch_index, index) = -pt*gamma(0)*focal_loss(batch_index) - alpha(batch_index)*std::pow(1-pt, gamma(0)+1);
          continue;
        }
        grads(batch_index, index) = softmax(batch_index, index)*(pt*gamma(0)*std::pow(1-pt, -1)*focal_loss(batch_index) + alpha(batch_index)*std::pow(1-pt, gamma(0)));
      }
    }

  }
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class PSROIPoolingGradOp : public OpKernel {
 public:
  explicit PSROIPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    const Tensor& labels_in = context->input(1);
    const Tensor& alphas_in = context->input(2);
    const Tensor& gamma_in = context->input(3);
    const Tensor& loss_in = context->input(4);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));


    const int batch_size = logits_in.dim_size(0);
    const int num_classes = logits_in.dim_size(1);

    OP_REQUIRES(context, (labels_in.dim_size(0) == batch_size), errors::InvalidArgument("labels must be vector of batch_size, not one-hot"));
    OP_REQUIRES(context, (loss_in.dim_size(0) == batch_size), errors::InvalidArgument("losses must be vector of size batch_size"));

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

    Tensor* grad_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {batch_size, num_classes}, &grad_out));

    PSROIPoolingGradFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(), logits_in.matrix<T>(), labels_in.vec<T>(), softmax_tensor.matrix<T>(), alpha_vec, gamma_in.scalar<T>(), loss_in.vec<T>(), grad_out->matrix<T>());

  }

private:

};

#if GOOGLE_CUDA
template <typename T>
class PSROIPoolingGradOp<GPUDevice, T> : public OpKernel {
 public:
  explicit PSROIPoolingGradOp(OpKernelConstruction* context) : OpKernel(context) { }

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    const Tensor& labels_in = context->input(1);
    const Tensor& alphas_in = context->input(2);
    const Tensor& gamma_in = context->input(3);
    const Tensor& loss_in = context->input(4);

    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));


    const int batch_size = logits_in.dim_size(0);
    const int num_classes = logits_in.dim_size(1);

    OP_REQUIRES(context, (labels_in.dim_size(0) == batch_size), errors::InvalidArgument("labels must be vector of batch_size, not one-hot"));
    OP_REQUIRES(context, (loss_in.dim_size(0) == batch_size), errors::InvalidArgument("losses must be vector of size batch_size"));

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

    Tensor* grad_out = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {batch_size, num_classes}, &grad_out));

    PSROIPoolingGradFunctorGPU<T>()(context->eigen_device<GPUDevice>(), logits_in.matrix<T>(), labels_in.vec<T>(), softmax_tensor.matrix<T>(), alpha_vec, gamma_in.scalar<T>(), loss_in.vec<T>(), grad_out->matrix<T>());

  }

private:

};
#endif

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("PSROIPoolingGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      PSROIPoolingGradOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  /* Declare explicit instantiations in kernel_example.cu.cc. */ \
  extern template struct PSROIPoolingGradFunctorGPU<float>;                \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("PSROIPoolingGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      PSROIPoolingGradOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
