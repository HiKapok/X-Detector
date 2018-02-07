CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(ps_roi_pooling)

find_package(CUDA REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr")

# compiler flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 ${OpenMP_CXX_FLAGS} -Wall -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -DGOOGLE_CUDA=1")

# TensorFlow dependencies
EXECUTE_PROCESS(COMMAND python3.5 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_include(), end='', flush=True)"  OUTPUT_VARIABLE TF_INC)

EXECUTE_PROCESS(COMMAND python3.5 -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(tf.sysconfig.get_lib(), end='', flush=True)"  OUTPUT_VARIABLE TF_LIB)


MESSAGE(STATUS "Found TF_INC: " ${TF_INC})
MESSAGE(STATUS "Found TF_INC_EXTERNAL: " ${TF_INC}/external/nsync/public)
MESSAGE(STATUS "Found TF_LIB: " ${TF_LIB})


INCLUDE_DIRECTORIES(${TF_INC})
INCLUDE_DIRECTORIES(${TF_INC}/external/nsync/public)
LINK_DIRECTORIES(${TF_LIB})

# approach 1
# CUDA_ADD_LIBRARY(ps_roi_pooling_gpu SHARED ps_roi_pooling_op.cu OPTIONS -I$TF_INC/tensorflow/stream_executor/cuda -I/usr/local)

# ADD_LIBRARY(ps_roi_pooling SHARED
#   ps_roi_pooling_op.h
#   ps_roi_pooling_op.cc
#   )

# TARGET_LINK_LIBRARIES(ps_roi_pooling tensorflow_framework ${CUDA_LIBRARIES} ps_roi_pooling_gpu)


# approach 2
CUDA_COMPILE(PSROI_POOLING_CU_O ps_roi_pooling_op.cu MODULE OPTIONS -I$TF_INC -I/usr/local)
CUDA_COMPILE(PSROI_POOLING_GRAD_CU_O ps_roi_pooling_grad_op.cu MODULE OPTIONS -I$TF_INC -I/usr/local)

ADD_LIBRARY(ps_roi_pooling SHARED
  ${PSROI_POOLING_CU_O}
  ${PSROI_POOLING_GRAD_CU_O}
  ps_roi_pooling_op.h
  ps_roi_pooling_op.cc
  ps_roi_pooling_grad_op.cc
  )

TARGET_LINK_LIBRARIES(ps_roi_pooling tensorflow_framework ${CUDA_LIBRARIES})


 #nvcc -std=c++11 -c -o cuda_op_kernel.cu.o ../focal_loss_op.cu -I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I/usr/local/ -I$TF_INC/tensorflow/stream_executor/cuda --expt-relaxed-constexpr -gencode arch=compute_61,code=sm_61


#TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
#TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
#g++ -std=c++11 -shared zero_out.cc -o zero_out.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2
#nvcc -std=c++11 -c -o cuda_op_kernel.cu.o ../focal_loss_op.cu -I $TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I/usr/local/ -I$TF_INC/tensorflow/stream_executor/cuda
#cd `python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())'`
#cd tensorflow/stream_executor/cuda
#curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/master/third_party/toolchains/gpus/cuda/cuda/cuda_config.h


# g++ -std=c++11 -shared -o libfocal_loss.so focal_loss_op.cc focal_loss_grad_op.cc cuda_compile_generated_focal_loss_op.cu.o -I $TF_INC -I$TF_INC/external/nsync/public -fPIC -lcudart -L$TF_LIB -ltensorflow_framework

# nvcc -std=c++11 -c -o focal_loss_op.cu.o focal_loss_op.cu -I$TF_INC -I$TF_INC/external/nsync/public -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -I/usr/local -O2 -gencode arch=compute_61,code=sm_61 --expt-relaxed-constexpr

# g++ -std=c++11 -shared -o libfocal_loss.so focal_loss_op.cc focal_loss_grad_op.cc focal_loss_op.cu.o -I$TF_INC -I$TF_INC/external/nsync/public -fPIC -L/usr/local/cuda/lib64 -lcudart -L$TF_LIB -ltensorflow_framework


