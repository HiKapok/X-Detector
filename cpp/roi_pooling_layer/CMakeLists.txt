CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(roi_pooling)

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


# approach 2
CUDA_COMPILE(ROI_POOLING_CU_O roi_pooling_op_gpu.cu MODULE OPTIONS -I$TF_INC -I/usr/local)


ADD_LIBRARY(roi_pooling SHARED
  ${ROI_POOLING_CU_O}
  roi_pooling_op.cc
  )

TARGET_LINK_LIBRARIES(roi_pooling tensorflow_framework ${CUDA_LIBRARIES})



