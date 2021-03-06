#ifndef __COMP_UNIT_HPP__
#define __COMP_UNIT_HPP__

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "cu_matrix.cuh"
#include "cublas_v2.h"
#include "cuda_ops.cuh"

using namespace std;
// AWARE: to free Matrix.data in long run
// It maintains a GPU context
class ComputingUnit {
   public:
    static ComputingUnit &getInstance();

    CuMatrix wrapMatrix(Matrix m);

    // GAT related
    CuMatrix exp(CuMatrix &m);
    CuMatrix leakyRelu(CuMatrix &m, float coef);
    CuMatrix leakyReluPrime(CuMatrix &m, float coef);
    CuMatrix hadamardAdd(CuMatrix &matLeft, CuMatrix &matRight);
    CuMatrix gatherRows(CuMatrix m, vector<int> indices);
    CuMatrix reduceColumns(CuMatrix m);
    // CuMatrix hadamardMulBcast(CuMatrix &matLeft, CuMatrix &vec);

    CuMatrix scaleRowsByVector(Matrix m, Matrix v);
    CuMatrix aggregate(CuMatrix &sparse, CuMatrix &dense);

    CuMatrix dot(Matrix &A, Matrix &B);
    void activate(CuMatrix &A);
    CuMatrix softmaxRows(CuMatrix &mat);
    CuMatrix hadamardSub(CuMatrix &matLeft, CuMatrix &matRight);
    CuMatrix hadamardMul(CuMatrix &matLeft, CuMatrix &matRight);
    CuMatrix activateBackward(CuMatrix &y, CuMatrix &gradient);

    unsigned checkAccuracy(CuMatrix &predictions, CuMatrix &labels);
    float checkLoss(CuMatrix &preds, CuMatrix &labels);

    cudnnHandle_t cudnnHandle;
    cusparseHandle_t spHandle;
    cublasHandle_t handle;
    cublasStatus_t stat;

    cudaStream_t stream;

   private:
    ComputingUnit();
    static ComputingUnit *instance;
};

#endif