#include "comp_unit.cuh"

const float alpha = 1.0f, beta = 0.0f;

ComputingUnit *ComputingUnit::instance = nullptr;
ComputingUnit &ComputingUnit::getInstance() {
    if (instance == nullptr) instance = new ComputingUnit();
    return *instance;
}

ComputingUnit::ComputingUnit() {
    // cudaStreamCreate(&stream);
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("CUBLAS initialization failed\n");
        printf("CUBLAS stat %u\n", stat);
        exit(EXIT_FAILURE);
    }
    cudnnStatus_t err = cudnnCreate(&cudnnHandle);
    if (err != CUDNN_STATUS_SUCCESS) {
        std::cout << "Error occurred: " << err << std::endl;
        std::exit(EXIT_FAILURE);
    }
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    auto cusparseStat = cusparseCreate(&spHandle);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
}

CuMatrix ComputingUnit::reduceColumns(CuMatrix m) {
    CuMatrix out(Matrix(1, m.getCols(), (char *)NULL), handle);
    CuMatrix ones(Matrix(m.getRows(), 1, (char *)NULL), handle);
    thrust::device_ptr<float> one_ptr(ones.devPtr);
    thrust::fill(one_ptr, one_ptr + ones.getNumElemts(), 1);
    cublasSgemv(handle, CUBLAS_OP_N, m.getCols(), m.getRows(), &alpha, m.devPtr,
                m.getCols(), ones.devPtr, 1, &beta, out.devPtr, 1);
    return out;
}

CuMatrix ComputingUnit::leakyRelu(CuMatrix &m, float coef) {
    Matrix out(m.getRows(), m.getCols(), (FeatType *)NULL);
    CuMatrix cu_out = wrapMatrix(out);
    thrust::device_ptr<FeatType> dptr_m(m.devPtr);
    thrust::device_ptr<FeatType> dptr_out(cu_out.devPtr);
    thrust::transform(dptr_m, dptr_m + m.getNumElemts(), dptr_out,
                      leakyRelu_functor(coef));
    return cu_out;
}

CuMatrix ComputingUnit::leakyReluPrime(CuMatrix &m, float coef) {
    Matrix out(m.getRows(), m.getCols(), (FeatType *)NULL);
    CuMatrix cu_out = wrapMatrix(out);
    thrust::device_ptr<FeatType> dptr_m(m.devPtr);
    thrust::device_ptr<FeatType> dptr_out(cu_out.devPtr);
    thrust::transform(dptr_m, dptr_m + m.getNumElemts(), dptr_out,
                      leakyReluPrime_functor(coef));
    return cu_out;
}

// CuMatrix ComputingUnit::hadamardMulBcast(CuMatrix &matLeft, CuMatrix &vec) {
//     // assert(vec.getRows() == 1);
//     // assert(matLeft.getCols() == vec.getCols());
//     CuMatrix res(Matrix(matLeft.getRows(), vec.getCols(), (FeatType *)NULL),
//                  handle);

//     thrust::device_ptr<float> cuLeft_ptr(matLeft.devPtr);
//     thrust::device_ptr<float> res_ptr(res.devPtr);

//     thrust::counting_iterator<int> cnt(0);
//     thrust::transform(cuLeft_ptr, cuLeft_ptr + matLeft.getNumElemts(), cnt,
//                       res_ptr, mulVec(vec.devPtr, vec.getNumElemts()));

//     return res;
// }

CuMatrix ComputingUnit::gatherRows(CuMatrix m, std::vector<int> indices) {
    CuMatrix out =
        wrapMatrix(Matrix(indices.size(), m.getCols(), (char *)NULL));
    int row_size = m.getCols() * sizeof(float);
    int row_cnt = m.getCols();

    for (size_t i = 0; i < indices.size(); ++i) {
        cudaMemcpyAsync(out.devPtr + i * row_cnt,
                        m.devPtr + indices[i] * row_cnt, row_size,
                        cudaMemcpyDeviceToDevice, stream);
    }
    return out;
}

CuMatrix ComputingUnit::exp(CuMatrix &m) {
    Matrix out(m.getRows(), m.getCols(), (FeatType *)NULL);
    CuMatrix cu_out = wrapMatrix(out);
    thrust::device_ptr<FeatType> dptr_m(m.devPtr);
    thrust::device_ptr<FeatType> dptr_out(cu_out.devPtr);
    thrust::transform(dptr_m, dptr_m + m.getNumElemts(), dptr_out,
                      exp_functor());
    return cu_out;
}

CuMatrix ComputingUnit::wrapMatrix(Matrix m) { return CuMatrix(m, handle); }

CuMatrix ComputingUnit::aggregate(CuMatrix &sparse, CuMatrix &dense) {
    CuMatrix C(Matrix(dense.getCols(), sparse.getRows(), (FeatType *)NULL),
               handle);

    cusparseSpMatDescr_t desA;
    cusparseDnMatDescr_t desB;
    cusparseDnMatDescr_t desC;

    auto cusparseStat = cusparseCreateCsr(
        &desA, sparse.getRows(), sparse.getCols(), sparse.nnz, sparse.csrRowPtr,
        sparse.csrColInd, sparse.csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
    cusparseStat = cusparseCreateDnMat(&desB, dense.getCols(), dense.getRows(),
                                       dense.getCols(), dense.devPtr,
                                       CUDA_R_32F, CUSPARSE_ORDER_COL);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
    cusparseStat = cusparseCreateDnMat(&desC, sparse.getRows(), dense.getCols(),
                                       sparse.getRows(), C.devPtr, CUDA_R_32F,
                                       CUSPARSE_ORDER_COL);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

    std::size_t buffer_size;
    cusparseStat = cusparseSpMM_bufferSize(
        spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_TRANSPOSE, &alpha, desA, desB, &beta, desC,
        CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, &buffer_size);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
    float *buffer;
    cudaMalloc((void **)&buffer, buffer_size * sizeof(float));
    cusparseStat =
        cusparseSpMM(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_TRANSPOSE, &alpha, desA, desB, &beta,
                     desC, CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, buffer);
    assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
    cudaFree(buffer);
    return C;
}

// this can be reimplemented with dgmm, but I am too lazy :P
CuMatrix ComputingUnit::scaleRowsByVector(Matrix m, Matrix v) {
    CuMatrix cuM = wrapMatrix(m);
    CuMatrix cuV = wrapMatrix(v);
    thrust::device_ptr<float> m_ptr(cuM.devPtr);
    thrust::device_ptr<float> v_ptr(cuV.devPtr);
    thrust::transform(
        m_ptr, m_ptr + m.getNumElemts(),
        thrust::make_permutation_iterator(
            v_ptr, thrust::make_transform_iterator(
                       thrust::make_counting_iterator(0),
                       linear_index_to_row_index<int>(m.getCols()))),
        m_ptr, thrust::multiplies<float>());
    return cuM;
}

// void ComputingUnit::scaleRowsByVector(CuMatrix &cuM, CuMatrix &cuV) {
//     stat = cublasSdgmm(handle, CUBLAS_SIDE_RIGHT, cuM.getCols(),
//     cuV.getRows(),
//                        cuM.devPtr, cuM.getCols(), cuV.devPtr, 1,
//                        cuM.devPtr, cuM.getCols());
//     assert(stat == CUBLAS_STATUS_SUCCESS);
// }

CuMatrix ComputingUnit::hadamardSub(CuMatrix &matLeft, CuMatrix &matRight) {
    assert(matLeft.getRows() == matRight.getRows());
    assert(matLeft.getCols() == matRight.getCols());
    CuMatrix res(Matrix(matLeft.getRows(), matLeft.getCols(), (FeatType *)NULL),
                 handle);

    thrust::device_ptr<float> cuLeft_ptr(matLeft.devPtr);
    thrust::device_ptr<float> cuRight_ptr(matRight.devPtr);
    thrust::device_ptr<float> res_ptr(res.devPtr);
    thrust::transform(cuLeft_ptr, cuLeft_ptr + matLeft.getNumElemts(),
                      cuRight_ptr, res_ptr, thrust::minus<float>());
    return res;
}

CuMatrix ComputingUnit::hadamardMul(CuMatrix &matLeft, CuMatrix &matRight) {
    assert(matLeft.getRows() == matRight.getRows());
    assert(matLeft.getCols() == matRight.getCols());
    CuMatrix res(Matrix(matLeft.getRows(), matLeft.getCols(), (FeatType *)NULL),
                 handle);

    thrust::device_ptr<float> cuLeft_ptr(matLeft.devPtr);
    thrust::device_ptr<float> cuRight_ptr(matRight.devPtr);
    thrust::device_ptr<float> res_ptr(res.devPtr);

    thrust::transform(cuLeft_ptr, cuLeft_ptr + matLeft.getNumElemts(),
                      cuRight_ptr, res_ptr, thrust::multiplies<float>());

    return res;
}

CuMatrix ComputingUnit::hadamardAdd(CuMatrix &matLeft, CuMatrix &matRight) {
    assert(matLeft.getRows() == matRight.getRows());
    assert(matLeft.getCols() == matRight.getCols());
    CuMatrix res(Matrix(matLeft.getRows(), matLeft.getCols(), (FeatType *)NULL),
                 handle);

    thrust::device_ptr<float> cuLeft_ptr(matLeft.devPtr);
    thrust::device_ptr<float> cuRight_ptr(matRight.devPtr);
    thrust::device_ptr<float> res_ptr(res.devPtr);

    thrust::transform(cuLeft_ptr, cuLeft_ptr + matLeft.getNumElemts(),
                      cuRight_ptr, res_ptr, thrust::plus<float>());

    return res;
}

CuMatrix ComputingUnit::softmaxRows(CuMatrix &mat) {
    CuMatrix res(Matrix(mat.getRows(), mat.getCols(), (FeatType *)NULL),
                 handle);
    cudnnTensorDescriptor_t srcTensorDesc, sftTensorDesc;
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnCreateTensorDescriptor(&sftTensorDesc);
    cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, mat.getRows(), 1, 1,
                               mat.getCols());
    cudnnSetTensor4dDescriptor(sftTensorDesc, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, mat.getRows(), 1, 1,
                               mat.getCols());
    cudnnSoftmaxForward(cudnnHandle, CUDNN_SOFTMAX_ACCURATE,
                        CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, srcTensorDesc,
                        mat.devPtr, &beta, sftTensorDesc, res.devPtr);
    return res;
}

CuMatrix ComputingUnit::activateBackward(CuMatrix &y, CuMatrix &gradient) {
    cudnnActivationDescriptor_t actDesc;
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_TANH,
                                 CUDNN_NOT_PROPAGATE_NAN, 1.0);

    cudnnTensorDescriptor_t yDesc, dyDesc;
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               y.getRows(), 1, 1, y.getCols());
    cudnnCreateTensorDescriptor(&dyDesc);
    cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               gradient.getRows(), 1, 1, gradient.getCols());

    cudnnActivationBackward(cudnnHandle, actDesc, &alpha, yDesc, y.devPtr,
                            dyDesc, gradient.devPtr, yDesc, y.devPtr, &beta,
                            dyDesc, gradient.devPtr);
    return gradient;
}

CuMatrix ComputingUnit::dot(Matrix &A, Matrix &B) {
    CuMatrix devA(A, handle);
    CuMatrix devB(B, handle);
    CuMatrix devC = devA.dot(devB);
    devC.updateMatrixFromGPU();
    return devC;
}

void ComputingUnit::activate(CuMatrix &A) {
    cudnnTensorDescriptor_t srcTensorDesc;
    cudnnCreateTensorDescriptor(&srcTensorDesc);
    cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, A.getRows(), 1, 1,
                               A.getCols());

    cudnnActivationDescriptor_t actDesc;
    cudnnCreateActivationDescriptor(&actDesc);
    cudnnSetActivationDescriptor(actDesc, CUDNN_ACTIVATION_TANH,
                                 CUDNN_NOT_PROPAGATE_NAN, 1.0);
    cudnnActivationForward(cudnnHandle, actDesc, &alpha, srcTensorDesc,
                           A.devPtr, &beta, srcTensorDesc, A.devPtr);
}

//** much slower than CPU only if Input Matrices are not loaded in GPU
// beforehand
unsigned ComputingUnit::checkAccuracy(CuMatrix &predictions, CuMatrix &labels) {
    unsigned rowSize = predictions.getCols();

    thrust::device_vector<FeatType *> row_starts(predictions.getRows());
    thrust::counting_iterator<int> idxfirst(0);

    thrust::transform(idxfirst, idxfirst + predictions.getRows(),
                      row_starts.begin(),
                      setRowStarts(predictions.devPtr, rowSize));
    thrust::device_vector<unsigned> pred_results(predictions.getRows());
    thrust::transform(row_starts.begin(), row_starts.end(),
                      pred_results.begin(), findRowMaximum(rowSize));

    thrust::transform(idxfirst, idxfirst + predictions.getRows(),
                      row_starts.begin(), setRowStarts(labels.devPtr, rowSize));
    thrust::device_vector<unsigned> true_results(predictions.getRows());
    thrust::transform(pred_results.begin(), pred_results.end(),
                      row_starts.begin(), true_results.begin(),
                      isPredictCorrect(rowSize));

    unsigned totalCorrect =
        thrust::reduce(true_results.begin(), true_results.end(), (unsigned)0,
                       thrust::plus<unsigned>());
    return totalCorrect;
}

//** much slower than CPU only if Input Matrices are not loaded in GPU
// beforehand
float ComputingUnit::checkLoss(CuMatrix &preds, CuMatrix &labels) {
    unsigned rowSize = preds.getCols();

    thrust::counting_iterator<int> idxfirst(0);
    thrust::device_vector<FeatType *> row_starts(preds.getRows());
    thrust::transform(idxfirst, idxfirst + preds.getRows(), row_starts.begin(),
                      setRowStarts(labels.devPtr, rowSize));
    thrust::device_vector<unsigned> true_labels(preds.getRows());
    thrust::transform(row_starts.begin(), row_starts.end(), true_labels.begin(),
                      findTrueLabel(rowSize));
    thrust::transform(idxfirst, idxfirst + preds.getRows(), row_starts.begin(),
                      setRowStarts(preds.devPtr, rowSize));
    thrust::device_vector<FeatType> losses(preds.getRows());
    thrust::transform(true_labels.begin(), true_labels.end(),
                      row_starts.begin(), losses.begin(), getLoss(rowSize));
    float totalLoss = thrust::reduce(losses.begin(), losses.end(), (float)0,
                                     thrust::plus<float>());
    return totalLoss;
}
