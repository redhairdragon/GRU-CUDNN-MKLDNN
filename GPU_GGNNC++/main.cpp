#include <cuda_runtime.h>
#include <cudnn.h>

#include <algorithm>
#include <iostream>
#include <vector>

#include "GPU-Computation/cu_matrix.cuh"
#include "constants.h"

// Define some error checking macros.
#define cudaErrCheck(stat) \
    { cudaErrCheck_(stat); }
void cudaErrCheck_(cudaError_t stat) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(stat));
    }
}

#define cudnnErrCheck(stat) \
    { cudnnErrCheck_(stat); }
void cudnnErrCheck_(cudnnStatus_t stat) {
    if (stat != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "cuDNN Error: %s\n", cudnnGetErrorString(stat));
    }
}

int dim[3] = {LAYER_NUM, CHUNK_SIZE, HIDDEN_SIZE};
int stride[3] = {CHUNK_SIZE * HIDDEN_SIZE, HIDDEN_SIZE, 1};
// Here is a RNN implementation using cuDNN
// https://gist.github.com/aonotas/c431f5bc55d1e9c3b9b7201f1ffbb2ab
using namespace std;

int main() {
    // create handle for cuda
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudnnHandle_t dnnHandle;
    cudnnErrCheck(cudnnCreate(&dnnHandle));

    // // create&set Dropout descriptor
    size_t stateSize;
    void *states;
    cudnnErrCheck(cudnnDropoutGetStatesSize(dnnHandle, &stateSize));
    cudaErrCheck(cudaMalloc(&states, stateSize));

    cudnnDropoutDescriptor_t dropoutDesc;
    cudnnErrCheck(cudnnCreateDropoutDescriptor(&dropoutDesc));
    cudnnErrCheck(cudnnSetDropoutDescriptor(dropoutDesc, dnnHandle, DROPOUT,
                                            states, stateSize, SEED));
    // create&set RNN descriptor
    cudnnRNNDescriptor_t rnnDesc;
    cudnnErrCheck(cudnnCreateRNNDescriptor(&rnnDesc));
    cudnnSetRNNBiasMode(rnnDesc, CUDNN_RNN_NO_BIAS);
    // Enable Tensor Core. Sound Cool But (∀ Dim % 8 == 0)
    // cudnnSetRNNMatrixMathType(rnnDesc, CUDNN_TENSOR_OP_MATH);

    cudnnErrCheck(cudnnSetRNNDescriptor(
        dnnHandle, rnnDesc, HIDDEN_SIZE, LAYER_NUM, dropoutDesc,
        CUDNN_LINEAR_INPUT,
        // CUDNN_LINEAR_INPUT, CUDNN_SKIP_INPUT(requires input_dim=hidden_dim)
        CUDNN_UNIDIRECTIONAL,
        // CUDNN_BIDIRECTIONAL(requires to double the sizes of tensors )
        CUDNN_GRU, CUDNN_RNN_ALGO_STANDARD,
        //  CUDNN_RNN_ALGO_PERSIST_STATIC  will be faster however requires small
        //  input dim
        CUDNN_DATA_FLOAT));

    // // a_v: Aggregate Feature Matrix -> dim: 2D*Chunk  *****COLUMN MAJOR*****
    // // A_v^T -> dim: 2DxD|V| Aggregate edge matrix (sparse)
    // // H Feat Matrix [h1  h2... h|V|] -> dim: D|V|* 1
    Matrix a_v(2 * FEAT_DIM, CHUNK_SIZE, new float[2 * FEAT_DIM * CHUNK_SIZE]);
    std::generate(&a_v.getData()[0], &a_v.getData()[2 * FEAT_DIM * CHUNK_SIZE],
                  []() {
                      static int i = 1;
                      return 0.01 * i++;
                  });
    CuMatrix a_v_cuda(a_v, handle);
    cudnnTensorDescriptor_t xDesc[SEQ_LEN];

    // // dx
    Matrix dx(2 * FEAT_DIM, CHUNK_SIZE, new float[2 * FEAT_DIM * CHUNK_SIZE]);
    CuMatrix dx_cuda(dx, handle);
    cudnnTensorDescriptor_t dxDesc[SEQ_LEN];

    for (int i = 0; i < SEQ_LEN; i++) {
        cudnnCreateTensorDescriptor(&xDesc[i]);
        cudnnCreateTensorDescriptor(&dxDesc[i]);
        int dimA[3] = {CHUNK_SIZE, 2 * FEAT_DIM, 1};
        int strideA[3] = {2 * FEAT_DIM, 1, 1};
        cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimA,
                                   strideA);
        cudnnSetTensorNdDescriptor(dxDesc[i], CUDNN_DATA_FLOAT, 3, dimA,
                                   strideA);
    }

    // h_(t-1)Previous GRU state
    // dhx: for bptt
    Matrix hx(FEAT_DIM, CHUNK_SIZE, new float[FEAT_DIM * CHUNK_SIZE]);
    std::generate(&hx.getData()[0], &hx.getData()[FEAT_DIM * CHUNK_SIZE],
                  []() { return 0.1; });
    Matrix dhx(FEAT_DIM, CHUNK_SIZE, new float[FEAT_DIM * CHUNK_SIZE]);
    CuMatrix hx_cuda(hx, handle);
    CuMatrix dhx_cuda(dhx, handle);
    cudnnTensorDescriptor_t hxDesc;
    cudnnTensorDescriptor_t dhxDesc;
    cudnnCreateTensorDescriptor(&hxDesc);
    cudnnCreateTensorDescriptor(&dhxDesc);
    cudnnSetTensorNdDescriptor(hxDesc, CUDNN_DATA_FLOAT, 3, dim, stride);
    cudnnSetTensorNdDescriptor(dhxDesc, CUDNN_DATA_FLOAT, 3, dim, stride);

    // C is used for LSTM (only)
    cudnnTensorDescriptor_t cxDesc, dcxDesc, cyDesc, dcyDesc;
    cudnnCreateTensorDescriptor(&cxDesc);
    cudnnCreateTensorDescriptor(&cyDesc);
    cudnnCreateTensorDescriptor(&dcxDesc);
    cudnnCreateTensorDescriptor(&dcyDesc);
    void *cx = NULL;
    void *dcx = NULL;
    void *cy = NULL;
    void *dcy = NULL;

    // GetWeights
    // this helps to get the size of parameters
    void *w;
    void *dw;
    size_t weightsSize;
    cudnnGetRNNParamsSize(dnnHandle, rnnDesc, xDesc[0], &weightsSize,
                          CUDNN_DATA_FLOAT);
    // std::cout << "weightsSize: " << weightsSize / 1024. << " KB" << endl;
    cudnnFilterDescriptor_t wDesc;
    cudnnFilterDescriptor_t dwDesc;
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateFilterDescriptor(&dwDesc);
    int dimW[3];
    dimW[0] = weightsSize / sizeof(float);
    dimW[1] = 1;
    dimW[2] = 1;
    cudnnSetFilterNdDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3,
                               dimW);
    cudnnSetFilterNdDescriptor(dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3,
                               dimW);
    cudaMalloc((void **)&w, weightsSize);
    cudaMalloc((void **)&dw, weightsSize);

    // y,dy: output which is not used for our case. need to be initialized
    // anyway (required by function)
    Matrix y(CHUNK_SIZE, HIDDEN_SIZE, new float[CHUNK_SIZE * HIDDEN_SIZE]);
    Matrix dy(CHUNK_SIZE, HIDDEN_SIZE, new float[CHUNK_SIZE * HIDDEN_SIZE]);
    std::generate(&dy.getData()[0],
                  &dy.getData()[CHUNK_SIZE * HIDDEN_SIZE],
                  []() { return 0.05; });
    CuMatrix y_cuda(y, handle);
    CuMatrix dy_cuda(dy, handle);
    cudnnTensorDescriptor_t yDesc[SEQ_LEN];
    cudnnTensorDescriptor_t dyDesc[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; i++) {
        cudnnCreateTensorDescriptor(&yDesc[i]);
        cudnnCreateTensorDescriptor(&dyDesc[i]);
        int dimA[3] = {CHUNK_SIZE, HIDDEN_SIZE, 1};
        int strideA[3] = {HIDDEN_SIZE, 1, 1};
        cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimA,
                                   strideA);
        cudnnSetTensorNdDescriptor(dyDesc[i], CUDNN_DATA_FLOAT, 3, dimA,
                                   strideA);
    }

    // hy: output of GRU which will be fed into a linear layer
    // dhy: for backprop
    Matrix hy(CHUNK_SIZE, HIDDEN_SIZE, new float[CHUNK_SIZE * HIDDEN_SIZE]);
    Matrix dhy(CHUNK_SIZE, HIDDEN_SIZE, new float[CHUNK_SIZE * HIDDEN_SIZE]);
    std::generate(&dhy.getData()[0],
                  &dhy.getData()[CHUNK_SIZE * HIDDEN_SIZE],
                  []() { return 0.06; });
    CuMatrix hy_cuda(hy, handle);
    CuMatrix dhy_cuda(dhy, handle);
    cudnnTensorDescriptor_t hyDesc;
    cudnnTensorDescriptor_t dhyDesc;
    cudnnCreateTensorDescriptor(&hyDesc);
    cudnnCreateTensorDescriptor(&dhyDesc);
    cudnnSetTensorNdDescriptor(hyDesc, CUDNN_DATA_FLOAT, 3, dim, stride);
    cudnnSetTensorNdDescriptor(dhyDesc, CUDNN_DATA_FLOAT, 3, dim, stride);

    void *workspace;  // extra VRAM needed for function to run
    size_t workSize;
    cudnnGetRNNWorkspaceSize(dnnHandle, rnnDesc, SEQ_LEN, xDesc, &workSize);
    cudaMalloc((void **)&workspace, workSize);
    // cout << "workSize: "<<workSize/1024. <<" KB"<<endl;

    void *reserveSpace;  // extra VRAM to save tensors for backprop
    size_t reserveSize;
    cudnnGetRNNTrainingReserveSize(dnnHandle, rnnDesc, SEQ_LEN, xDesc,
                                   &reserveSize);
    cudaMalloc((void **)&reserveSpace, reserveSize);
    // cout << "reserveSize: " << reserveSize / 1024. << " KB" << endl;
    // //------------Setup Parameters Done------------
    // //------------------------------------------------------------------------

    // Set weight
    int numLinearLayers = 6;  // 6 weights for GRU
    for (int layer = 0; layer < LAYER_NUM; layer++) {
        for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
            cudnnFilterDescriptor_t linLayerMatDesc;
            cudnnCreateFilterDescriptor(&linLayerMatDesc);
            float *linLayerMat;

            cudnnGetRNNLinLayerMatrixParams(
                dnnHandle, rnnDesc, layer, xDesc[0], wDesc, w, linLayerID,
                linLayerMatDesc, (void **)&linLayerMat);

            cudnnDataType_t dataType;
            cudnnTensorFormat_t format;
            int nbDims;
            int filterDimA[3];
            cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, &dataType, &format,
                                       &nbDims, filterDimA);
            cout << filterDimA[0] << " ";
            cout << filterDimA[1] << " ";
            cout << filterDimA[2] << endl;
            //****Here it should copy weights into GPU memory.
            vector<float> v(filterDimA[0] * filterDimA[1] * filterDimA[2]);
            if (linLayerID < 3)
                std::fill(v.begin(), v.end(), 0.2);
            else
                std::fill(v.begin(), v.end(), 0.1);
            cudaDeviceSynchronize();
            cudaErrCheck(cudaMemcpy(linLayerMat, v.data(),
                                    v.size() * sizeof(float),
                                    cudaMemcpyHostToDevice));
            cudnnDestroyFilterDescriptor(linLayerMatDesc);

            // // set bias
            // cudnnFilterDescriptor_t linLayerBiasDesc;
            // cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
            // float *linLayerBias;
            // cudnnErrCheck(cudnnGetRNNLinLayerBiasParams(
            //     dnnHandle, rnnDesc, layer, xDesc[0], wDesc, w, linLayerID,
            //     linLayerBiasDesc, (void **)&linLayerBias));

            // cudnnErrCheck(cudnnGetFilterNdDescriptor(
            //     linLayerBiasDesc, 3, &dataType, &format, &nbDims, filterDimA));
            // vector<float> b(filterDimA[0] * filterDimA[1] * filterDimA[2]);
            // std::fill(b.begin(), b.end(), 0.000);
            // cudaErrCheck(cudaMemcpy(linLayerBias, b.data(),
            //                         b.size() * sizeof(float),
            //                         cudaMemcpyHostToDevice));
            // // cout <<filterDimA[0] <<" "<< filterDimA[1] <<" "<< filterDimA[2]<<endl;
            // cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
        }
    }

    // //---------------------Forward Setup Done here-----
    // data request: a_v, hx
    // data  update: y, hy, reserved
    cudnnRNNForwardTraining(dnnHandle, rnnDesc, SEQ_LEN, xDesc, a_v_cuda.devPtr,
                            hxDesc, hx_cuda.devPtr, cxDesc, cx, wDesc, w, yDesc,
                            y_cuda.devPtr, hyDesc, hy_cuda.devPtr, cyDesc, cy,
                            workspace, workSize, reserveSpace, reserveSize);

    // Forward done
    cudaDeviceSynchronize();
    y_cuda.updateMatrixFromGPU();
    cout << "y: " << y_cuda.str();
    hy_cuda.updateMatrixFromGPU();
    cout << "hy: " << hy_cuda.str();

    // --------------Allocating Tensors for Backprop--------------------

    cudnnRNNBackwardData(dnnHandle, rnnDesc, SEQ_LEN, yDesc, y_cuda.devPtr,
                         dyDesc, dy_cuda.devPtr, dhyDesc, dhy_cuda.devPtr,
                         dcyDesc, dcy, wDesc, w, hxDesc, hx_cuda.devPtr, cxDesc,
                         cx, dxDesc, dx_cuda.devPtr, dhxDesc, dhx_cuda.devPtr,
                         dcxDesc, dcx, workspace, workSize, reserveSpace,
                         reserveSize);

    cudnnRNNBackwardWeights(dnnHandle, rnnDesc, SEQ_LEN, xDesc, a_v_cuda.devPtr,
                            hxDesc, hx_cuda.devPtr, yDesc, y_cuda.devPtr,
                            workspace, workSize, dwDesc, dw, reserveSpace,
                            reserveSize);

    for (int layer = 0; layer < LAYER_NUM; layer++) {
        for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
            cudnnFilterDescriptor_t linLayerMatDesc;
            cudnnCreateFilterDescriptor(&linLayerMatDesc);
            float *linLayerMat;

            cudnnGetRNNLinLayerMatrixParams(
                dnnHandle, rnnDesc, layer, xDesc[0], dwDesc, dw, linLayerID,
                linLayerMatDesc, (void **)&linLayerMat);

            cudnnDataType_t dataType;
            cudnnTensorFormat_t format;
            int nbDims;
            int filterDimA[3];
            cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, &dataType, &format,
                                       &nbDims, filterDimA);
            //****Here it should copy weights into GPU memory.
            vector<float> v(filterDimA[0] * filterDimA[1] * filterDimA[2]);
            cudaDeviceSynchronize();
            cudaErrCheck(cudaMemcpy( v.data(), linLayerMat,
                                    v.size() * sizeof(float),
                                    cudaMemcpyDeviceToHost));
            if(linLayerID<3) cout<<"dW: \n";
            else cout<<"dU: \n";
            for(unsigned i = 0;i< v.size();++i)
                cout<< v[i]<<" ";
            cout<<endl;
            cudnnDestroyFilterDescriptor(linLayerMatDesc);
        }
    }











    // Destroy Every Cuda-Related
    cudnnDestroyTensorDescriptor(cxDesc);
    cudnnDestroyTensorDescriptor(cyDesc);
    cudnnDestroyTensorDescriptor(dcxDesc);
    cudnnDestroyTensorDescriptor(dcyDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyTensorDescriptor(hxDesc);
    cudnnDestroyTensorDescriptor(dhxDesc);
    cudnnDestroyTensorDescriptor(hyDesc);
    cudnnDestroyTensorDescriptor(dhyDesc);

    cudaFree(reserveSpace);
    cudaFree(states);
    cudaFree(w);
    cudaFree(dw);
    cudaFree(workspace);

    for (int i = 0; i < SEQ_LEN; i++) {
        cudnnDestroyTensorDescriptor(xDesc[i]);
        cudnnDestroyTensorDescriptor(dxDesc[i]);
        cudnnDestroyTensorDescriptor(yDesc[i]);
        cudnnDestroyTensorDescriptor(dyDesc[i]);
    }

    cublasDestroy(handle);
    cudnnDestroy(dnnHandle);
    cudnnDestroyRNNDescriptor(rnnDesc);
    cudnnDestroyDropoutDescriptor(dropoutDesc);
    // cout << "Done\n";
    return 0;
}
