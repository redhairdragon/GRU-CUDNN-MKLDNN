#include <cuda_runtime.h>
#include <cudnn.h>

#include <iostream>

#include "GPU-Computation/comp_unit.cuh"
#include "GPU-Computation/cu_matrix.cuh"
//refer to https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
using namespace std;
static auto cu = ComputingUnit::getInstance();
// graph attention network
class GAT {
   public:
    GAT(unsigned in_, unsigned out_, CuMatrix adj_) {
        in = in_;
        out = out_;
        float *w_arr = new float[in_ * out_];
        float *a_arr = new float[2 * out];
        for (int i = 0; i < in_ * out_; ++i) w_arr[i] = 0.01 * i;
        for (int i = 0; i < 2 * out; ++i) a_arr[i] = 0.01 * i;
        weight = cu.wrapMatrix(Matrix(in, out, (FeatType *)w_arr));
        a = cu.wrapMatrix(Matrix(out, 2, (FeatType *)a_arr));
        // a.updateMatrixFromGPU();
        // cout << a.str();

        // adj should already have RowPtr, ColInd, Value
        // get rowidx
        adj = adj_;
        cusparseXcsr2coo(cu.spHandle, adj.csrRowPtr, adj.nnz, adj.getRows(),
                         adj.csrRowInd, CUSPARSE_INDEX_BASE_ZERO);

        // int* rowInd=new int[adj.nnz];
        // int* colInd=new int[adj.nnz];
        // cudaMemcpy(rowInd, adj.csrRowInd, sizeof(unsigned) *
        // adj.nnz,cudaMemcpyDeviceToHost); cudaMemcpy(colInd, adj.csrColInd,
        // sizeof(unsigned) * adj.nnz,cudaMemcpyDeviceToHost); 
        // for(int i = 0;i
        // <adj.nnz;++i)
        //     cout<<rowInd[i]<<" ";
        // cout<<"\n";
        // for(int i = 0;i <adj.nnz;++i)
        //     cout<<colInd[i]<<" ";
        // cout<<"\n";

        e = adj;
        alpha = adj;
    }

    CuMatrix forward(CuMatrix feature) {
        // Linear
        unsigned n = feature.getRows();

        auto z = feature.dot(weight);
        // z.updateMatrixFromGPU();
        // cout<<z.str();
        auto az = z.dot(a);
        // az.updateMatrixFromGPU();
        // cout << az.str();
        az = az.transpose();

        CuMatrix e_row = cu.wrapMatrix(
            Matrix(1, n,
                   (float *)NULL));  // store partial row result for e value
        CuMatrix e_col = cu.wrapMatrix(
            Matrix(1, n,
                   (float *)NULL));  // store partial col result for e value
        auto cusparseStat = cusparseSgthr(
            cu.spHandle, adj.nnz, az.devPtr, e_col.devPtr, adj.csrColInd,
            CUSPARSE_INDEX_BASE_ZERO);  // gather the 1st half of az//
        assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
        // based on colidx
        cusparseStat = cusparseSgthr(
            cu.spHandle, adj.nnz, az.devPtr + n, e_row.devPtr, adj.csrRowInd,
            CUSPARSE_INDEX_BASE_ZERO);  // gather the 2nd half of
        assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

        // int *x=new int[adj.nnz];
        // cudaMemcpy(x,adj.csrRowPtr, sizeof(unsigned) * (adj.nnz),
        //        cudaMemcpyDeviceToHost);

        // az based on rowidx
        // e_row.updateMatrixFromGPU();
        // e_col.updateMatrixFromGPU();
        // cout << e_row.str();
        // cout << e_col.str();

        auto edge_content = cu.hadamardAdd(e_row, e_col);
        edge_content = cu.leakyRelu(edge_content, 0.01);
        auto exp_edge = cu.exp(edge_content);
        // exp_edge.updateMatrixFromGPU();
        // cout<<exp_edge.str();
        // NEED TO SET EDGE MATRIX HERE
        e.csrVal = exp_edge.devPtr;

        CuMatrix ones = cu.wrapMatrix(Matrix(n, 1, (char *)NULL));
        thrust::device_ptr<float> one_dptr(ones.devPtr);
        thrust::fill(one_dptr, one_dptr + n, (float)1.);
        CuMatrix neighor_value = cu.aggregate(e, ones);
        // neighor_value.updateMatrixFromGPU();
        // cout<<neighor_value.str();
        auto nv_dptr = thrust::device_ptr<float>(neighor_value.devPtr);
        thrust::transform(nv_dptr, nv_dptr + n, nv_dptr,
                          inverse_functor());  // 1/neighbor
        CuMatrix alpha_content = cu.hadamardMul(neighor_value, exp_edge);
        alpha.csrVal = alpha_content.devPtr;
        CuMatrix out = cu.aggregate(alpha, z);
        out=out.transpose();
        return out;
    }
    // void backward() {

    // }

    unsigned in;
    unsigned out;
    // param
    CuMatrix a;
    CuMatrix weight;

    // shared graph strucutre not value.
    CuMatrix adj;    // sparse
    CuMatrix alpha;  // sparse
    CuMatrix e;      // sparse
};

int main() {
    float feat_arr[5 * 5];
    for (int i = 0; i < 25; ++i) feat_arr[i] = 0.01 * i;
    CuMatrix features = cu.wrapMatrix(Matrix(5, 5, feat_arr));

    CuMatrix adj;
    adj.setRows(5);
    adj.setCols(5);
    adj.nnz = 5;
    cudaMalloc((void **)&adj.csrVal, adj.nnz * sizeof(float));
    cudaMalloc((void **)&adj.csrColInd, adj.nnz * sizeof(unsigned));
    cudaMalloc((void **)&adj.csrRowInd, adj.nnz * sizeof(unsigned));
    cudaMalloc((void **)&adj.csrRowPtr, (adj.getRows() + 1) * sizeof(unsigned));
    int indices[] = {2, 2, 0, 3, 2};
    int indptr[] = {0, 1, 2, 4, 5};
    int data[] = {1, 1, 1, 1, 1};
    cudaMemcpy(adj.csrVal, data, sizeof(float) * adj.nnz,
               cudaMemcpyHostToDevice);
    cudaMemcpy(adj.csrColInd, indices, sizeof(unsigned) * adj.nnz,
               cudaMemcpyHostToDevice);
    cudaMemcpy(adj.csrRowPtr, indptr, sizeof(unsigned) * (adj.getRows() + 1),
               cudaMemcpyHostToDevice);
    auto gat = GAT(5, 7, adj);
    auto result = gat.forward(features);

    result.updateMatrixFromGPU();
    cout << result.str();
    return 0;
}

// LeakyRelu test
//  Matrix x(5,5,new float[25]);
// for (int i=0;i<25;i++)
//     x.getData()[i]=i%2==0?i:-i;
// cout<<x.str()<<endl;
// auto cu_x=cu.wrapMatrix(x);
// auto cu_y=cu.LeakyRelu(cu_x,0.01);
// cu_y.updateMatrixFromGPU();
// cout<<cu_y.str()<<endl;