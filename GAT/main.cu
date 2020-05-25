#include <cuda_runtime.h>
#include <cudnn.h>

#include <iostream>

#include "GPU-Computation/comp_unit.cuh"
#include "GPU-Computation/cu_matrix.cuh"
// Sadly it is a simplified version due to time constrain
// refer to https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
// no concat and no softmax norm
using namespace std;
static auto cu = ComputingUnit::getInstance();
// graph attention network
class GAT {
   public:
    GAT(unsigned in_, unsigned out_, CuMatrix adj_) {
        in = in_;
        out = out_;
        float *w_arr = new float[in_ * out_];
        float *a_arr = new float[out];
        for (int i = 0; i < in_ * out_; ++i) w_arr[i] = 0.01 * i;
        for (int i = 0; i < out; ++i) a_arr[i] = 0.01 * i;
        weight = cu.wrapMatrix(Matrix(in, out, (FeatType *)w_arr));
        a = cu.wrapMatrix(Matrix(out, 1, (FeatType *)a_arr));
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
    }

    CuMatrix forward(CuMatrix feature) {
        // Linear
        unsigned n = feature.getRows();

        z = feature.dot(weight);
        // z.updateMatrixFromGPU();
        // cout<<z.str();
        az = z.dot(a);
        // az.updateMatrixFromGPU();
        // cout << az.str();

        CuMatrix e_row = cu.wrapMatrix(Matrix(
            1, n, (char *)NULL));  // store partial row result for e value

        auto cusparseStat = cusparseSgthr(
            cu.spHandle, adj.nnz, az.devPtr, e_row.devPtr, adj.csrColInd,
            CUSPARSE_INDEX_BASE_ZERO);  // gather the 1st half of az//
        assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);
        auto edge_content = cu.leakyRelu(e_row, 0.01);
        // NEED TO SET EDGE MATRIX HERE
        e.csrVal = edge_content.devPtr;

        CuMatrix h = cu.aggregate(e, z);
        h = h.transpose();
        return h;
    }
    CuMatrix predict() { return cu.softmaxRows(h); }

    void backward(CuMatrix Label, CuMatrix Pred) {
        CuMatrix d_P = cu.hadamardSub(Pred, Label);      // n x out
        CuMatrix d_lrelu = cu.leakyReluPrime(az, 0.01);  // n x 1

        CuMatrix d_lrelu_edge = cu.wrapMatrix(Matrix(adj.nnz, 1, (char *)NULL));
        // BCAST |V| to |E|
        auto cusparseStat = cusparseSgthr(
            cu.spHandle, adj.nnz, d_lrelu.devPtr, d_lrelu_edge.devPtr,
            adj.csrColInd,  // Not sure need to see the actually adjmatrix***
            CUSPARSE_INDEX_BASE_ZERO);  // gather the 1st half of az//
        assert(CUSPARSE_STATUS_SUCCESS == cusparseStat);

        vector<int> src_indices(adj.nnz);
        vector<int> dst_indices(adj.nnz);
        cudaMemcpy(src_indices.data(), adj.csrRowInd, sizeof(int) * adj.nnz,cudaMemcpyDeviceToHost);
        cudaMemcpy(dst_indices.data(), adj.csrColInd, sizeof(int) * adj.nnz,cudaMemcpyDeviceToHost);
        auto d_P_edge = cu.gatherRows(z, src_indices);

        auto d_Act = cu.scaleRowsByVector(d_P_edge, d_lrelu_edge);
        auto d_A = d_Act.dot(a);

        auto z_src = cu.gatherRows(z, src_indices);
        cudaDeviceSynchronize();
        auto z_dst = cu.gatherRows(z, dst_indices);
        auto d_a = d_Act.dot(z_src, false, true);
        cudaDeviceSynchronize();
        d_a = d_a.dot(z_dst);

        // CuMatrix d_act = d_lrelu.dot(d_out,true); // 1 x out
        // CuMatrix d_A=cu.hadamardMulBcast(d_act,a);
        // CuMatrix d_a=d_act.dot(z,false,true);

        // CuMatrix d_z=cu.hadamardAdd(cu.aggregate(e,d_out),d_A.dot());
    }
    // Intermediate
    CuMatrix z;
    CuMatrix h;
    CuMatrix az;

    unsigned in;
    unsigned out;

    // param
    CuMatrix a;
    CuMatrix weight;

    // shared graph strucutre not value.
    CuMatrix adj;  // sparse
    CuMatrix e;    // sparse
};

int main() {
    printf("HI\n");
    float feat_arr[5 * 5];
    for (int i = 0; i < 25; ++i) feat_arr[i] = 0.01 * i;
    CuMatrix features = cu.wrapMatrix(Matrix(5, 5, feat_arr));
    // cout<<features.str();
    // auto out = cu.reduceColumns(features);
    // out.updateMatrixFromGPU();
    // cout<<out.str();

    // float feat_arr[5 * 5];
    // for (int i = 0; i < 25; ++i) feat_arr[i] = 0.01 * i;
    // CuMatrix features = cu.wrapMatrix(Matrix(5, 5, feat_arr));

    // CuMatrix adj;
    // adj.setRows(5);
    // adj.setCols(5);
    // adj.nnz = 5;
    // cudaMalloc((void **)&adj.csrVal, adj.nnz * sizeof(float));
    // cudaMalloc((void **)&adj.csrColInd, adj.nnz * sizeof(unsigned));
    // cudaMalloc((void **)&adj.csrRowInd, adj.nnz * sizeof(unsigned));
    // cudaMalloc((void **)&adj.csrRowPtr, (adj.getRows() + 1) *
    // sizeof(unsigned)); int indices[] = {2, 2, 0, 3, 2}; int indptr[] = {0, 1,
    // 2, 4, 5}; int data[] = {1, 1, 1, 1, 1}; cudaMemcpy(adj.csrVal, data,
    // sizeof(float) * adj.nnz,
    //            cudaMemcpyHostToDevice);
    // cudaMemcpy(adj.csrColInd, indices, sizeof(unsigned) * adj.nnz,
    //            cudaMemcpyHostToDevice);
    // cudaMemcpy(adj.csrRowPtr, indptr, sizeof(unsigned) * (adj.getRows() + 1),
    //            cudaMemcpyHostToDevice);
    // auto gat = GAT(5, 7, adj);
    // auto result = gat.forward(features);

    // result.updateMatrixFromGPU();
    // cout << result.str();
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

// MulBcast test
// float x_arr[]={1,2,3,4,5,6,7,8,9,10,11,12};
// CuMatrix x=cu.wrapMatrix(Matrix(4,3,x_arr));
// float vec_arr[]={.1,.01,.001};
// CuMatrix v=cu.wrapMatrix(Matrix(1,3,vec_arr));
// CuMatrix res= cu.hadamardMulBcast(x,v);
// res.updateMatrixFromGPU();
// cout<<res.str();

// float feat_arr[5 * 6];
// for (int i = 0; i < 30; ++i) feat_arr[i] = 0.01 * i;
// CuMatrix features = cu.wrapMatrix(Matrix(5, 6, feat_arr));
// vector<int> idx({1,2,3,1,1,4,0});
// auto out=cu.gatherRows(features,idx);
// out.updateMatrixFromGPU();
// cout<<features.str();
// cout<<out.str();