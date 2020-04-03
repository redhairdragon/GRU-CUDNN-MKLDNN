#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "Matrix/matrix.hpp"
#include "constants.h"
#include "dnnl.hpp"
// LSTM example https://intel.github.io/mkl-dnn/lstm_example_cpp.html
// Simple intro to DNNL,Please refer to 
// https://intel.github.io/mkl-dnn/getting_started_cpp.html#getting_started_cpp_sub3
using namespace dnnl;
using std::cout;
using std::endl;

using tag = memory::format_tag;
using dt = memory::data_type;
using dim_t = dnnl::memory::dim;

const memory::dim N = CHUNK_SIZE,  // batch size
    T = SEQ_LEN,                   // time steps
    C = FEAT_DIM,                  // channels
    G = 3,                         // gates (GRU has 6->3 U, 3 W)
    L = LAYER_NUM,                 // layers
    D = 1;                         // directions

int main(int argc, char** argv) {
    // Create execution dnnl::engine.
    dnnl::engine engine(dnnl::engine::kind::cpu, 0);
    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    memory::dims src_dims = {T, N, C};
    memory::dims weights_dims = {L, D, C, G, C};
    memory::dims bias_dims = {L, D, G, C};
    memory::dims dst_dims = {T, N, C};

    // Allocate Matrices
    // x_t according to the GRU formula
    Matrix a_v(2 * FEAT_DIM, CHUNK_SIZE, new float[2 * FEAT_DIM * CHUNK_SIZE]);
    //weights: according to MKLDNN, W and U allocated separetedly
    Matrix weights_W(G, FEAT_DIM * FEAT_DIM, new G* FEAT_DIM* FEAT_DIM);
    Matrix weights_U(G, FEAT_DIM * FEAT_DIM, new G* FEAT_DIM* FEAT_DIM);

    // std::vector<float> dst_layer_data(product(dst_dims));
    // std::vector<float> bias_data(product(bias_dims));
}
