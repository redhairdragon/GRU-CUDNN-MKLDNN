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

// Read from handle, write to memory
inline void write_to_dnnl_memory(void* handle, dnnl::memory& mem) {
    dnnl::engine eng = mem.get_engine();
    size_t bytes = mem.get_desc().get_size();

    if (eng.get_kind() == dnnl::engine::kind::cpu) {
        uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
        for (size_t i = 0; i < bytes; ++i) dst[i] = ((uint8_t*)handle)[i];
    }
}

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
    auto src_layer_md = memory::desc(src_dims, dt::f32, tag::tnc);
    auto src_layer_mem = memory(src_layer_md, engine);
    write_to_dnnl_memory(a_v.getData(), src_layer_mem);

    Matrix dx(2 * FEAT_DIM, CHUNK_SIZE, new float[2 * FEAT_DIM * CHUNK_SIZE]);

    Matrix hx(FEAT_DIM, CHUNK_SIZE, new float[FEAT_DIM * CHUNK_SIZE]);
    auto src_iter_md = memory::desc(src_dims, dt::f32, tag::tnc);
    auto src_iter_mem = memory(src_iter_md, engine);
    write_to_dnnl_memory(a_v.getData(), src_layer_mem);

    // Matrix dhx(FEAT_DIM, CHUNK_SIZE, new float[FEAT_DIM * CHUNK_SIZE]);

    // weights: according to MKLDNN, W and U allocated separetedly
    //weights layer contains W, weight_iter contains U
    // 5D RNN weights tensor in the format  --careful when init
    // (num_layers, num_directions, input_channels, num_gates, output_channels).
    //FYI: For GRU cells, the gates order is update, reset and output gate.
    Matrix weights_W(G, FEAT_DIM * FEAT_DIM,
                     new float[G * FEAT_DIM * FEAT_DIM]);
    Matrix weights_U(G, FEAT_DIM * FEAT_DIM,
                     new float[G * FEAT_DIM * FEAT_DIM]);
    auto user_weights_layer_mem
            = memory({weights_dims, dt::f32, tag::ldigo}, engine);
    auto user_weights_iter_mem
            = memory({weights_dims, dt::f32, tag::ldigo}, engine);
    write_to_dnnl_memory(weights_W.getData(), user_weights_layer_mem);
    write_to_dnnl_memory(weights_U.getData(), user_weights_iter_mem);
    // Create memory descriptors for weights with format_tag::any. This enables
    // the LSTM primitive to choose the optimized memory layout.
    auto gru_weights_layer_md = memory::desc(weights_dims, dt::f32, tag::any);
    auto gru_weights_iter_md = memory::desc(weights_dims, dt::f32, tag::any);
    auto bias_md = memory::desc();

    // y,dy: output which is not used for our case. need to be initialized
    // anyway (required by function)
    // Matrix y(CHUNK_SIZE, HIDDEN_SIZE, new float[CHUNK_SIZE * HIDDEN_SIZE]);
    // DNNL will allocate memory for y 
    auto dst_layer_md = memory::desc(dst_dims, dt::f32, tag::tnc);    
    // Matrix dy(CHUNK_SIZE, HIDDEN_SIZE, new float[CHUNK_SIZE * HIDDEN_SIZE]);

    // hy: output of GRU which will be fed into a linear layer
    // dhy: for backprop
    // Matrix hy(CHUNK_SIZE, HIDDEN_SIZE, new float[CHUNK_SIZE * HIDDEN_SIZE]);
    auto dst_iter_md = memory::desc(dst_dims, dt::f32, tag::tnc);

    // Matrix dhy(CHUNK_SIZE, HIDDEN_SIZE, new float[CHUNK_SIZE * HIDDEN_SIZE]);

    auto gru_desc = gru_forward::desc(
        prop_kind::forward_training, rnn_direction::unidirectional,
        src_layer_md,  //
        src_iter_md, gru_weights_layer_md, gru_weights_iter_md, bias_md,
        dst_layer_md, dst_iter_md);

    // Create primitive descriptor.
    auto gru_pd = gru_forward::primitive_desc(gru_desc, engine);
    auto gru_weights_layer_mem = user_weights_layer_mem;
    auto gru_weights_iter_mem = user_weights_iter_mem;

    if (gru_pd.weights_desc() != user_weights_layer_mem.get_desc()) {
        cout<<"Reordering\n";
        gru_weights_layer_mem = memory(gru_pd.weights_desc(), engine);
        reorder(user_weights_layer_mem, gru_weights_layer_mem)
                .execute(engine_stream, user_weights_layer_mem,
                        gru_weights_layer_mem);
    }
}
