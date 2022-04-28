#include "tensortype.hpp"
#include "dnnl/impl.hpp"

namespace tt { namespace dnnl {

#include "dnnl/base.hpp"
#include "dnnl/math.hpp"
#include "dnnl/math_reduce.hpp"
#include "dnnl/math_rand.hpp"
#include "dnnl/math_relu.hpp"
#include "dnnl/math_fc.hpp"
#include "dnnl/math_softmax.hpp"
#include "dnnl/cnn_concat.hpp"
#include "dnnl/cnn_resample.hpp"
#include "dnnl/cnn_conv2d.hpp"
#include "dnnl/cnn_pool2d.hpp"
#include "dnnl/cnn_batchnorm.hpp"
#include "dnnl/cnn_layernorm.hpp"
#include "dnnl/rnn_gru.hpp"

namespace dnnl_help {
    dnnl_engine_t DNNL_ENGINE_DEFAULT = nullptr;
    dnnl_stream_t DNNL_STREAM_DEFAULT = nullptr;
}

cpu_float_t* build_dense_plain_float(const ShapeType& shape) {
    return new CPUTensor<DataType::Float>(shape);
}


}}
