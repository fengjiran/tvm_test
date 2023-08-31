//
// Created by 赵丹 on 2023/8/31.
//

#ifndef TVM_TEST_TEST_TOPI_OP_H
#define TVM_TEST_TEST_TOPI_OP_H

#include "tvm/te/operation.h"
#include "tvm/tir/expr.h"
#include "tvm/tir/op.h"
#include "tvm/topi/tags.h"

template<typename T>
inline tvm::te::Tensor relu(const tvm::te::Tensor &data, T threshold = static_cast<T>(0),
                            const std::string &name = "T_relu",
                            const std::string &tag = tvm::topi::kElementWise) {
    auto fcompute = [&](const tvm::Array<tvm::tir::Var> &indices) {
        auto threshold_const = tvm::tir::make_const(data->dtype, threshold);
        return tvm::max(data(indices), threshold_const);
    };
    return tvm::te::compute(data->shape, fcompute, name, tag);
}

#endif //TVM_TEST_TEST_TOPI_OP_H
