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
    //
}

#endif //TVM_TEST_TEST_TOPI_OP_H
