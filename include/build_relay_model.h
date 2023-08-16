//
// Created by 赵丹 on 2023/8/4.
//

#ifndef TVM_TEST_BUILD_RELAY_MODEL_H
#define TVM_TEST_BUILD_RELAY_MODEL_H

#include "tvm/relay/expr.h"
#include "tvm/relay/analysis.h"

tvm::IRModule BuildRelayModel_1();

tvm::relay::Expr BuildConvBNRelu(const tvm::relay::Expr &x, int cin, int cout, int stride,
                                 int padding, int dilation, int groups, int ksize = 3);

tvm::relay::Expr BuildAddExpr(const tvm::relay::Expr& x, const tvm::relay::Expr& y);
#endif //TVM_TEST_BUILD_RELAY_MODEL_H
