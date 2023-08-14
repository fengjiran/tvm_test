//
// Created by 赵丹 on 2023/8/4.
//

#ifndef TVM_TEST_BUILD_RELAY_MODEL_H
#define TVM_TEST_BUILD_RELAY_MODEL_H

#include "tvm/relay/expr.h"
#include "tvm/relay/analysis.h"

tvm::IRModule BuildRelayModel_1();

tvm::relay::Expr BuildRelayModel_2(const tvm::relay::Expr &x);

#endif //TVM_TEST_BUILD_RELAY_MODEL_H
