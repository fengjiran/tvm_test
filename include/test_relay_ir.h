//
// Created by 赵丹 on 2023/7/23.
//

#ifndef TVM_TEST_TEST_RELAY_IR_H
#define TVM_TEST_TEST_RELAY_IR_H

#include "tvm/relay/expr.h"

tvm::relay::Constant generate_constant_node(int rows, int cols, tvm::DataType dtype);

void test_constant_expr();

void ListAllOpNames();

#endif //TVM_TEST_TEST_RELAY_IR_H
