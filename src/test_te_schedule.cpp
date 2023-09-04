//
// Created by richard on 9/3/23.
//
#include "gtest/gtest.h"
#include "tvm/te/tensor.h"
#include "tvm/te/operation.h"
#include "tvm/te/schedule.h"
#include "tvm/tir/var.h"
#include "tvm/tir/op.h"
#include "tvm/topi/reduction.h"

using namespace tvm;

TEST(TESchedule, split) {
    int n = 1024;
    auto A = te::placeholder(Array<PrimExpr>{n}, DataType::Float(32), "A");
    auto T = topi::sum(A, {0});
    auto s = te::create_schedule(Array<te::Operation>{T->op});
//    auto x = s[T].split()
}