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
    std::cout << "stage num: " << s->stages.size() << std::endl;
    tir::IterVar outer;
    tir::IterVar inner;
    auto reduce_axis = Downcast<te::ComputeOp>(T->op)->reduce_axis[0];
    auto x = s[T].split(reduce_axis, 16, &outer, &inner);
}