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
#include "tvm/driver/driver_api.h"

using namespace tvm;

TEST(TESchedule, split) {
    int m = 1024;
    int n = 1024;
    int c = 10;
    auto A = te::placeholder(Array<PrimExpr>{c, m, n}, DataType::Float(32), "A");
    auto T = topi::sum(A, {1, 2});
    auto s = te::create_schedule(Array<te::Operation>{T->op});
    std::cout << "stage num: " << s->stages.size() << std::endl;
    auto reduce_axes = Downcast<te::ComputeOp>(T->op)->reduce_axis;
    ASSERT_EQ(reduce_axes.size(), 2);

    tir::IterVar outer0;
    tir::IterVar inner0;
    tir::IterVar outer1;
    tir::IterVar inner1;
    auto stage = s[T].split(reduce_axes[0], 16, &outer0, &inner0)
            .split(reduce_axes[1], 16, &outer1, &inner1);
    auto mod = LowerSchedule(s, Array<te::Tensor>{A, T}, "main", {}, GlobalVarSupply(NameSupply("")), true);
    std::cout << mod << std::endl;
}

TEST(TESchedule, fuse) {
    //
}