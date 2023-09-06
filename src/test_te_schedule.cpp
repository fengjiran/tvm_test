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
    std::cout << "All iter var num: " << stage->all_iter_vars.size() << std::endl;
    std::cout << "Leaf iter var num: " << stage->leaf_iter_vars.size() << std::endl;
    auto mod = LowerSchedule(s, Array<te::Tensor>{A, T}, "main", {}, GlobalVarSupply(NameSupply("")), true);
    std::cout << mod << std::endl;
}

TEST(TESchedule, reorder) {
    auto m = tir::SizeVar("m");
    auto n = tir::SizeVar("n");
    auto A = te::placeholder(Array<PrimExpr>{m, n}, DataType::Float(32), "A");
    auto B = te::placeholder(Array<PrimExpr>{m, n}, DataType::Float(32), "B");
    auto T = te::compute({m, n}, [&](const tir::Var &i, const tir::Var &j) {
        return A(i, j) + B(i, j);
    });
    auto sch1 = te::create_schedule(Array<te::Operation>{T->op});
    ASSERT_EQ(sch1->stages.size(), 3);
    auto stage = sch1[T];
    LOG_INFO << "Print schedule before reorder:";
    LOG_INFO << "All iter var num before reorder: " << stage->all_iter_vars.size();
    std::cout << "Leaf iter var num before reorder: " << stage->leaf_iter_vars.size() << std::endl;
    auto mod1 = LowerSchedule(sch1, Array<te::Tensor>{A, B, T}, "main", {}, GlobalVarSupply(NameSupply("")), true);
    std::cout << mod1 << std::endl;

}