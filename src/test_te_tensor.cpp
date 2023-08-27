//
// Created by 赵丹 on 2023/8/25.
//
#include "gtest/gtest.h"
#include "tvm/te/tensor.h"
#include "tvm/te/operation.h"
#include "tvm/tir/var.h"
#include "tvm/tir/op.h"

using namespace tvm;

TEST(TE, Tensor) {
    auto m = tir::SizeVar("m");
    auto n = tir::SizeVar("n");
    auto l = tir::SizeVar("l");

    auto A = te::placeholder({m, l}, DataType::Float(32), "A");
    auto B = te::placeholder({n, l}, DataType::Float(32), "B");
    auto fcompute = [&](const Array<tir::Var> &idx) {
        ICHECK(idx.size() == 3);
        auto A_indices = Array<tir::Var>{idx[0], idx[2]};
        auto B_indices = Array<tir::Var>{idx[1], idx[2]};
        return A(A_indices) * B(B_indices);
    };
    auto T = te::compute({m, n, l}, fcompute, "test.compute");
    auto body = Downcast<te::ComputeOp>(T->op)->body;
    ASSERT_TRUE(A->op->IsInstance<te::PlaceholderOpNode>());
    std::cout << "The compute tensor:\n"
              << T << std::endl;
    std::cout << "Tensor shape: " << T->shape << std::endl;
    std::cout << "The tensor compute body:\n"
              << body << std::endl;
}

TEST(TE, ZeroRank) {
    auto m = tir::SizeVar("m");
    auto A = te::placeholder({m}, DataType::Float(32), "A");
    auto scale = te::placeholder({}, DataType::Float(32), "s");
//    auto dom = Range{0, m};
    auto k = te::reduce_axis(Range(0, m), "k");
    auto fcompute = [&](const Array<tir::Var>& idx) {
        ICHECK(idx.size() == 0);

    };
//    auto T = te::compute({}, fcompute, "test.zero_rank");
}

TEST(TE, Reduce) {
    auto m = tir::SizeVar("m");

}