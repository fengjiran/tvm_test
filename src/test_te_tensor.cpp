//
// Created by 赵丹 on 2023/8/25.
//
#include "gtest/gtest.h"
#include "tvm/te/tensor.h"
#include "tvm/te/operation.h"
#include "tvm/tir/var.h"

using namespace tvm;

TEST(TE, Tensor) {
    auto m = tir::SizeVar("m");
    auto n = tir::SizeVar("n");
    auto l = tir::SizeVar("l");

    auto A = te::placeholder({m, l}, DataType::Float(32), "A");
    auto B = te::placeholder({n, l}, DataType::Float(32), "B");
    auto fcompute = [&](const Array<tir::Var> &idx) {
        ICHECK(idx.size() == 3);
        auto i = idx[0];
        auto j = idx[1];
        auto k = idx[2];
        return A(Array<tir::Var>{i, k}) * B(Array<tir::Var>{j, k});
    };
    auto T = te::compute({m, n, l}, fcompute);
}