//
// Created by 赵丹 on 2023/8/25.
//
#include "gtest/gtest.h"
#include "tvm/te/tensor.h"
#include "tvm/te/operation.h"
#include "tvm/tir/var.h"
#include "tvm/tir/op.h"
#include "tvm/topi/reduction.h"

using namespace tvm;

std::vector<int> GetRealAxis(int ndim, const Array<Integer> &axis) {
    std::vector<int> real_axis;
    if (!axis.defined()) {
        for (int i = 0; i < ndim; i++) {
            real_axis.push_back(i);
        }
    } else {
        for (auto elem: axis) {
            auto val = elem->value;
            if (val < 0) {
                val += ndim;
            }
            ICHECK_LT(val, ndim);
            ICHECK_GE(val, 0);
            real_axis.push_back(static_cast<int>(val));
        }
        std::sort(real_axis.begin(), real_axis.end());
        real_axis.resize(std::unique(real_axis.begin(), real_axis.end()) - real_axis.begin());
    }
    return real_axis;
}

te::Tensor test_sum(const te::Tensor &data, const Array<Integer> &axis, bool keepdims = false, bool atleast1d = false) {
    auto ndim = data->shape.size();
    ICHECK_NE(ndim, 0) << "Can not reduce a 0 dim Tensor.";
    auto real_axis = topi::GetRealAxis(static_cast<int>(ndim), axis);
    auto target_shape = topi::MakeReduceTargetShape(real_axis, data, keepdims, atleast1d);
    auto squeeze_axes = keepdims ? std::vector<int>() : real_axis;
    auto reduce_axes = topi::MakeReduceAxes(real_axis, data);
    auto fcompute = [&](const Array<tir::Var>& indices) {
        //
    };

}

TEST(TE, GetRealAxis) {
    int ndim = 4;
    Array<Integer> reduce_axis{2, 3};
//    auto real_axis = GetRealAxis(ndim, reduce_axis);
    auto real_axis = topi::GetRealAxis(ndim, reduce_axis);
    std::cout << "\n";
}

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
    auto scale = te::placeholder({}, DataType::Float(32), "scale");
    auto k = te::reduce_axis(Range(0, m), "k");
    auto T = topi::sum(A, {0});
    std::cout << "The compute tensor:\n"
              << T << std::endl;
}

TEST(TE, Reduce) {
    auto m = tir::SizeVar("m");

}