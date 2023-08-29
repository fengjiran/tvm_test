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

// like topi::sum
te::Tensor test_sum(const te::Tensor &data, const te::Tensor &scale, const Array<Integer> &axis, bool keepdims = false,
                    bool atleast1d = false) {
    auto ndim = data.ndim();
    ICHECK_NE(ndim, 0) << "Can not reduce a 0 dim Tensor.";
    ICHECK_EQ(scale.ndim(), 0) << "The dim of scale must be 0.";
    auto real_axis = topi::GetRealAxis(static_cast<int>(ndim), axis);
    auto target_shape = topi::MakeReduceTargetShape(real_axis, data, keepdims, atleast1d);
    auto squeezed_axes = keepdims ? std::vector<int>() : real_axis;
    auto reduce_axes = topi::MakeReduceAxes(real_axis, data);
    auto fcompute = [&](const Array<tir::Var> &indices) {
        if (keepdims) {
            ICHECK_EQ(indices.size(), ndim);
        }
        Array<PrimExpr> eval_range;
        int arg_counter = 0;
        int reduce_counter = 0;
        for (int i = 0; i < ndim; i++) {
            if (std::find(real_axis.begin(), real_axis.end(), i) != real_axis.end()) {
                eval_range.push_back(reduce_axes[reduce_counter++]);
                if (keepdims) {
                    arg_counter++;
                }
                continue;
            }
            eval_range.push_back(indices[arg_counter++]);
        }
        return tvm::sum(data(eval_range) * scale(), reduce_axes, {}, Span());
    };
    return te::compute(target_shape, fcompute, data->op->name + "_reduce", topi::kCommReduce);
}

TEST(TE, GetRealAxis) {
    int ndim = 4;
    Array<Integer> reduce_axis{2, 3};
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
    auto res = test_sum(A, scale, {0});
    std::cout << "The compute tensor:\n"
              << res << std::endl;
    ASSERT_EQ(res.ndim(), 0);

//    auto T = topi::sum(A, {0});
//    std::cout << "The compute tensor:\n"
//              << T << std::endl;
}

TEST(TE, conv1d) {
    auto m = tir::SizeVar("m");
    auto A = te::placeholder({m + 2});
    auto fcompute = [&](const Array<tir::Var> &idx) {
        return A[idx[0]] + A[idx[0] + 1] + A[idx[0] + 2];
    };
    auto T = te::compute({m}, fcompute, "conv1d");
    ASSERT_EQ(T.ndim(), 1);
}

TEST(TE, TensorSlice) {
    auto m = tir::SizeVar("m");
    auto A = te::compute(Array<PrimExpr>{m, m}, [&](const Array<tir::Var> &idx) {
        ICHECK_EQ(idx.size(), 2);
        return FloatImm(DataType::Float(32), 1);
    });
    auto B = te::compute(Array<PrimExpr>{m}, [&](const Array<tir::Var> &idx) {
        ICHECK_EQ(idx.size(), 1);
        return A(Array<PrimExpr>{0, idx[0]}) + A(Array<PrimExpr>{0, idx[0]});
    });
    ASSERT_EQ(A.ndim(), 2);
    ASSERT_EQ(B.ndim(), 1);
    auto x = static_cast<ObjectRef>(m);
    auto y = static_cast<ObjectRef>(A->shape[0]);
    ASSERT_TRUE(x == y);
}

TEST(TE, ReduceMultiAxis) {
    auto m = tir::SizeVar("m");
    auto n = tir::SizeVar("n");
    auto A = te::placeholder(Array<PrimExpr>{m, n}, DataType::Float(32), "A");

}

TEST(TE, Reduce) {
    auto m = tir::SizeVar("m");

}