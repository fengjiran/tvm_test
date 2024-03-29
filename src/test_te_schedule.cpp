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
    LOG_INFO << "Leaf iter var num before reorder: " << stage->leaf_iter_vars.size();
    auto mod1 = LowerSchedule(sch1, Array<te::Tensor>{A, B, T}, "main", {}, GlobalVarSupply(NameSupply("")), true);
    std::cout << mod1 << std::endl;

    tir::IterVar k0_outer;
    tir::IterVar k0_inner;
    tir::IterVar k1_outer;
    tir::IterVar k1_inner;
    auto axes = Downcast<te::ComputeOp>(T->op)->axis;
    ASSERT_EQ(axes.size(), 2);
    stage.split(axes[0], 32, &k0_outer, &k0_inner)
            .split(axes[1], 32, &k1_outer, &k1_inner)
            .reorder({k0_outer, k1_outer, k1_inner, k0_inner});
    LOG_INFO << "Print schedule after reorder:";
    LOG_INFO << "All iter var num after reorder: " << stage->all_iter_vars.size();
    LOG_INFO << "Leaf iter var num after reorder: " << stage->leaf_iter_vars.size();
    std::cout << LowerSchedule(sch1, Array<te::Tensor>{A, B, T}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
}

TEST(TESchedule, tile) {
    auto m = tir::SizeVar("m");
    auto n = tir::SizeVar("n");
    auto k = tir::SizeVar("k");
    auto A = te::placeholder(Array<PrimExpr>{m, k}, DataType::Float(32), "A");
    auto B = te::placeholder(Array<PrimExpr>{k, n}, DataType::Float(32), "B");
    auto red_k = te::reduce_axis(Range(0, k), "red_k");
    auto T = te::compute(Array<PrimExpr>{m, n}, [&](const tir::Var &i, const tir::Var &j) {
        return A(i, red_k) * B(red_k, j);
    });

    auto sch = te::create_schedule(Array<te::Operation>{T->op});
    LOG_INFO << "Print schedule before tile:";
    ASSERT_EQ(sch->stages.size(), 3);
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B, T}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
    auto stage = sch[T];
    auto axes = Downcast<te::ComputeOp>(T->op)->axis;
    ASSERT_EQ(axes.size(), 2);

    tir::IterVar k0_outer;
    tir::IterVar k0_inner;
    tir::IterVar k1_outer;
    tir::IterVar k1_inner;
    stage.tile(axes[0], axes[1], 10, 5, &k0_outer, &k1_outer, &k0_inner, &k1_inner);
    LOG_INFO << "Print schedule after tile:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B, T}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
}

TEST(TESchedule, fuse) {
    auto m = tir::SizeVar("m");
    auto n = tir::SizeVar("n");
    auto A = te::placeholder(Array<PrimExpr>{m, n}, DataType::Float(32), "A");
    auto T = te::compute(Array<PrimExpr>{m, n}, [&](const tir::Var &i, const tir::Var &j) {
        return A(i, j);
    });

    auto sch = te::create_schedule(Array<te::Operation>{T->op});
    auto axes = Downcast<te::ComputeOp>(T->op)->axis;
    ASSERT_EQ(axes.size(), 2);

    tir::IterVar k0_outer;
    tir::IterVar k0_inner;
    tir::IterVar k1_outer;
    tir::IterVar k1_inner;
    sch[T].tile(axes[0], axes[1], 10, 5, &k0_outer, &k1_outer, &k0_inner, &k1_inner);
    tir::IterVar fused;
    sch[T].fuse(k0_outer, k1_outer, &fused);
}

TEST(TESchedule, CacheRead) {
    auto m = tir::SizeVar("m");
    auto A = te::placeholder(Array<PrimExpr>{m, m}, DataType::Float(32), "A");
    auto T = topi::sum(A, {0});
    auto sch = te::create_schedule(Array<te::Operation>{T->op});
    LOG_INFO << "Print schedule before cache read:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, T}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
    sch.cache_read(A, "shared", {T->op});
    LOG_INFO << "Print schedule after cache read:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, T}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
}

TEST(TESchedule, CacheWrite) {
    auto m = tir::SizeVar("m");
    auto A = te::placeholder(Array<PrimExpr>{m, m}, DataType::Float(32), "A");
    auto T = topi::sum(A, {0});
    auto sch = te::create_schedule(Array<te::Operation>{T->op});
    LOG_INFO << "Print schedule before cache write:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, T}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
    sch.cache_write(T, "local");
    std::cout << "----------------------------cut line-------------------------------\n";
    LOG_INFO << "Print schedule after cache write:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, T}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
}

TEST(TESchedule, StorageAlign) {
    auto m = tir::SizeVar("m");
    auto n = tir::SizeVar("n");
    auto A = te::placeholder(Array<PrimExpr>{m, n}, DataType::Float(32), "A");
    auto T = topi::sum(A, {1});
    auto sch = te::create_schedule(Array<te::Operation>{T->op});
    auto AA = sch.cache_read(A, "shared", {T->op});
    LOG_INFO << "Print schedule before storage align:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, T}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
    auto axes = Downcast<te::ComputeOp>(AA->op)->axis;
    sch[T].storage_align(axes[0], 100, 8);
    std::cout << "----------------------------cut line-------------------------------\n";
    LOG_INFO << "Print schedule after storage align:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, T}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
}

TEST(TESchedule, ComputeAt) {
    auto m = tir::SizeVar("m");
    auto A = te::placeholder(Array<PrimExpr>{m}, DataType::Float(32), "A");
    auto B = te::compute(Array<PrimExpr>{m}, [&](const tir::Var &i) {
        return A(i) + 1;
    });
    auto C = te::compute(Array<PrimExpr>{m}, [&](const tir::Var &i) {
        return B(i) * 2;
    });
    auto sch = te::create_schedule(Array<te::Operation>{C->op});
    LOG_INFO << "Print schedule before compute at:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B, C}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
    sch[B].compute_at(sch[C], Downcast<te::ComputeOp>(C->op)->axis[0]);
    LOG_INFO << "Print schedule after compute at:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B, C}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
}

TEST(TESchedule, ComputeInline) {
    auto m = tir::SizeVar("m");
    auto A = te::placeholder(Array<PrimExpr>{m}, DataType::Float(32), "A");
    auto B = te::compute(Array<PrimExpr>{m}, [&](const tir::Var &i) {
        return A(i) + 1;
    });
    auto C = te::compute(Array<PrimExpr>{m}, [&](const tir::Var &i) {
        return B(i) * 2;
    });
    auto sch = te::create_schedule(Array<te::Operation>{C->op});
    LOG_INFO << "Print schedule before compute inline:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B, C}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
    sch[B].compute_inline();
    LOG_INFO << "Print schedule after compute inline:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B, C}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
}

TEST(TESchedule, Vectorize) {
    auto m = tir::SizeVar("m");
    auto n = tir::SizeVar("n");
    auto A = te::placeholder(Array<PrimExpr>{m, n}, DataType::Float(32), "A");
    auto B = te::placeholder(Array<PrimExpr>{m, n}, DataType::Float(32), "B");
    auto C = te::compute(Array<PrimExpr>{m, n}, [&](const tir::Var &i, const tir::Var &j) {
        return A(i, j) + B(i, j);
    });
    te::Schedule sch = te::create_schedule(Array<te::Operation>{C->op});
    auto axes = Downcast<te::ComputeOp>(C->op)->axis;
    ASSERT_EQ(axes.size(), 2);

    tir::IterVar k0_outer;
    tir::IterVar k0_inner;
    tir::IterVar k1_outer;
    tir::IterVar k1_inner;
    sch[C].tile(axes[0], axes[1], 32, 32, &k0_outer, &k1_outer, &k0_inner, &k1_inner);
    LOG_INFO << "Print schedule before vectorize:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B, C}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
    sch[C].vectorize(k1_inner);
    LOG_INFO << "Print schedule after vectorize:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B, C}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
}

TEST(TESchedule, Unroll) {
    int m = 1024;
    auto A = te::placeholder(Array<PrimExpr>{m, m}, DataType::Float(32), "A");
    auto B = te::placeholder(Array<PrimExpr>{m, m}, DataType::Float(32), "B");
    auto C = te::compute(Array<PrimExpr>{m, m}, [&](const tir::Var &i, const tir::Var &j) {
        return A(i, j) + B(i, j);
    });
    te::Schedule sch = te::create_schedule(Array<te::Operation>{C->op});
    auto axes = Downcast<te::ComputeOp>(C->op)->axis;
    ASSERT_EQ(axes.size(), 2);

    tir::IterVar k0_outer;
    tir::IterVar k0_inner;
    sch[C].split(axes[0], 4, &k0_outer, &k0_inner);
    LOG_INFO << "Print schedule before unroll:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B, C}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
    sch[C].unroll(k0_inner);
    LOG_INFO << "Print schedule after unroll:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B, C}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
}

TEST(TESchedule, Bind) {
    int m = 1024;
    auto A = te::placeholder(Array<PrimExpr>{m}, DataType::Float(32), "A");
    auto B = topi::sum(A, {0});
    te::Schedule sch = te::create_schedule(Array<te::Operation>{B->op});
    auto reduce_axes = Downcast<te::ComputeOp>(B->op)->reduce_axis;
    ASSERT_EQ(reduce_axes.size(), 1);

    tir::IterVar k0_outer;
    tir::IterVar k0_inner;
    sch[B].split(reduce_axes[0], 32, &k0_outer, &k0_inner);
    LOG_INFO << "Print schedule before bind:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
    tir::IterVar block_iter_var(Range(), tir::Var(), tir::kThreadIndex, "blockIdx.x");
    tir::IterVar thread_iter_var(Range(), tir::Var(), tir::kThreadIndex, "threadIdx.x");
    sch[B].bind(k0_outer, block_iter_var);
    sch[B].bind(k0_inner, thread_iter_var);
    LOG_INFO << "Print schedule after bind:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
}

TEST(TESchedule, Parallel) {
    int m = 1024;
    int n = 1024;
    auto A = te::placeholder(Array<PrimExpr>{m, n}, DataType::Float(32), "A");
    auto B = topi::sum(A, {1});
    auto sch = te::create_schedule(Array<te::Operation>{B->op});
    LOG_INFO << "Print schedule before parallel:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
    std::cout << "----------------------------cut line-------------------------------\n";
    sch[B].parallel(Downcast<te::ComputeOp>(B->op)->reduce_axis[0]);
    LOG_INFO << "Print schedule after parallel:";
    std::cout << LowerSchedule(sch, Array<te::Tensor>{A, B}, "main", {}, GlobalVarSupply(NameSupply("")), true)
              << std::endl;
}

TEST(TESchedule, NaiveConv) {
    auto n = tir::SizeVar("n");
    int w = 3;
    auto input = te::placeholder(Array<PrimExpr>{n, n}, DataType::Float(32), "input");
    auto filter = te::placeholder(Array<PrimExpr>{w, w}, DataType::Float(32), "filter");
    auto ki = te::reduce_axis(Range(0, w), "ki");
    auto kj = te::reduce_axis(Range(0, w), "kj");
    auto output = te::compute(Array<PrimExpr>{n - w + 1, n - w + 1}, [&](const tir::Var &i, const tir::Var &j) {
        return tvm::sum(input(i + ki, j + kj) * filter(ki, kj), {ki, kj});
    });
    auto sch = te::create_schedule(Array<te::Operation>{output->op});
    LOG_INFO << "Print naive conv:";
    std::cout
            << LowerSchedule(sch, Array<te::Tensor>{input, filter, output}, "main", {}, GlobalVarSupply(NameSupply("")),
                             true)
            << std::endl;
}
