//
// Created by 赵丹 on 2023/8/4.
//

#include "build_relay_model.h"
#include "make_op.h"
#include "tvm/relay/expr.h"
#include "tvm/relay/function.h"
#include "tvm/relay/analysis.h"

using namespace tvm;

IRModule BuildRelayModel_1() {
//    auto MakeAdd = [](const relay::Expr &lhs, const relay::Expr &rhs) {
//        const Op &add_op = Op::Get("add");
//        return relay::Call(add_op, {lhs, rhs});
//    };
    relay::Var x1 = relay::Var("x1",
                               TensorType({1, 16, 64, 64},
                                          DataType::Float(32)));
    relay::Constant c1 = relay::Constant(runtime::NDArray::Empty({1, 16, 64, 64},
                                                                 {kDLFloat, 32, 1},
                                                                 {kDLCPU, 0}));
    relay::Var w1 = relay::Var("w1", TensorType());
    relay::Expr x2 = relay::MakeAdd(x1, c1);
    relay::Expr x3 = relay::MakeConv<relay::Conv2DAttrs>(x2, w1,
                                                         {1, 1}, {0, 0},
                                                         {1, 1}, 1,
                                                         16, {1, 1},
                                                         "NCHW", "OIHW",
                                                         "", DataType(),
                                                         "nn.conv2d");
    relay::Constant c2 = relay::Constant(runtime::NDArray::Empty({1},
                                                                 {kDLFloat, 32, 1},
                                                                 {kDLCPU, 0}));
    relay::Expr x4 = relay::MakeAdd(x3, c2);
    relay::Expr x5 = relay::MakeAdd(x3, x4);
    relay::Var w2 = relay::Var("w2", TensorType());
    relay::Var w3 = relay::Var("w3", TensorType());
    relay::Expr x6 = relay::MakeConv<relay::Conv2DAttrs>(x5, w2,
                                                         {1, 1}, {0, 0},
                                                         {1, 1}, 1,
                                                         16, {1, 1},
                                                         "NCHW", "OIHW",
                                                         "", DataType(),
                                                         "nn.conv2d");
    relay::Expr x7 = relay::MakeConv<relay::Conv2DAttrs>(x5, w3,
                                                         {1, 1}, {0, 0},
                                                         {1, 1}, 1,
                                                         16, {1, 1},
                                                         "NCHW", "OIHW",
                                                         "", DataType(),
                                                         "nn.conv2d");
    relay::Expr x8 = relay::MakeAdd(x6, x7);
    relay::Function foo = relay::Function(relay::FreeVars(x8), x8, relay::Type(), {});
    return IRModule::FromExpr(foo);
}

IRModule BuildRelayModel_2() {
    //
}

IRModule BuildResBlock() {
    //
}
