//
// Created by 赵丹 on 2023/8/4.
//

#include "build_relay_model.h"
#include "make_op.h"
#include "tvm/relay/expr.h"
#include "tvm/relay/function.h"

using namespace tvm;

IRModule BuildRelayModel_1() {
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

relay::Expr BuildConvBNRelu(const relay::Expr &x, int cin, int cout, int stride,
                            int padding, int dilation, int groups, int ksize) {

    relay::Constant w1 = relay::Constant(runtime::NDArray::Empty({cout, cin, ksize, ksize},
                                                                 {kDLFloat, 32, 1},
                                                                 {kDLCPU, 0}));
    relay::Constant gamma = relay::Constant(runtime::NDArray::Empty({cout},
                                                                    {kDLFloat, 32, 1},
                                                                    {kDLCPU, 0}));
    relay::Constant beta = relay::Constant(runtime::NDArray::Empty({cout},
                                                                   {kDLFloat, 32, 1},
                                                                   {kDLCPU, 0}));
    relay::Constant moving_mean = relay::Constant(runtime::NDArray::Empty({cout},
                                                                          {kDLFloat, 32, 1},
                                                                          {kDLCPU, 0}));
    relay::Constant moving_var = relay::Constant(runtime::NDArray::Empty({cout},
                                                                         {kDLFloat, 32, 1},
                                                                         {kDLCPU, 0}));

    relay::Expr x1 = relay::MakeConv<relay::Conv2DAttrs>(x, w1,
                                                         {stride, stride}, {padding, padding},
                                                         {dilation, dilation}, groups,
                                                         cout, {ksize, ksize},
                                                         "NCHW", "OIHW",
                                                         "", DataType(),
                                                         "nn.conv2d");
    relay::Expr x2 = relay::MakeBatchNorm(x1, gamma, beta, moving_mean, moving_var, 1, 1e-5, true, true);
    relay::Expr x3 = relay::MakeRelu(x2);
    return x3;
}

relay::Expr BuildResBasicBlock(relay::Expr x) {
    //
}

relay::Expr BuildAddExpr(const relay::Expr& x, const relay::Expr& y) {
    auto z = relay::MakeAdd(x, y);
    z = relay::MakeRelu(z);
    return z;
}
