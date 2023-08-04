//
// Created by 赵丹 on 2023/7/23.
//

//#include "test_relay_ir.h"
#include "utils.h"
#include "make_conv.h"
#include "gtest/gtest.h"
#include "tvm/relay/expr.h"
#include "tvm/relay/function.h"
#include "tvm/relay/analysis.h"
#include "tvm/runtime/device_api.h"
#include "tvm/runtime/registry.h"

using namespace tvm;

relay::Constant generate_constant_node(int rows, int cols, DataType dtype) {
    ICHECK(dtype.is_int()) << "This data type is not supported now.";
    auto *data = new int32_t[rows * cols];
    random_matrix<int32_t>(data, rows, cols);

    std::cout << "the original data:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    DLTensor tensor;
    ShapeTuple shape({rows, cols});
    tensor.data = data;
    tensor.ndim = static_cast<int>(shape.size());
    tensor.shape = const_cast<ShapeTuple::index_type *>(shape.data());
    tensor.dtype = dtype.operator DLDataType();
    tensor.strides = nullptr;
    tensor.byte_offset = runtime::kAllocAlignment -
                         reinterpret_cast<size_t>(static_cast<char *>(tensor.data)) % runtime::kAllocAlignment;
    tensor.device = DLDevice{kDLCPU, 0};
    //    size_t mod = reinterpret_cast<size_t>(static_cast<char*>(tensor.data) + tensor.byte_offset) % kAllocAlignment;
    //    std::cout << "the mod: " << mod << std::endl;
    runtime::NDArray x = runtime::NDArray::FromExternalDLTensor(tensor);

    return relay::Constant(x, Span());
}

void test_constant_expr() {
    int rows = 4;
    int cols = 3;

    relay::Constant const1 = generate_constant_node(rows, cols, DataType::Int(32));

    std::cout << "the constant data:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << static_cast<int32_t *>(const1->data->data)[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    const PackedFunc *make_relu = runtime::Registry::Get("relay.op.nn._make.relu");
    relay::Call call_relu1 = (*make_relu)(const1);
    relay::Call call_relu2 = (*make_relu)(const1);
    std::cout << "relu1 op addr: " << &call_relu1->op << std::endl;
    std::cout << "relu2 op addr: " << &call_relu2->op << std::endl;
    delete[] static_cast<int32_t *>(const1->data->data);
}

void test_let_expr() {
    int rows = 4;
    int cols = 3;
    TensorType TT({rows, cols}, DataType::Int(32));
    relay::Var var("var", TT);

}

IRModule CreateRelayGraph1() {
    auto MakeAdd = [](const relay::Expr &lhs, const relay::Expr &rhs) {
        const Op &add_op = Op::Get("add");
        return relay::Call(add_op, {lhs, rhs});
    };
    relay::Var x1 = relay::Var("x1",
                               TensorType({1, 16, 64, 64},
                                          DataType::Float(32)));
    relay::Constant c1 = relay::Constant(runtime::NDArray::Empty({1, 16, 64, 64},
                                                                 {kDLFloat, 32, 1},
                                                                 {kDLCPU, 0}));
    relay::Var w1 = relay::Var("w1", TensorType());
    relay::Expr x2 = MakeAdd(x1, c1);
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
    relay::Expr x4 = MakeAdd(x3, c2);
    relay::Expr x5 = MakeAdd(x3, x4);
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
    relay::Expr x8 = MakeAdd(x6, x7);
    relay::Function foo = relay::Function(relay::FreeVars(x8), x8, relay::Type(), {});
    return IRModule::FromExpr(foo);
}

TEST(Relay, ListAllOpNames) {
    GTEST_SKIP();
    const PackedFunc *fp = runtime::Registry::Get("ir.ListOpNames");
    Array<String> op_names = (*fp)();
    LOG_INFO << "List all " << op_names.size() << " ops:";
    for (const auto &item: op_names) {
        std::cout << item << std::endl;
    }
}

TEST(Relay, PrintGraph) {
    auto func = []() -> void {
        runtime::NDArray c_data = runtime::NDArray::Empty(
                {2, 3},
                {kDLFloat, 32, 1},
                {kDLCPU, 0}
        );
        relay::Constant c1 = relay::Constant(c_data);
        auto make_add = [](const relay::Expr &lhs, const relay::Expr &rhs) {
            const Op &add_op = Op::Get("add");
            return relay::Call(add_op, {lhs, rhs});
        };

        relay::Call y1 = make_add(c1, c1);
        for (int i = 0; i < 5; i++) {
            y1 = make_add(c1, y1);
        }
        relay::Function foo = relay::Function({}, y1, relay::Type(), {});
        IRModule mod = IRModule::FromExpr(foo);
        std::string result = relay::AsText(mod);
        string_to_file("relay_graph.txt", result);
    };
    ASSERT_EXIT((func(), exit(0)), testing::ExitedWithCode(0), ".*");
}

TEST(Relay, Graph1) {
    IRModule mod = CreateRelayGraph1();
    std::string result = relay::AsText(mod, false);
    string_to_file("relay_graph1.txt", result);
    ASSERT_GT(result.size(), 0);
}