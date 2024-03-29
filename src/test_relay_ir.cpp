//
// Created by 赵丹 on 2023/7/23.
//

#include "utils.h"
#include "make_op.h"
#include "build_relay_model.h"
#include "gtest/gtest.h"
#include "tvm/relay/expr.h"
#include "tvm/relay/function.h"
#include "tvm/runtime/device_api.h"
#include "tvm/runtime/registry.h"
#include "tvm/node/serialization.h"
#include "tvm/tir/function.h"

using namespace tvm;

relay::Constant generate_constant_node(int rows, int cols, DataType dtype) {
    ICHECK(dtype.is_int()) << "This data type is not supported now.";
    auto *data = new int32_t[rows * cols];
    random_matrix<int32_t>(data, rows, cols);

//    std::cout << "the original data:\n";
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            std::cout << data[i * cols + j] << " ";
//        }
//        std::cout << std::endl;
//    }

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

void test_let_expr() {
    int rows = 4;
    int cols = 3;
    TensorType TT({rows, cols}, DataType::Int(32));
    relay::Var var("var", TT);

}

TEST(Relay, ConstantExpr) {
    int rows = 4;
    int cols = 3;

    relay::Constant const1 = generate_constant_node(rows, cols, DataType::Int(32));
    std::string const_seri = SaveJSON(const1);
//    ICHECK(tvm::StructuralEqual()(const1, LoadJSON(const_seri)));
    std::string res = relay::AsText(const1, false);
    std::cout << res << std::endl;

//    std::cout << "the constant data:\n";
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            std::cout << static_cast<int32_t *>(const1->data->data)[i * cols + j] << " ";
//        }
//        std::cout << std::endl;
//    }

    const PackedFunc *make_relu = runtime::Registry::Get("relay.op.nn._make.relu");
    relay::Call call_relu1 = (*make_relu)(const1);
    relay::Call call_relu2 = (*make_relu)(const1);
    ASSERT_TRUE(call_relu1->op.same_as(call_relu2->op));
//    std::cout << "relu1 op addr: " << &call_relu1->op << std::endl;
//    std::cout << "relu2 op addr: " << &call_relu2->op << std::endl;
    delete[] static_cast<int32_t *>(const1->data->data);
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

TEST(Relay, PrintToyModel) {
    auto func = []() -> void {
        runtime::NDArray c_data = runtime::NDArray::Empty(
                {2, 3},
                {kDLFloat, 32, 1},
                {kDLCPU, 0}
        );
        relay::Constant c1 = relay::Constant(c_data);
        relay::Expr y1 = relay::MakeAdd(c1, c1);
        for (int i = 0; i < 5; i++) {
            y1 = relay::MakeAdd(c1, y1);
        }
        relay::Function foo = relay::Function({}, y1, relay::Type(), {});
        IRModule mod = IRModule::FromExpr(foo);
        std::string result = relay::AsText(mod, false);
        string_to_file("relay_toy_model.txt", result);
    };
    ASSERT_EXIT((func(), exit(0)), testing::ExitedWithCode(0), ".*");
}

TEST(Relay, RelayModel1) {
    IRModule mod = BuildRelayModel_1();
    std::string result = relay::AsText(mod, false);
    string_to_file("relay_model_1.txt", result);
    ASSERT_GT(result.size(), 0);
}

TEST(Relay, RelayModel2) {
    relay::Var x = relay::Var("x",
                              TensorType({1, 3, 64, 64},
                                         DataType::Float(32)));
    relay::Expr output = BuildConvBNRelu(x, 3, 16, 1, 1, 1, 1, 3);
    auto expr = Downcast<relay::Call>(output)->op.as<GlobalVarNode>();
    ASSERT_TRUE(expr == nullptr);
    relay::Function foo = relay::Function(relay::FreeVars(output), output, relay::Type(), {});
    IRModule mod = IRModule::FromExpr(foo);
    std::string result = relay::AsText(mod, false);
    string_to_file("relay_conv_bn_relu.txt", result);
    ASSERT_GT(result.size(), 0);
}
