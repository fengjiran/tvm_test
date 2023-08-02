//
// Created by 赵丹 on 2023/7/23.
//

#include "test_relay_ir.h"

#include <random>
#include <vector>
#include <fstream>
#include <gtest/gtest.h>

#include "tvm/relay/expr.h"
#include "tvm/relay/op.h"
#include "tvm/relay/function.h"
#include "tvm/runtime/device_api.h"
#include "tvm/runtime/registry.h"

using namespace tvm;

template<typename T>
void random_matrix(T *matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    T low = -20;
    T high = 20;

    std::uniform_real_distribution<float> dist(low, high);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = static_cast<T>(dist(gen));
        }
    }
}

int string_to_file(const std::string &file_name, const std::string &str) {
    std::ofstream outfile;
    outfile.open(file_name);
    if (!outfile.is_open()) {
        std::cout << "Open file failed!\n";
        return -1;
    }
    outfile << str << std::endl;
    outfile.close();
    return 0;
}

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
//    const PackedFunc *fp = Registry::Get("relay.ir.Constant");
//    Constant const1 = (*fp)(x, Span());
//    Constant const2(x, Span());

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
//    const PackedFunc *fp = Registry::Get("relay.ir.Let");
    int rows = 4;
    int cols = 3;
    TensorType TT({rows, cols}, DataType::Int(32));
    relay::Var var("var", TT);

}

void ListAllOpNames() {
    const PackedFunc *fp = runtime::Registry::Get("ir.ListOpNames");
    Array<String> op_names = (*fp)();
    LOG_INFO << "List all " << op_names.size() << " ops:";
    for (const auto &item: op_names) {
        std::cout << item << std::endl;
    }
}

TEST(Relay, PrintGraph) {
    auto func = []() -> void {
        relay::Op add_op = relay::Op::Get("add");
        runtime::NDArray c_data = runtime::NDArray::Empty(
                {1, 2, 3},
                {kDLFloat, 32, 1},
                {kDLCPU, 0}
        );
        relay::Constant c1 = relay::Constant(c_data);
        const PackedFunc *make_add_op = runtime::Registry::Get("relay.op._make.add");
        ICHECK_NE(make_add_op, nullptr);

        relay::Call y1 = (*make_add_op)(c1, c1);
        for (int i = 0; i < 5; i++) {
            y1 = (*make_add_op)(c1, y1);
        }
        relay::Function foo = relay::Function({}, y1, relay::Type(), {});
        IRModule mod = IRModule::FromExpr(foo);
        std::string result = relay::AsText(mod);
        string_to_file("relay_graph.txt", result);
        ASSERT_GT(0, result.size());
    };
    ASSERT_EXIT((func(), exit(0)), testing::ExitedWithCode(0), ".*");
}

TEST(Relay, Graph1) {
    auto func = []() -> void {
        relay::Var input = relay::Var("input",
                                      TensorType({1, 16, 64, 64},
                                                 DataType::Float(32)));
        relay::Constant w1 = relay::Constant(runtime::NDArray::Empty({1, 16, 64, 64},
                                                                     {kDLFloat, 32, 1},
                                                                     {kDLCPU, 0}));
        relay::Op add_op = relay::Op::Get("add");
    };
}