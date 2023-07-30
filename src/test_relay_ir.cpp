//
// Created by 赵丹 on 2023/7/23.
//

#include "test_relay_ir.h"

#include <random>
#include <vector>

#include "tvm/relay/expr.h"
#include "tvm/relay/op.h"
#include "tvm/runtime/registry.h"
#include "tvm/runtime/device_api.h"

using namespace tvm::runtime;
using namespace tvm::relay;

void random_matrix(int32_t *matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(-20, 20);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = dist(gen);
        }
    }
}

Constant generate_constant_node(int rows, int cols) {
    auto *data = new int32_t[rows * cols];
    random_matrix(data, rows, cols);

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
    tensor.dtype = DLDataType{kDLInt, 32, 1};
    tensor.strides = nullptr;
    tensor.byte_offset = kAllocAlignment - reinterpret_cast<size_t>(static_cast<char *>(tensor.data)) % kAllocAlignment;
    tensor.device = DLDevice{kDLCPU, 0};
//    size_t mod = reinterpret_cast<size_t>(static_cast<char*>(tensor.data) + tensor.byte_offset) % kAllocAlignment;
//    std::cout << "the mod: " << mod << std::endl;
    NDArray x = NDArray::FromExternalDLTensor(tensor);
    const PackedFunc *fp = Registry::Get("relay.ir.Constant");
    Constant const1 = (*fp)(x, Span());

    return const1;
}

void test_constant() {
    int rows = 4;
    int cols = 3;

    Constant const1 = generate_constant_node(rows, cols);

    std::cout << "the constant data:\n";
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << static_cast<int32_t *>(const1->data->data)[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    const PackedFunc *relu_pf = Registry::Get("relay.op.nn._make.relu");
    Call call_relu1 = (*relu_pf)(const1);
    Call call_relu2 = (*relu_pf)(const1);
    std::cout << "relu1 op addr: " << &call_relu1->op << std::endl;
    std::cout << "relu2 op addr: " << &call_relu2->op << std::endl;
    delete[] static_cast<int32_t *>(const1->data->data);
}

void ListAllOpNames() {
    const PackedFunc *fp = Registry::Get("ir.ListOpNames");
    Array<String> op_names = (*fp)();
    LOG_INFO << "List all " << op_names.size() << " ops:";
    for (const auto &item: op_names) {
        std::cout << item << std::endl;
    }
}
