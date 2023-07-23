//
// Created by 赵丹 on 2023/7/23.
//

#include "test_relay_ir.h"

#include <random>
#include <vector>

#include "tvm/relay/expr.h"
#include "tvm/runtime/registry.h"

using namespace tvm::runtime;
using namespace tvm::relay;

void random_matrix(int64_t *matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int64_t> dist(0, 20);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = dist(gen);
        }
    }
}

void test_constant() {
    //    const char *name = "relay.ir.Constant";
    //    const PackedFunc *fp = Registry::Get(name);
    //    NDArray xx = NDArray::Empty(
    //            {4, 3},
    //            DLDataType{kDLFloat, 32, 1},
    //            DLDevice{kDLCPU, 0}
    //    );

    int rows = 4;
    int cols = 3;
    auto *data = new int64_t[rows * cols];
    random_matrix(data, rows, cols);
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
    tensor.dtype = DLDataType{kDLInt, 64, 1};
    tensor.strides = nullptr;
    tensor.byte_offset = 0;
    tensor.device = DLDevice{kDLCPU, 0};
    size_t mod = reinterpret_cast<size_t>(static_cast<char*>(tensor.data) + tensor.byte_offset) % 64;
    std::cout << "the mod: " << mod << std::endl;
    NDArray x = NDArray::FromExternalDLTensor(tensor);

    const char *name = "relay.ir.Constant";
    const PackedFunc *fp = Registry::Get(name);
    Constant const1 = (*fp)(x, Span());
    Constant const2(x);

    std::cout << "const1 dim: " << const1->data->ndim << std::endl;
    std::cout << "const2 dim: " << const2->data->ndim << std::endl;

    delete[] data;
}
