//
// Created by 赵丹 on 2023/7/23.
//

#include "test_relay_ir.h"
#include "tvm/runtime/registry.h"
#include "tvm/relay/expr.h"
#include <vector>

using namespace tvm::runtime;

void random_matrix(int32_t* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (int32_t)mrand48();
        }
    }
}

void test_constant() {
    const char *name = "relay.ir.Constant";
    const PackedFunc *fp = Registry::Get(name);
    NDArray x = NDArray::Empty(
            {4, 3},
            DLDataType{kDLFloat, 32, 1},
            DLDevice{kDLCPU, 0}
    );

    int rows = 4;
    int cols = 3;
    auto *matrix = new int32_t[rows * cols];
    random_matrix(matrix, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * rows + j] << " ";
        }
        std::cout << std::endl;
    }

    delete [] matrix;

}
