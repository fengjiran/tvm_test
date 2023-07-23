//
// Created by 赵丹 on 2023/7/23.
//

#include "test_relay_ir.h"
#include "tvm/runtime/registry.h"
#include "tvm/relay/expr.h"
#include <vector>
#include <random>

using namespace tvm::runtime;

void random_matrix(int32_t *matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int32_t> dist(0, 20);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = dist(gen);
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
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    DLTensor tet {matrix};

    delete[] matrix;

}
