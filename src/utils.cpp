//
// Created by 赵丹 on 2023/8/3.
//

#include "utils.h"

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