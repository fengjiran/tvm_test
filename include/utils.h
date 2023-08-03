//
// Created by 赵丹 on 2023/8/3.
//

#ifndef TVM_TEST_UTILS_H
#define TVM_TEST_UTILS_H

#include <random>
#include <vector>
#include <fstream>
#include <iostream>

template<typename T>
void random_matrix(T *matrix, int rows, int cols);

int string_to_file(const std::string &file_name, const std::string &str);

#endif //TVM_TEST_UTILS_H
