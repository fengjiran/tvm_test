//
// Created by 赵丹 on 2023/8/3.
//

#include "utils.h"

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
