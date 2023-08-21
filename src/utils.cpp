//
// Created by 赵丹 on 2023/8/3.
//

#include "utils.h"
#include "tvm/node/serialization.h"
#include "tvm/node/structural_equal.h"

using namespace tvm;

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

void check_json_roundtrip(const runtime::ObjectRef& expr) {
    std::string json_str = SaveJSON(expr);
    runtime::ObjectRef back = LoadJSON(json_str);
    ICHECK(StructuralEqual()(expr, back));
}