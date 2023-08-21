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

bool check_json_roundtrip(const ObjectRef &expr) {
//    std::string json_str = SaveJSON(expr);
//    std::cout << "json str:\n" << json_str;
    const PackedFunc *check_se = runtime::Registry::Get("node.StructuralEqual");
    ObjectRef back = LoadJSON(SaveJSON(expr));
    std::cout << relay::AsText(expr, false);
    std::cout << relay::AsText(back, false);
    return (*check_se)(back, expr, true, true);
//    Optional<ObjectPathPair> first_mismatch;
//    return SEqualHandlerDefault(false, &first_mismatch, false).Equal(expr, back, false);
}
