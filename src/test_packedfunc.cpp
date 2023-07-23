//
// Created by 赵丹 on 2023/7/22.
//

#include "test_packedfunc.h"
#include <iostream>
#include "tvm/runtime/registry.h"
#include "tvm/runtime/c_runtime_api.h"

//using namespace tvm;
using namespace tvm::runtime;

template<typename T>
inline T toy_add(T a, T b) {
    return a + b;
}

template<typename T>
inline T toy_sub(T a, T b) {
    return a - b;
}

TVM_REGISTER_GLOBAL("toy_add").set_body([](TVMArgs args, TVMRetValue *rv) -> void {
    *rv = toy_add<double>(args[0], args[1]);
});

TVM_REGISTER_GLOBAL("toy_sub").set_body([](TVMArgs args, TVMRetValue *rv) -> void {
    *rv = toy_sub<double>(args[0], args[1]);
});

void test_toy_add(float a, float b) {
    const char *fname = "toy_add";
//    TVMFunctionHandle fp1;
//    TVMFuncGetGlobal(fname, &fp1);
//    auto *fp2 = static_cast<PackedFunc*>(fp1);
//    double res = (*fp2)(a, b);
//    LOG_INFO << "Result: " << res;
    const PackedFunc *fp = Registry::Get(fname);
    ICHECK(fp != nullptr);
    double res = (*fp)(a, b);
    LOG_INFO << "Result: " << res;
}

void test_toy_sub(float a, float b) {
    const char* fname = "toy_sub";
    const PackedFunc *fp = Registry::Get(fname);
    ICHECK(fp != nullptr);
    double res = (*fp)(a, b);
    LOG_INFO << "Result: " << res;
}

void ListGlobalFuncNames() {
    int global_func_num;
    const char **plist;
    TVMFuncListGlobalNames(&global_func_num, &plist);

    LOG_INFO << "List all " << global_func_num << " global functions:";
    for (int i = 0; i < global_func_num; i++) {
        std::cout << plist[i] << std::endl;
    }
}

void ListTypeTable() {
    String fname = "runtime.DumpTypeTable";
    const PackedFunc *fp = Registry::Get(fname);
    ICHECK(fp != nullptr);
    (*fp)(0);
}