//
// Created by 赵丹 on 2023/7/22.
//

#include "test_packedfunc.h"
#include <iostream>
#include <gtest/gtest.h>
#include "tvm/runtime/registry.h"
#include "tvm/runtime/c_runtime_api.h"

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

TEST(PackedFunc, toy_add) {
    const PackedFunc *fp = Registry::Get("toy_add");
    EXPECT_NE(fp, nullptr);
    double a = 1.5;
    double b = 4.1;
    double res = (*fp)(a, b);
    EXPECT_DOUBLE_EQ(res, a + b);
}

TEST(PackedFunc, toy_sub) {
    const PackedFunc *fp = Registry::Get("toy_sub");
    EXPECT_NE(fp, nullptr);
    double a = 1.5;
    double b = 4.1;
    double res = (*fp)(a, b);
    EXPECT_DOUBLE_EQ(res, a - b);
}

TEST(PackedFunc, ListGlobalFuncNames) {
    GTEST_SKIP();
    int global_func_num;
    const char **plist;
    EXPECT_EQ(TVMFuncListGlobalNames(&global_func_num, &plist), 0);

    LOG_INFO << "global function num: " << global_func_num;
    for (int i = 0; i < global_func_num; i++) {
        std::cout << plist[i] << std::endl;
    }
}

TEST(PackedFunc, ListTypeTable) {
    GTEST_SKIP();
    const PackedFunc *fp = Registry::Get("runtime.DumpTypeTable");
    ICHECK(fp != nullptr);
    (*fp)(0);
}
