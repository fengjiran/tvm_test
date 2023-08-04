#include "test_relay_ir.h"
#include "gtest/gtest.h"

int main() {
    test_constant_expr();
    testing::InitGoogleTest();
//    testing::FLAGS_gtest_filter = "PackedFunc.toy_add:PackedFunc.toy_sub";
    return RUN_ALL_TESTS();
}
