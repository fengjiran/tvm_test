#include "gtest/gtest.h"

int main() {
    testing::InitGoogleTest();
//    testing::FLAGS_gtest_filter = "PackedFunc.toy_add:PackedFunc.toy_sub";
    return RUN_ALL_TESTS();
}
