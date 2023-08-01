#include "test_packedfunc.h"
#include "test_relay_ir.h"
#include <gtest/gtest.h>

int main() {
//    ListTypeTable();
//    ListGlobalFuncNames();
    ListAllOpNames();
    test_constant_expr();

    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
