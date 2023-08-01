#include "test_packedfunc.h"
#include "test_relay_ir.h"
#include <gtest/gtest.h>

int main() {
//    ListTypeTable();
//    ListGlobalFuncNames();
    ListAllOpNames();
    test_toy_add(1, 4);
    test_toy_sub(1, 4);
    test_constant_expr();

    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}
