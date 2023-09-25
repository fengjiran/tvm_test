//
// Created by richard on 9/25/23.
//
#include "gtest/gtest.h"
#include "tvm/ir/name_supply.h"

using namespace tvm;

TEST(IR, NameSupply) {
    std::string prefix = "test";
    std::unordered_map<std::string, int> name_map;
    auto name_supply = NameSupply(prefix, name_map);
    auto x = name_map.insert({prefix, 0});
    ASSERT_EQ(x.first->first, "test");
    ASSERT_EQ(x.second, true);
}
