//
// Created by 赵丹 on 2023/8/24.
//
#include "gtest/gtest.h"

TEST(Sort, SortCmpFunc) {
    std::vector<std::vector<int>> edges{{0, 1, 2},
                                        {2, 3, 8},
                                        {3, 4, 1}};
    std::sort(edges.begin(), edges.end(),
              [](std::vector<int>& a, std::vector<int>& b) { return a[2] < b[2]; });
    for (auto & edge : edges){
        std::cout<< edge[0] << ' ' << edge[1] << ' ' << edge[2] << std::endl;
    }
}