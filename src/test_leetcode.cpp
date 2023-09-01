//
// Created by 赵丹 on 2023/8/24.
//
#include "gtest/gtest.h"

TEST(Sort, SortCmpFunc) {
    std::vector<std::vector<int>> edges{{0, 1, 2},
                                        {2, 3, 8},
                                        {3, 4, 1}};
    std::sort(edges.begin(), edges.end(),
              [](std::vector<int> &a, std::vector<int> &b) { return a[2] < b[2]; });
    for (auto &edge: edges) {
        std::cout << edge[0] << ' ' << edge[1] << ' ' << edge[2] << std::endl;
    }

    std::vector<int> a{0, 2, 1, 1};
    std::sort(a.begin(), a.end());
    a.resize(std::unique(a.begin(), a.end()) - a.begin());
    for (int x: a) {
        std::cout << x << ", ";
    }
}

TEST(Sort, SortArray) {
    std::vector<std::vector<int>> envelops = {{5, 4},
                                              {6, 4},
                                              {6, 7},
                                              {2, 3}};
    auto cmp = [](std::vector<int> &a, std::vector<int> &b) {
        return a[0] == b[0] ? b[1] < a[1] : a[0] < b[0];
    };

    std::sort(envelops.begin(), envelops.end(), cmp);
    for (auto &edge: envelops) {
        std::cout << edge[0] << ' ' << edge[1] << std::endl;
    }
}
