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
        return a[0] == b[0] ? a[1] > b[1] : a[0] < b[0];
    };

    std::sort(envelops.begin(), envelops.end(), cmp);
    for (auto &edge: envelops) {
        std::cout << edge[0] << ' ' << edge[1] << std::endl;
    }
}

TEST(DP, CoinChange1) {
    auto coin_change = [](const std::vector<int> &coins, int amount) {
        std::vector<int> dp(amount + 1, amount + 1);
        dp[0] = 0;
        for (int i = 1; i < amount + 1; i++) {
            for (int coin: coins) {
                if (i - coin < 0) {
                    continue;
                }
                dp[i] = std::min(dp[i], dp[i - coin] + 1);
            }
        }
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    };

    ASSERT_EQ(coin_change({1, 2, 5}, 11), 3);
    ASSERT_EQ(coin_change({2}, 3), -1);
    ASSERT_EQ(coin_change({1}, 0), 0);
}

int coin_change_dp(const std::vector<int> &coins, std::vector<int> &memo, int amount) {
    if (amount == 0) {
        return 0;
    }

    if (amount < 0) {
        return -1;
    }

    if (memo[amount] != -999) {
        return memo[amount];
    }

    int res = std::numeric_limits<int>::max();
    for (int coin: coins) {
        int subp = coin_change_dp(coins, memo, amount - coin);
        if (subp == -1) {
            continue;
        }
        res = std::min(res, subp + 1);
    }
    memo[amount] = res == std::numeric_limits<int>::max() ? -1 : res;
    return memo[amount];
}

TEST(DP, CoinChange2) {
    auto coin_change = [](const std::vector<int> &coins, int amount) {
        std::vector<int> memo(amount + 1, -999);
        return coin_change_dp(coins, memo, amount);
    };

    ASSERT_EQ(coin_change({1, 2, 5}, 11), 3);
    ASSERT_EQ(coin_change({2}, 3), -1);
    ASSERT_EQ(coin_change({1}, 0), 0);
}
