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

TEST(DP, bag) {
    //
}

TEST(Array, findMedianSortedArrays) {
    auto findMedianSortedArrays = [](const std::vector<int> &nums1,
                                     const std::vector<int> &nums2) {
        int m = nums1.size();
        int n = nums2.size();
        int p1 = 0;
        int p2 = 0;
        std::vector<int> res;

        while (p1 < m && p2 < n) {
            if (nums1[p1] < nums2[p2]) {
                res.push_back(nums1[p1]);
                p1++;
            } else {
                res.push_back(nums2[p2]);
                p2++;
            }
        }

        if (p1 == m) {
            while (p2 < n) {
                res.push_back(nums2[p2]);
                p2++;
            }
        }

        if (p2 == n) {
            while (p1 < m) {
                res.push_back(nums1[p1]);
                p1++;
            }
        }

        double mid;
        if ((m + n) % 2 == 0) {
            mid = (res[(m + n - 2) / 2] + res[(m + n) / 2]) / 2.0;
        } else {
            mid = res[(m + n - 1) / 2];
        }
        return mid;
    };
    ASSERT_DOUBLE_EQ(findMedianSortedArrays({1, 3}, {2}), 2.0);
    ASSERT_DOUBLE_EQ(findMedianSortedArrays({1, 2}, {3, 4}), 2.5);
}

TEST(Array, maxArea) {
    auto maxArea = [](const std::vector<int> &height) {
        int left = 0;
        int right = height.size() - 1;
        int area = 0;
        while (left < right) {
            int cur_area = std::min(height[left], height[right]) * (right - left);
            area = std::max(area, cur_area);
            if (height[left] < height[right]) {
                left++;
            } else {
                right--;
            }
        }
        return area;
    };

    ASSERT_EQ(maxArea({1, 8, 6, 2, 5, 4, 8, 3, 7}), 49);
    ASSERT_EQ(maxArea({1, 1}), 1);
}

TEST(Array, threeSumClosest) {
    auto twoSumClosest = [](const std::vector<int> &nums, int start, int target) {
        int low = start;
        int high = nums.size() - 1;
        int delta = std::numeric_limits<int>::max();
        while (low < high) {
            int sum = nums[low] + nums[high];
            if (std::abs(target - sum) < std::abs(delta)) {
                delta = target - sum;
            }

            if (sum < target) {
                low++;
            } else {
                high--;
            }
        }
        return target - delta;
    };

    auto threeSumClosest = [&twoSumClosest](std::vector<int> &nums, int target) {
        std::sort(nums.begin(), nums.end());
        int delta = std::numeric_limits<int>::max();
        for (int i = 0; i < nums.size() - 2; i++) {
            int sum = twoSumClosest(nums, i + 1, target - nums[i]) + nums[i];
            if (std::abs(target - sum) < std::abs(delta)) {
                delta = target - sum;
            }
        }
        return target - delta;
    };

    std::vector<int> nums1{-1, 2, 1, -4};
    std::vector<int> nums2{0, 0, 0};

    ASSERT_EQ(threeSumClosest(nums1, 1), 2);
    ASSERT_EQ(threeSumClosest(nums2, 1), 0);
}

TEST(Array, 4sum) {
    auto twoSum = [](std::vector<int> &nums, int start, long target) {
        int low = start;
        int high = nums.size() - 1;
        std::vector<std::vector<int>> res;
        while (low < high) {
            int left = nums[low];
            int right = nums[high];
            int sum = left + right;
            if (sum < target) {
                while (low < high && nums[low] == left) {
                    low++;
                }
            } else if (sum > target) {
                while (low < high && nums[high] == right) {
                    high--;
                }
            } else {
                res.push_back({left, right});
                while (low < high && nums[low] == left) {
                    low++;
                }

                while (low < high && nums[high] == right) {
                    high--;
                }
            }
        }
        return res;
    };

    auto threeSum = [&twoSum](std::vector<int> &nums, int start, long target) {
        int n = nums.size();
        std::vector<std::vector<int>> res;
        for (int i = start; i < n; i++) {
            auto tuples = twoSum(nums, i + 1, target - nums[i]);
            for (auto &tuple: tuples) {
                tuple.push_back(nums[i]);
                res.push_back(tuple);
            }
            while (i < n - 1 && nums[i] == nums[i + 1]) {
                i++;
            }
        }
        return res;
    };

    auto fourSum = [&threeSum](std::vector<int> &nums, int target) {
        std::sort(nums.begin(), nums.end());
        int n = nums.size();
        std::vector<std::vector<int>> res;
        for (int i = 0; i < n; i++) {
            auto tuples = threeSum(nums, i + 1, target - nums[i]);
            for (auto &tuple: tuples) {
                tuple.push_back(nums[i]);
                res.push_back(tuple);
            }
            while (i < n - 1 && nums[i] == nums[i + 1]) {
                i++;
            }
        }
        return res;
    };



}
