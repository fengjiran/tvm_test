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

TEST(Array, removeDuplicates) {
    auto removeDuplicates = [](std::vector<int> &nums) {
        int length = nums.size();
        int slow = 0;
        int fast = 0;
        while (fast < length) {
            if (nums[slow] != nums[fast]) {
                nums[++slow] = nums[fast];
            }
            fast++;
        }
        return slow + 1;
    };

    std::vector<int> nums1{1, 1, 2};
    std::vector<int> nums2{0, 0, 1, 1, 1, 2, 2, 3, 3, 4};
    int k1 = removeDuplicates(nums1);
    int k2 = removeDuplicates(nums2);

    ASSERT_EQ(k1, 2);
    ASSERT_EQ(k2, 5);

    std::vector<int> expected1{1, 2};
    std::vector<int> expected2{0, 1, 2, 3, 4};

    for (int i = 0; i < k1; i++) {
        ASSERT_EQ(nums1[i], expected1[i]);
    }

    for (int i = 0; i < k2; i++) {
        ASSERT_EQ(nums2[i], expected2[i]);
    }
}

TEST(Array, swap) {
    int a = 1;
    int b = 2;
    std::swap(a, b);
    ASSERT_EQ(a, 2);
    ASSERT_EQ(b, 1);
}

TEST(Array, nextPermutation) {
    auto nextPernutation = [](std::vector<int> &nums) {
        int length = nums.size();
        if (length == 1) {
            return;
        }

        int i = length - 2;
        while (i >= 0) {
            if (nums[i] < nums[i + 1]) {
                break;
            }
            i--;
        }
        if (i == -1) {
            std::sort(nums.begin(), nums.end());
            return;
        }

        for (int j = length - 1; j > i; j--) {
            if (nums[j] > nums[i]) {
                std::swap(nums[i], nums[j]);
                break;
            }
        }

        std::sort(nums.begin() + i + 1, nums.end());
    };

    std::vector<int> nums1{1, 2, 3};
    std::vector<int> nums2{3, 2, 1};
    std::vector<int> nums3{1, 1, 5};

    std::vector<int> expected1{1, 3, 2};
    std::vector<int> expected2{1, 2, 3};
    std::vector<int> expected3{1, 5, 1};

    nextPernutation(nums1);
    nextPernutation(nums2);
    nextPernutation(nums3);

    for (int i = 0; i < 3; i++) {
        ASSERT_EQ(nums1[i], expected1[i]);
        ASSERT_EQ(nums2[i], expected2[i]);
        ASSERT_EQ(nums3[i], expected3[i]);
    }
}

TEST(Array, SearchRotatedSortedArray) {
    auto searchRotateSortedArray = [](std::vector<int> &nums, int target) {
        int left = 0;
        int right = nums.size();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            }

            if (nums[mid] >= nums[0]) {
                if (target >= nums[left] && target < nums[mid]) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            } else {
                if (target > nums[mid] && target <= nums[right - 1]) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
        }
        return -1;
    };

    std::vector<int> nums1{4, 5, 6, 7, 0, 1, 2};
    std::vector<int> nums2{4, 5, 6, 7, 0, 1, 2};
    std::vector<int> nums3{1};
    std::vector<int> nums4{1, 3};
    std::vector<int> nums5{3, 1};

    ASSERT_EQ(searchRotateSortedArray(nums1, 0), 4);
    ASSERT_EQ(searchRotateSortedArray(nums2, 3), -1);
    ASSERT_EQ(searchRotateSortedArray(nums3, 0), -1);
    ASSERT_EQ(searchRotateSortedArray(nums4, 1), 0);
    ASSERT_EQ(searchRotateSortedArray(nums5, 2), -1);
}

TEST(Array, SearchRange) {
    auto searchLeftBound = [](std::vector<int> &nums, int target) {
        int left = 0;
        int right = nums.size();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (target == nums[mid]) {
                right = mid;
            } else if (target > nums[mid]) {
                left = mid + 1;
            } else if (target < nums[mid]) {
                right = mid;
            }
        }
        if (left == nums.size()) {
            return -1;
        }
        return nums[left] == target ? left : -1;
    };

    auto searchRightBound = [](std::vector<int> &nums, int target) {
        int left = 0;
        int right = nums.size();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (target == nums[mid]) {
                left = mid + 1;
            } else if (target > nums[mid]) {
                left = mid + 1;
            } else if (target < nums[mid]) {
                right = mid;
            }
        }
        if (left - 1 < 0) {
            return -1;
        }

        return nums[left - 1] == target ? (left - 1) : -1;
    };

    std::vector<int> nums1{5, 7, 7, 8, 8, 10};
    std::vector<int> nums2{5, 7, 7, 8, 8, 10};
    std::vector<int> nums3;

    ASSERT_EQ(searchLeftBound(nums1, 8), 3);
    ASSERT_EQ(searchRightBound(nums1, 8), 4);

    ASSERT_EQ(searchLeftBound(nums2, 6), -1);
    ASSERT_EQ(searchRightBound(nums2, 6), -1);

    ASSERT_EQ(searchLeftBound(nums3, 0), -1);
    ASSERT_EQ(searchRightBound(nums3, 0), -1);
}

TEST(Array, validSudoku) {
    auto isValidSudoku = [](std::vector<std::vector<char>> &board) {
        std::vector<std::vector<bool>> rows(9, std::vector<bool>(9, false));
        std::vector<std::vector<bool>> cols(9, std::vector<bool>(9, false));
        std::vector<std::vector<bool>> subBlocks(9, std::vector<bool>(9, false));

        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                char c = board[i][j];
                if (c == '.') {
                    continue;
                }

                int value = c - '0' - 1;
                if (rows[i][value] || cols[j][value] || subBlocks[i / 3 * 3 + j / 3][value]) {
                    return false;
                }

                rows[i][value] = true;
                cols[j][value] = true;
                subBlocks[i / 3 * 3 + j / 3][value] = true;
            }
        }
        return true;
    };
}

TEST(Array, TrapRainWater) {
    auto trap_rain_water = [](std::vector<int> &height) {
        int left = 0;
        int right = height.size() - 1;
        int lmax = 0;
        int rmax = 0;
        int res = 0;
        while (left < right) {
            lmax = std::max(lmax, height[left]);
            rmax = std::max(rmax, height[right]);
            if (lmax < rmax) {
                res += lmax - height[left];
                left++;
            } else {
                res += rmax - height[right];
                right--;
            }
        }
        return res;
    };

    std::vector<int> height1{0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
    std::vector<int> height2{4, 2, 0, 3, 2, 5};
    ASSERT_EQ(trap_rain_water(height1), 6);
    ASSERT_EQ(trap_rain_water(height2), 9);
}

TEST(Array, JumpGame) {
    auto jump_game = [](std::vector<int>& nums) {
        //
    };
}
