from typing import List


class Solution(object):
    # 1. Two Sum     hash/dict
    @staticmethod
    def twoSum(nums: List[int], target: int) -> List[int]:
        dic = {}
        for i in range(len(nums)):
            if nums[i] not in dic:
                dic[target - nums[i]] = i
            else:
                return [dic[nums[i]], i]
        return []

    # 11. Container With Most Water   two pointer
    @staticmethod
    def maxArea(height: List[int]) -> int:
        left, right = 0, len(height) - 1
        area = 0
        while left < right:
            area = max(area, min(height[left], height[right]) * (right - left))
            if height[left] <= height[right]:
                left += 1
            else:
                right -= 1
        return area

    # 42. Trapping Rain Water  two pointer
    @staticmethod
    def trap(height: List[int]) -> int:
        left, right = 0, len(height) - 1
        l_max, r_max, area = 0, 0, 0
        while left < right:
            l_max, r_max = max(l_max, height[left]), max(r_max, height[right])
            if l_max > r_max:
                area += r_max - height[right]
                right -= 1
            else:
                area += l_max - height[left]
                left += 1
        return area


if __name__ == "__main__":
    print(Solution.twoSum([2, 7, 11, 15], 9))
    print(Solution.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))
    print(Solution.trap([4, 2, 0, 3, 2, 5]))
