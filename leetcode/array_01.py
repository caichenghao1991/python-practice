from typing import List


class Solution(object):
    # 1. Two Sum     hash/dict
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        dic = {}
        for i in range(len(nums)):
            if nums[i] not in dic:
                dic[target - nums[i]] = i
            else:
                return [dic[nums[i]], i]
        return []

    # 11. Container With Most Water   two pointer
    def maxArea(self, height: List[int]) -> int:
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
    def trap(self, height):
        left, right = 0, len(height) - 1
        lmax, rmax, area = 0, 0, 0
        while left < right:
            lmax, rmax = max(lmax, height[left]), max(rmax, height[right])
            if lmax > rmax:
                area += rmax - height[right]
                right -= 1
            else:
                area += lmax - height[left]
                left += 1
        return area


if __name__ == "__main__":
    solution = Solution()
    print(solution.twoSum([2, 7, 11, 15], 9))
    print(solution.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))