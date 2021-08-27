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


class MyQueue(object):

    def __init__(self):
        self.s1 = []
        self.s2 = []

    def push(self, x: int):
        self.s1.append(x)

    def pop(self) -> int:
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())
        return self.s2.pop()

    def peek(self) -> int:
        if self.s2:
            return self.s2[-1]
        return self.s1[0]

    def empty(self):
        return not self.s1 and not self.s2


if __name__ == "__main__":
    print(Solution.twoSum([2, 7, 11, 15], 9))
    print(Solution.maxArea([1, 8, 6, 2, 5, 4, 8, 3, 7]))
    print(Solution.trap([4, 2, 0, 3, 2, 5]))
    q = MyQueue()
    q.push(1)
    q.push(2)
    print(q.pop())
    print(q.peek())
    print(q.empty())
