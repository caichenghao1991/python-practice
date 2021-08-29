from typing import List


class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # nums.sort()
        # return nums[len(nums)-k]

        def partition(num, start, end):
            p = start
            for _ in range(start, end):
                if num[_] < num[end]:
                    num[p], num[_] = num[_], num[p]
                    p += 1
            num[end], num[p] = num[p], num[end]
            return p

        '''
        index = len(nums)-k
        left, right = 0, len(nums) - 1
        while left <= right and left < len(nums) and right >= 0:
            i = partition(nums, left, right)
            if i == index:
                return nums[i]
            elif i < index:
                left = i + 1
            else:
                right = right - 1
        return -1
        '''

        # recursive
        l, r = 0, len(nums) - 1
        i = partition(nums, l, r)
        if i < len(nums) - k:
            return self.findKthLargest(nums[i + 1:], k)
        elif i > len(nums) - k:
            return self.findKthLargest(nums[:i], k - len(nums) + i)
        else:
            return nums[i]

        '''
        # method 2 divide and conquer
        pivot = nums[0]
        left = [l for l in nums if l < pivot]
        equal = [e for e in nums if e == pivot]
        right = [r for r in nums if r > pivot]

        if k <= len(right):
            return self.findKthLargest(right, k)
        elif (k - len(right)) <= len(equal):
            return equal[0]
        else:
            return self.findKthLargest(left, k - len(right) - len(equal))
        '''

    @staticmethod
    def searchRange(nums: List[int], target: int) -> List[int]:
        start, end = -1, -1
        l1, l2, r1, r2 = 0, 0, len(nums) - 1, len(nums) - 1

        while l1 <= r1:
            mid = l1 + (r1 - l1) // 2
            if nums[mid] > target:
                r1 = mid - 1
            elif nums[mid] < target:
                l1 = mid + 1
            else:
                start = mid
                r1 = mid - 1
        while l2 <= r2:
            mid = l2 + (r2 - l2) // 2
            if nums[mid] < target:
                l2 = mid + 1

            elif nums[mid] > target:
                r2 = mid - 1
            else:
                end = mid
                l2 = mid + 1

        return [start, end]


if __name__ == '__main__':
    solution = Solution()
    print(solution.findKthLargest([3, 2, 1, 5, 6, 4], 2))
    print(Solution.searchRange([5, 7, 7, 8, 8, 10], 8))
