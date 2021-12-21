'''
1. come up with a brute-force solution
2. Think of a simpler version of the problem (decompose the problem)
3. Think with simpler examples -> try noticing a pattern
4. use some visualization for example
5. test your solution on a few examples


array
    array traverse from beginning or end, two array can start both beginning or end or one each
    use hash to save time complexity
    how to reduce the possibilities without eliminating the correct answer
    sliding window, two pointer (start from beginning or end, can have different pace)
    check edge cases, and unique cases
    manipulate original solution to easier representation with same solution
linked list

recursion + backtracking

binary tree

graph traversal (DFS+BFS)

greedy algorithm
    each step choose local most optimized option, generate final global possible most efficient solution under certain
    situations. choose a greedy way to ensure the local optimization can also ensure global optimization. example
    questions: intervals (sort on end/ start depends on question), assignment problems

'''


def rotate(nums, k):
    """
    :type nums: List[int]
    :type k: int
    :rtype: None Do not return anything, modify nums in-place instead.
    """
    size = len(nums)
    if k == size / 2:
        for i in range(k):
            nums[i], nums[i + k] = nums[i + k], nums[i]
    else:
        counter = 0
        p = 0
        v = nums[p]
        while counter <= size:
            follower = (p + k) % len(nums)
            temp = nums[follower]
            nums[follower] = v
            v = temp
            p = follower
            counter += 1
    return nums

print(rotate([1,2,3,4,5,6,7],3))