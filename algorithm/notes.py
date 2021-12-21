'''
1. come up with a brute-force solution
2. Think of a simpler version of the problem (decompose the problem, or manipulate original solution to easier
    representation with same solution)
3. Think with simpler examples -> try noticing a pattern
4. use some visualization for example
5. test your solution on a few examples
6. computer don't know whole picture, they check item one by one, which is different than our multi combined steps
        thinking in mind. keep this in mind during coding
7. check edge cases, and unique cases
8. reduce the possibilities without eliminating the correct answer


sliding window
        two pointer (start from beginning or end, can have different pace)
        giving target and sorted array, use one forward and one backward pointer
        fast slow pointers: fast move two step, slow move one step. meet at point means circle,after first meet, move
            fast pointer to start and both move one step a time, second time meet at start of circle (Floyd Algorithm)
        double pointer can consider both pointer in outer for loop and pick the easier one

binary search
    avoid not including correct answer, and avoid infinite loop, compare with mid index value and eliminate half of data
    if monotone increasing, harder question need run binary search twice (multiple time) with different conditions

    1.  first_occurrence
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] > value:
                right = mid - 1
            elif arr[mid] < value:
                left = mid + 1
            else:
                # res = mid   for search an unique exist item
                res = mid
                right = mid - 1
        return res
    2. first_greater or equal
        while left <= right:
            mid = left + (right - left) // 2
            if arr[mid] < value:  <= for greater
                left = mid + 1
            else:
                res = mid
                right = mid - 1
        return res

sorting
    quick sort: choose pivot (for example: right edge as pivot, need random shuffle ahead), find correct position of
        pivot by (pointer init at left edge, traverse from left to right swap with pointer item if <= pivot, and
        increase pointer value, finally swap pointer and pivot and return correct pivot position as pointer location),
        then repeat for left and right part

        quick selection find kth largest element: find partition of correct index for an item, then binary search left
            and right part according to comparison target with pivot value

    merge sort: recursive merge sort until length 1, each time split to right and left half then two pointer from front
        and merge them together into sorted array

    insertion sort: from left to right increasing the left sorted size by one, consider the item beside the sorted part,
        from right to left, if number greater than this, shift to the right

    bubble sort: from left to right compare adjacent pairs and switch if in wrong order, after one iteration the largest
        /smallest value is at the right end, then repeat each time one more value at right end is sorted

    selection sort: from left to right find the smallest value among all the value, then switch with the left end value,
        after one iteration the smallest value is at the left end, then repeat each time one more value at left end is
        sorted

Searching
    used in graph and tree search
    DFS: first explore the current node, can implemented by stack (less chance for stack overflow) or recursive
        usually main function used for search every possible start location, sub function is recursive to dfs search
        can add state for memorization to mark some node and avoid repetition
        can use DFS for circle detection, track of parent node for each traversed node, if one visited node has
        different parent node, then it has circle

        # stack method
        for node in space:   # tree search space is root, and graph space is each nodes
            if state(node):   # check for whether visited
                stat = 1   # optional for statistical data
                state(node) = xxx  # update state of visited node
                stack = []    # create stack
                stack.append(node)    # either push in node object or index for graph
                while len(stack) > 0:
                    curr = stack.pop()  # current node
                    for child in curr.children:   # child nodes for tree and neighbor for graph
                        if valid(child):  # check match valid condition
                            state(child) = xxx    # update state of child node
                            stat += 1   # update statistical data
                            stack.append(child)   # add child in stack
        return stat

        # recursive method
        valid check can add to match condition then dfs, or dfs anyway then check valid or not
        for node in space:
            if state(node):
                res = operation(dfs(node))
        return res
        dfs(node, state[], graph):
            if state(node):    # end return condition   or if state(node) or not_valid(node):  # post check valid
                return 0
            state(node) = xxx    # update state of visited node
            stat = 1     # not necessary has statistical data
            for child in node.children:
                if valid(child):   # valid check ahead
                    stat += dfs(child)
            return stat   # not always need return
                # return stat + sum([dfs(i) for i in node.children])   for post check valid

    Backtracking
        use DFS with saving state to solve permutation and combination problem
        when unsatisfying during dfs, back track to previous node and change to previous state, only need update the
        combined overall state, instead of creating sub state for each condition, use reference to pass the state,
        change back the state(flag or output) after recursion

        def main(state):
            ans = []
                # visited = []   optional create visited matrix
            for node in space:  # space can only have one node
                backtrack(state, visited, node_index, init_val, ans)
            return ans
        def backtrack(state, xx, ans):   # xx can be multiple variables pass into function
            if match(xx):
                ans.update()
                return
            if not valid(xx):  # optional
                return
            for child in children:
                update(state)
                backtrack(state, xx_new, ans)
                update_back(state)


    BFS: first traverse all the current node children then proceed to next grand children layer, use fifo queue. used
        for getting the shortest route. can BFS from both start and end to reduce search time (1+2+4+8+16 vs 2*(1+2+4))
        (each time switch queue for smaller size queue and BFS on that queue) can use together with backtracking

        queue = deque()
        queue.append(init_node)
        res = init_val
        while len(queue) > 0:   # tree search space is root, and graph space is each nodes
            curr = stack.pop()  # current node
            update(res)
            for child in curr.children:   # child nodes for tree and neighbor for graph
                if valid(child):  # check match valid condition
                    stack.append(child)   # add child in stack
                    if child_match(end_condition):
                        return res

array
    array traverse from beginning or end, two array can start both beginning or end or one each
    use hash to save time complexity
    sort array might help


linked list

Queue & Stack
    q = deque()   # queue
    q.append(1)
    print(q[0], q.popleft(), len(q))

    s = []   # stack
    s.append(1)
    print(s.pop())
    print(s[len(s) - 1])   # peek

recursion + backtracking

binary tree








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
print(5//2)