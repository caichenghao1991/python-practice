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

greedy algorithm
    each step choose local most optimized option, generate final global possible most efficient solution under certain
    situations. choose a greedy way to ensure the local optimization can also ensure global optimization. example
    questions: intervals (sort on end/ start depends on question), assignment problems
    if there are 2 dimension, sort the first one and considering next dimension based on the fixed first dimension,
        don't consider both dimension at the same time
        merge stop point for previous item, consider traverse from back to front if traverse from front to back will
            cause chain updates(need to update more than one previous item)

sliding window
    two pointer (start from beginning or end, can have different pace)
    from beginning, end meet at middle, code easier than start from middle and one go to beginning and one go to end
    giving target and sorted array, use one forward and one backward pointer
    fast slow pointers: fast move two step, slow move one step. meet at point means circle,after first meet, move
        fast pointer to start and both move one step a time, second time meet at start of circle (Floyd Algorithm)
    double pointer can consider both pointer in outer for loop and pick the easier one
    two pointer with one has delay of several steps

    def minSubArrayLen(self, s: int, nums: List[int]) -> int:
        res, sum, index = float("inf"), 0, 0
        for i in range(len(nums)):
            sum += nums[i]
            while sum >= s:
                res = min(res, i-index+1)
                sum -= nums[index]
                index += 1
        return 0 if res==float("inf") else res
    3sum (O(n^2)), 4sum(O(n^3)) sort then use 2 different side pointers
    2 pointer can also used to get min value of sorted list, instead of use global min variable

recursion:
    define base case (corresponding to lowest index(index able to represent each stage)), and recurrence relationship.
    either top down (process current item first then child node, function need pass result for current node). or bottom
    up approach (get result from child then use that to get result for that current item) use memorization to avoid
    duplication calculation
    return None if no need result for bottom up case, pass in parameter need to process during recursion, don't define
        result inside recursion function as local variable usually top down approach use parameter, and bottom up
        approach return None (if modify global/nonlocal (outer function) variable) or some value

    recursion has top down approach: visit current node first, if current node answer is known, then pass down deduced
        child node answer for calculation when calling recursively on its child nodes, similar to preorder traversal

        def maxDepth(self, root):   # top down
            self.maximum = 0   # define instance variable and inner function to avoid global variable in leetcode, or
                               # write __init__ method and initialize self.var
            def f(node, depth):    # no need self
                if not node.left and not node.right:
                    self.maximum = max(self.maximum, depth + 1)
                if node.left: f(node.left, depth+1)
                if node.right: f(node.right, depth+1)
            if not root: return 0
            f(root, 0)    # need define inner function ahead
            return self.maximum

        bottom up approach: get the answer for child node first, then process current node base on child nodes' result,
            similar to post order traversal  (dfs post order)
        def bottom_down(node):
            if not root:
                return 0
            left_val = top_down(node.left)
            right_val = top_down(node.right)
            return func(left_val, right_val)

    memorization: save calculated result in dictionary and avoid duplicated calculation to save time
        def climbStairs(self, n):
            mem = {}
            def helper(n):
                if n<=2: return n
                if mem.get(n,-1)!=-1:
                    return mem[n]
                res = helper(n-1)+helper(n-2)
                mem[n]=res
                return res
            return helper(n)

    time complexity: draw execution tree,  number of nodes * process time per node
        Master Theorem limitation: only applies to sub problems with equal size.
        T(n) = aT(n/b) + f(n), let k = log_b (a)
        1. f(n) = O(n^p)    p < k      T(n) = O(n^k)
        2. f(n) = O(n^p)    p > k      T(n) = O(f(n))
        3. if exist c >= 0 such that f(n) = O(n^k log_c (n))   T(n) = O(n^k log_(c+1) (n))

    space complexity: need space for returning address (stack to track function call), parameters for function call, and
        local variables inside function. the space is freed after function call is done, function (memory) chain up
        successively until reach base case.

    tail recursion: wrap extra component inside function parameter, no memory cost for system stack, release after call.
        save space, avoid stack overflow
        def sum_tail_recursion(ls):
            def helper(ls, acc):
                if len(ls) == 0:
                    return acc
                return helper(ls[1:], ls[0] + acc)   # instead of non tail recursion # return ls[0] + helper2(ls[1:])
        return helper(ls, 0)

        python don't support tail recursion, use lambda
            def Y(F):
                Y_comb = lambda F: (lambda x: F(lambda *args: lambda: x(x)(*args)))
                                   (lambda x: (F(lambda *args: lambda: x(x)(*args))))
                def wrapper(*args):
                    res = Y_comb(F)(*args)
                    while callable(res):
                        res = res()
                    return res
                return wrapper
            F = lambda f: lambda n,acc: acc if not n else f(n-1,acc+n)
            recSumY = Y(F)
            recSumY(1000,0)   # recursion max depth 1000

            to overcome max depth 1000, use another decorator which use while loop to raise error and get new loop
                parameter via error message, so always depth 1

    use stack/queue to convert recursion to iteration


binary search
    avoid not including correct answer, and avoid infinite loop, compare with mid index value and eliminate half of data
    if monotone increasing, harder question need run binary search twice (multiple time) with different conditions
    only apply on sorted array, arr[mid] can compare with neighbor arr[mid-1], arr[mid+1], need consider index range
    and make sure cover len(arr)==1 if comparing neighbor

    1.  first_occurrence
        while left <= right:
            mid = left + (right - left) // 2    # update mid inside loop
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
            and right part according to comparison target with pivot value   time: O(n)

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
    used in graph and tree search, for graph need check for cycle in dfs and bfs valid check via create visited list/set
    DFS: first explore the current node, can implemented by stack (less chance for stack overflow) or recursive
        usually main function used for search every possible start location, sub function is recursive to dfs search
        can add state for memorization to mark some node and avoid repetition
        can use DFS for circle detection, track of parent node for each traversed node, if one visited node has
        different parent node, then it has circle
        time complexity: O(E+V) if using adjacency list, O(n^2) if using adjacency matrix, O(N) for tree
        space complexity: max tree/graph depth  O(V)

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

        # recursive method, use call stack
        valid check can add to match condition then dfs, or dfs anyway then check valid or not
        def dfs(curr, target, visited)
            if curr is target: return True
            for node in curr.neighbor:
                if not visited[node]:
                    return dfs(node)
            return False

        dfs(node, state[], graph):
            if state(node):    # end return condition   or   if state(node) or not_valid(node):  # post check valid
                return 0       # if not root: return None    if left>right: return None
            state(node) = xxx    # update state of visited node
            stat = 1     # not necessary has statistical data
            for child in node.children:
                if valid(child):   # valid check ahead
                    stat += dfs(child,state_new,graph)
            return stat   # not always need return
                # return stat + sum([dfs(i) for i in node.children])   for post check valid




    Backtracking
        use DFS with saving state to solve permutation and combination problem (restore state allow traverse through
        previous passed route), splitting string(array), n queen/ sudoku problem, time complexity is visiting every
        possible answer (O(2^n)(subset,combination) O(n*2^n if using list), O(n!)(permutation), O(k^n), k is constant)
        when unsatisfying during dfs, back track to previous node and change to previous state as soon as seeing not
        satisfy the potential solution, only need update the combined overall state, instead of creating sub state for
        each condition, use reference to pass the state, change back the state(flag or output) after recursion
        iterative method put extra variable together with node into a tuple and add to stack/queue
        must reverse change if using global/ nonlocal variable and modify value during expanding child node stage. in
        comparison, if only modify passed in function parameter no need to reverse change. function usually don't have
        return value
        prune the for loop if start index plus k nodes required larger than end index
        not suitable for iterative since too complicated
        return bool if only need find one path

        def main(state):
            ans = []  # can use global/nonlocal variable replace function parameters
                # visited = []   optional create visited matrix
            for node in space:  # space can only have one node, also node doesn't have to be tree explicitly
                backtrack(state, visited, node_index, init_val, ans)
            return ans
        def backtrack(state, xx, ans):   # xx can be multiple variables pass into function
            if match(xx):                # back track avoid create local variable ans
                                        # if len(path)==len(res)    if startIndex == len(res)
                ans.update()
                    # ans.append(state[:])  # must use copy() for mutable variable otherwise update_back(state) will
                                                # cause issue. no need copy() if update state in function param
                return

            for child in children:  # combination from different arr horizontal traverse, ran O(V) same as vertices
            # for i in range(startIndex, (len(arr)-(k-len(path)))+2)  # for combination in one arr, can sort arr first
                                # to avoid duplicated value, or add dict/set/list (list preferred with update and
                                # un update to save space) as function param to track used item better prune when
                                #  comparing same level previous node, instead of check in path
                if valid(state, xx):
                    if not valid(state):
                        continue
                    update(state)
                    backtrack(state, xx_new, ans)     # vertical traverse
                        # difference with dfs:  dfs(new_state, xx_new, ans)
                        # if backtrack(state, xx_new, ans): return True   for only search one result
                    update_back(state)


        def combine(self, n, k):
            result = []
            def gen_comb(start, cur_comb):
                if k == len(cur_comb):
                    # base case, also known as stop condition
                    result.append( cur_comb[::] )
                    return
                else:    # back track                                      # dfs
                    for i in range(start, n+1):                             for i in range(start, n+1):
                        cur_comb.append( i )                                    gen_comb(i+1, cur_comb+[i])
                        gen_comb(i+1, cur_comb)
                        cur_comb.pop()
                    return
            gen_comb( start=1, cur_comb=[] )
            return result

    BFS: first traverse all the current node children then proceed to next grand children layer, use fifo queue. used
        for getting the shortest route. can BFS from both start and end to reduce search time (1+2+4+8+16 vs 2*(1+2+4))
        (each time switch queue for smaller size queue and BFS on that queue) can use together with backtracking
        time complexity: O(E+V), if using adjacency list O(n^2),  O(N) for tree
        space complexity: max queue size (max items in a layer)  O(V)

        queue = deque()
        queue.append(init_node)
        res = init_val
        while len(queue) > 0:   # tree search space is root, and graph space is each nodes
            curr = queue.popleft()  # current node
            update(res)     # res update once per node
            for child in curr.children:   # child nodes for tree and neighbor for graph
                if valid(child) and not visited[child]:  # check match valid condition, must check if graph has loop
                                                         # via maintain a visited list or set
                    stack.append(child)   # add child in stack
                    visited[child] = True
                    if child_match(end_condition):
                        return res
        while len(queue) > 0:   # tree search space is root, and graph space is each nodes
            size = len(queue)
            for i in range(size):
                curr = stack.popleft()  # current node
                if match(end_condition): return step
                for child in curr.children:
                    if valid(child):  # check match valid condition
                        stack.append(child)   # add child in stack
            step += 1    # res update once per level
        return -1


Dynamic Programming
    solve problem with repeating sub problem, store sub problem solution in memory. only able to solve local
    optimization lead to global optimization problem. find state transfer function from prev state to next state.
    Also able to use space optimization if only use constant previous step state, no need store all states result.
    It's a bottom up algorithm, solve sub problem then main problem. If need solution path, need state record.
    dp[i] only consider the result consist state i, if need get all solution, might need sum all dp[i] i=0,...n
    for 4 direction (up, down, left, right) grid search can run two times one from top left search right down, and other
    from bottom right search top left.
    dp can also store intermediate sub solution, and accumulate sum/ length of dp become final answer
    try keep the state transfer function simple by derive the new state by <=2 previous state
    including subcategory problem: splitting value, common sequence, knapsacks, string manipulation, stock trading
    1. define index i and dp[i] meaning
    2. get transfer function
    3. initialize dp value  start with index that actually match the index meaning, not always need dp[0],dp[1]
    4. get traverse direction
    5. validate by example
    must check fpr loop dp list index is valid, might need initiate a row/column values instead of 1/2 value in 2d
        list

    dp=[0 for i in range(xx)]
    if n <= x:      # base case
        return xxx
    dp[0],...,dp[x] = x,...,x  # update base case
        # s0,s1,...,sn = x,...x   # memorization compression
    for i in range(x, xx):
        dp[i] = some function(dp[i-1],dp[i-2],...)
            # cur = some function(s0,s1,...,sn)    # memorization compression
            # s0,s1,...,sn-1,sn =s1, ...sn,cur   # shift one step to newer state
    return dp[n]

    #2D bagging problem
    for i in range(x, xx):
        for j in range(x, xx):
        dp[i][j] = some function(dp[i-1][j],dp[i][j-1],..., state[i][j])
            # dp[j] = some function(dp[j-1],...state[i][j])  # space compression
            # consider only need one line of stored data if only require dp[i-1][j],dp[i][j-1], dp[i-1][j] transfer to
            # d[j], and dp[i][j-1] transfer to d[j-1]
            # need consider second loop whether forward traverse(depend on current level previous state) or backward
            # traverse(depend on previous level previous state)
                # knapsack one item one time (first i item, total weight j):
                    for i in range(N): for j in range(W):
                        if j>= w[i]: dp[i][j]=max(dp[i-1][j], dp[i-1][j-w[i]]+v[i])
                        else: dp[i][j]=dp[i-1][j]
                    for i in range(N): for j in range(W, w[i],-1): dp[j]=max(dp[j],dp[j-w[i]]+v[i])
                # knapsack one item multiple time (first i item, total weight j):
                    for i in range(N): for j in range(W):
                        if j>= w[i]: dp[i][j]=max(dp[i-1][j], dp[i][j-w[i]]+v[i])
                        else: dp[i][j]=dp[i-1][j]
                    for i in range(N): for j in range(w[i],W,1): dp[j]=max(dp[j],dp[j-w[i]]+v[i])
                # if value multi-dimension, add addition inner for loop

    return dp[i][j]   # dp[j]

    also can initialize values inside dp update loop with if condition
    splitting problem loop through max length, then loop through valid item list, dp[i] = func(dp[i-valid item length])
    subsequence problem dp[i] means stat for subsequence ends on i, need count all dp in the end. update
        dp[i]=max(dp[i],dp[j]+1)
    string comparison problem usually 2 for loop for each char in both string correspondingly
    stock problem can have 2 dp matrix track max profit for k times buy and sell
     loop through days and max buy count   buy[j] = max(buy[j], sell[j-1] - prices[i])
            sell[j] = max(sell[j], buy[j] + prices[i]);
        complex stock problem with cooldown can draw state machine, for each state create matrix with size of time steps
        track money in out between state and write transfer functions for each state


Divide and Conquer (recursion)
    time complexity: or draw the recursion tree
    T(n) = aT(n/b) + f(n), let k = log_b (a)
    1. f(n) = O(n^p)    p < k      T(n) = O(n^k)
    2. f(n) = O(n^p)    p > k      T(n) = O(f(n))
    3. if exist c >= 0 such that f(n) = O(n^k log_c (n))   T(n) = O(n^k log_(c+1) (n))
    # use memoization+ divide conquer or dynamic programming

    space complexity: proportion to max depth

    memo=[[0]*n for _ in range(n)]
    def rec(memo,left, right):
        if left+1 == right: return xx    # base case
        if memo[left][right]: return memo[pos]    # already memorized
        for i in range(left, right):
            l = rec(memo,left, i-1)
            r = rec(memo,i+1, right)
            res = func(l,r)
        memo[left][right] = res
        return res

    dp = [[0]*n for _ in xrange(n)]
    for k in xrange(2, n):
        for left in xrange(0, n - k):
            right = left + k
            for i in xrange(left + 1,right):
                dp[left][right] = func(dp, i)
    return dp[0][n - 1]


Math Problems
    greatest common divisor: keep use bigger value divide small value, and replace big value with remainder, till
        remainder become 0, last denominator(除数) is the gcd
    least common multiple: a*b/gcd
    count prime number less than n: initialize list of true with size n, for loop 2 to n, if current prime is true,
         set multiple of current value to false, finally count how many true/false remained
    convert from 10 based to k based value: each time divide current value with k, add remainder to the right of answer
        and replace current value with quotient(商)
    Fisher-Yates shuffle array of number: from array index 0~n-1 random pick one swap with last item, and consider last
        item is in complete state. then consider 0~n-2 index random pick one swap with second last item. repeat till the
        first item
    weight sampling: add all items in weights array [w1,w2,...wn] together as total,  random pick a value x from
        [0,total], and find corresponding index  x- w1 -w2-... till <0
    random sample k values from array length n with same probability: initialize size n array with all possible values
        if necessary(is its linked list), loop from index 0 to n-1, each time at index i, generate a random number j
        from [1, i], if j<=k, swap array[j] = array[i].  first k values are the final answer
        consider at index i: has k/i chance to add in sample, i+1 index has k/(i+1) add to sample and i has 1/k chance
        to be removed, so overall for each value has chance P = (k/i)*(1-(k/(i+1))*(1/k))*.... = k/i * (i/i+1) *
        ((i+1)/(i+2))*...*((n-1)/n) = k/n

Bitwise Operation
    ^ XOR     ~ NOT     & and    | or    <<    >>
    x ^ 0s = x    x ^ 1s = ~x   x ^ x = 0      x ^ 0s = x    x ^ 1s = ~x   x ^ x = 0
    x | 0s = x    x | 1s = 1   x | x = x
    n = 11110100       n & (n-1)  11110000  remove lowest 1
    n = 11110100       n & (-n)  00000100  get lowest 1    -n: ~n+1
    add extra 0 at left side on shorter length of operand  1010 & 1 = 1011
    print(int("100111",2))   bin(10)

array
    python use list instead, continuous memory location, need shift items if insert/delete, index start 0
    array traverse from beginning or end, two array can start both beginning or end or one each.
    use hash to save time complexity
    sort array might help   .sort() is in place    sorted() is not
    be careful with index of array +1, -1   index in range(len(arr))  [0,len(arr))
    rotate array by k step use reverse array

String
    convert string to list for operations to save memory for recreate string for every modification, finally convert
    back to string. consider doing operation from end of string to front of string (avoid shifting after add an element)
    consider reverse the whole string, then reverse again for the word.  also reverse whole string and then reverse
    part of string

        search match string index : KMP  O(m+n)
        prepare longest proper prefix which is also suffix array for each index position of the pattern, value is the
        length of longest suffix end at that position that match the prefix (if miss match happen move to index with
        this value will avoid checking the same prefix value, since same as suffix)
        a,b,a,b,d  list(0,0,1,2,0)  if mismatch at d, go to the index of its corresponding value for the position
        left beside of d which is b, has value 2, so go to list[2], check match second a or not, if not match continue
        repeat till match or front of pattern, if at front of pattern, move to next index of string. if matching move
        both pointer for string and pattern to the right one step

        or create lps array with addition -1 at beginning indicate start

    def strStr(haystack, needle):    # needle is pattern
        n, h = len(needle), len(haystack)
        i, j, lps = 1, 0, [-1]+[0]*n                # i: pointer for pattern, j: pointer for string, lps array
        while i < n:                                # calculate next array
            if j == -1 or needle[i] == needle[j]:
                i += 1
                j += 1
                lps[i] = j
            else:
                j = lps[j]                # [-1, 0, 0, 1, 2, 0]  lps for keep track of index should go back to after a
        i = j = 0                         # match failed
        while i < h and j < n:
            if j == -1 or haystack[i] == needle[j]:
                i += 1
                j += 1
            else:
                j = lps[j]
        return i-j if j == n else -1     # if last d (j=4) not match, j=lps[4] (previous character match length) =2
                                         # check second a next
    print(strStr('ababcababababd','ababd'))   # 9 index of match start


linked list
    compare to array, access slower O(n), but insert delete O(1)
    class Node():
        def __init__(self, value=None, next=None):
            self.value = value
            self.next = next
    class LinkedList():
        def __init__(self, value):
            self.head = Node(0)  # head is the first element of the linked list
            self.length = 0
            self.tail = self.head  # tail is the end of linked list, optional
    take care of order of updating node and its neighbors
    create dummy node when need delete or modify node (if not create dummy node, logic would be different when delete
    head node and rest of the nodes) and when using 2 pointer(header don't have pre node)
    dummy node pointing to the head of linked list and return dummy.next to retrieve linked list each step only consider
    the current node relation (current node's next pointer)

    dummy = Node(0) dummy.next=head  curr = dummy      while curr    or dummy (while curr.next)
    curr = head   while curr  return head (change original linked list)

    def reverseList(self, head: ListNode) -> ListNode:
        def reverse(pre,cur):
            if not cur:
                return pre
            tmp = cur.next
            cur.next = pre
            return reverse(cur,tmp)
        return reverse(None,head)

    def reverse(head):                 # reverse linked list non recursive
        pre, curr = None, head          # can't use dummy node here
        while curr:
            nex = curr.next
            curr.next = pre
            pre = curr
            curr = nex
        return pre

    rotate linked list can link last item next pointer to first item

    fast slower (different speed) pointer, two pointer with one has a delay.  run through both linked list once and get
        length and use that info for problem equivalent transfer
    if only 1 step between pre, and current cursor, no need 2 cursors, can use while cur.next: cur.next=cur.next.next
    pay attention for each step whether assign new current node, or assign current node's next pointer

    for double linked list first link addition node with pre and post node, then assign pre_node.next, post_node.pre
        double linked list might (not must) has a tail node available beside head

Queue & Stack
    stack implement via list/linked list, queue implemented by deque(based on double linked list)
    from collections import deque
    q = deque()   # queue    or implement via linked list
    q.append(1)
    print(q[0], q.popleft() **  , len(q))

    s = []   # stack
    s.append(1)
    print(s.pop())
    print(s[len(s) - 1])   # peek

    implement queue with 2 stacks(one hold input, one output), for dequeue pop out stack if has item. if empty, pop all
        input and push to output stack. For enqueue, just push to input stack. for peek just pop output stack, then push
        it back. O(1) for push pop peek.
    implement stack with queue: for push, just enqueue. for pop get the size n of queue, repeat n-1 times dequeue first
        item and enqueue back, then last time dequeue is the item popped.
    consider using stack if need constantly modify the last or last several items, while consider the last added item
        first. add items in stack until some signal to pop items, then repeat till end of loop. (ex. reverse polish
        notation)

    monotone increasing/decreasing queue/sliding window queue (via deque/list)
        maintain all possible largest item in queue(not all items inside window), while item in queue is from large to
        small. during push item, continue pop last item until smaller than peek item, or till empty. during pop, only
        remove first item if it's the item exiting the window


    priority queue (implemented via heap(complete binary tree, parent node larger than child node if max heap))
        heap time complexity: build heap: O(log1) + O(log2) + O(log3) + … O(logn) = O(n)
            heapify: O(log n)
        keep k largest item use minheap, pop item if size > k
        usually use array for complete binary tree   child index i//2 => parent index   parent index => 2i+1, 2i+2
        O(1) get largest/smallest item, O(log n) insert or remove top item
        li = [2,3,1,5,4]   # li=[]
        heapq.heapify(li)    # min heap, heapify make li[0] smallest item, inside call siftup  O(log n)
        heapq.heappush(li, 6)  # O(logn)    append new item to last then sift up (continue swap with parent if smaller
                               # than parent)
        heapq.heappush(li, (6,'index'))    # heap sort based on first item of tuple (6)
        v=heapq.heappop(li)    # O(logn)   move last item to first then sift down (continue swap with larger child node)
        print(li[0])  # 2    # get smallest O(1)
        heapq.merge(li, li2)   # merge 2 list  O(n)
        heap.nlargest(3,li)  # [5,4,3]   nsmallest

        # max heap
        heapq._heapify_max(li)                  # heapify
        li.append(6)                            # heappush
        heapq._siftdown_max(li, 0, len(li)-1)   # heappush
        v=heapq._heappop_max(ll)                # heap pop

        # heap sort
        build heap, remove first item, get first item(largest), put last item to first place, and do heapify. repeat
            until empty tree



HashSet, HashMap:
    set and dict
    if more value after hash function than bucket number, use remainder   (hash_val % bucket_count)
    can use list to replace dict/set if the bucket size is fixed, list faster than hash(dict, set) since no need hash
        function, but too many buckets cost lots memory. usually use list for 26 character
    remove item once done, can save time complexity by eliminate duplicate search on that item
    for geometry coordinates problem can consider slope as dict key
    if constant number of items in each bucket, use linked list . if variable size or large, use height-balanced binary
        search tree. Or when bucket size> item size, can relocate items in same bucket to new bucket
    design key: sort string,offset with first value, tree node(or serialization of tree node(string with child info),
        row/ column index
    can use hashset to check whether contain cycle
    collections.OrderedDict()    keep insertion order. use double linked list and dict, all operations O(1), query use
        dict
    collections.defaultDict()   # has default value, even for key not exist, also all value must same type
        d = defaultdict(int)

range sum query 1d and 2d
    1d store previous sum for each index in p[] and use p[j+1]-p[i] to get the sum of arr[i,j]
    2d use dp, for each index i,j store the area sum for the rectangle with diagonal coordinate of 0,0 and i,j
        dp[i][j] = nums[i-1][j-1] + dp[i-1][j] + dp[i][j-1] - dp[i-1][j-1]
        when query 2 coordinate (row1, col1), (row2,col2) use
        sum = dp[row2+1][col2+1] - dp[row2+1][col1] - dp[row1][col2+1] + dp[row1][col1];


binary tree
    binary tree parent node has left and right child node. pre, in, post order traversal, recursive (iterative) way
    most problem are solving recursively with helper function (base empty case, and return parent node state with
    recursive call of child node state as known), only need consider one parent and corresponding one child layer
    usually
    full binary tree, last layer full (2^k - 1 nodes (k: depth of tree)).
    complete binary tree , last layer right side can be empty, last node's parent can only have left child node as last
        node
    balanced binary tree, left child height and right child height difference less than 1

    post order traversal is the delete tree node order, also easier for math operation with values and operand as node
        push values in stack, when met operand, pop two values and push the result back in stack

    list presentation: parent index i, left/right child index 2i+1 and 2i+2
    traversal: dfs(pre(root first), in(root middle), post order(root last)), bfs(level order)
        pre-order: init root in stack, pop node n, add to result, push n.left, n.right in stack
        post-order: same as pre order, but push n.right, then n.left. finally reverse result
        in-order: loop when stack not empty or curr node not none: if curr, add to stack, update curr = curr.left, else
            curr = stack.pop(), add it to result, and update curr = curr.right

        or add a None after add root node, use universal for switch order to reverse correct order inside loop
        for pre order:  right, left, mid    in-order: right, mid, left    post-order:mid, right, left
        post-order
        def postorderTraversal(self, root: TreeNode) -> List[int]:
            result, st = [],[], depth=0  (depth some other stats need to trace)
            if root: st.append(root)
            while st:
                node = st.pop()
                if node != None:
                    st.append(node) # mid
                    st.append(None)
                    depth += 1
                    if node.right: st.append(node.right)
                    if node.left: st.append(node.left)
                else:
                    node = st.pop()
                    depth -= 1   # here need reverse change 1 step
                    result.append(node.val)
            return result

        bfs(level order): add root to deque, get size of deque, iterate size time: node = deque.popleft(), do some
            operation, if node.left add node.left to deque, same for right

    class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    binary search tree: left child less than parent, right child larger than parent. in order traversal return sorted
        list. search, insert, delete time complexity O(h)  (height of tree)
        search node with value v: compare current node, if equal return current node, if v less than current node value,
            search in left child node, if greater than current node value, search right child node
        insert node v: similar to search, only insert node when the position is empty, and added node become leaf node
        delete node v: 1. If the target node has no child, we can simply remove the node.
            2. If the target node has one child, we can use its child to replace itself.
            3. If the target node has two children, replace the node with its in-order successor or predecessor node
        binary search tree search node iterative method no need stack/queue, since only need explore one node (either
            left or right base on comparison value with root value) a time

        also can consider traverse right-> mid -> left if need get sum of nodes on given node's right side

        balanced binary search tree: left sub tree and right subtree, height difference <= 1

    complete binary tree: can consider left and right subtree, can separate the tree to full binary trees (has 2^h-1)
        nodes. when check full binary tree, compare left.left... and right.right....  same height or not

    iterative method traverse one path to node no need stack/queue 

    trie
        root is empty string, each node present a character. all the descendants of a node have a common prefix of the
        string associated with that node. That's why Trie is also called prefix tree. word can end at middle of tree,
        not always at leaf node, so need boolean flag to indicate end of words
    class Node:
    def __init__(self, char: str):
        self.char = char
        self.children = []   # if need using index O(1) access time, need create constant list size children,
        self.is_finished = False
        self.counter = 1
#   or use dict with key of character and value of trie node (a bit slower O(1), more flexible, save space)
        self.children = {}

    # insert word
    curr = Node(' ')
    for c in list(str):
        if c not in curr.children.keys():
            curr.children[c] = Node(c)
        curr = curr.children[c]

    # search word
    for c in list(str):
        if c not in curr.children.keys():
           return False
        curr = curr.children[c]
    else:
        return curr.is_finished


Graph
    directed/undirected, cyclic/acyclic, connected/disconnected. DAG (directed acyclic graph)
    adjacency list: each index(node) its keep neighbors nodes (array or linked list). for each node has one index
    adjacency matrix: 2D matrix of nodes vs nodes  G[i][j]  store edge ij 1 or value for weighted graph, 0 if no edge,
        consume more space but O(1) for checking exist edge ij or not
    store list of edges (i,j)

    topological sort  O(V + E)
        put all 0 in_degree(no in edge) vertex in queue, pop a 0 degree vertex and append to result order  list, update
            all out edge vertex in_degree - 1. cyclic if result list size not equal to vertex count
        def topologicalSort(self):    # queue way
            in_degree = [0]*(self.V)
            for i in self.graph:
                for j in self.graph[i]:
                    in_degree[j] += 1

            queue = []
            for i in range(self.V):
                if in_degree[i] == 0:
                    queue.append(i)
            cnt = 0   # Initialize count of visited vertices
            top_order = []
            while queue:
                u = queue.pop(0)
                top_order.append(u)
                for i in self.graph[u]:
                    in_degree[i] -= 1
                    if in_degree[i] == 0:
                        queue.append(i)
                cnt += 1
            if cnt != self.V:
                return -1 b  # cyclic
            else :
                return top_order

        # stack + recursion way
        def topologicalSortUtil(self, v, visited, stack):
            visited.append(v)
            for i in self.graph[v]:
                if i not in visited:
                    self.topologicalSortUtil(i, visited, stack)
            stack.insert(0, v)
        def topologicalSort(self):
            visited = []
            stack = []
            for k in list(self.graph):
                if k not in visited:
                    self.topologicalSortUtil(k, visited, stack)
            return stack

    # Dijkstra  t: O(n + E)logn   s: O(n+E)

    Dijkstra: add source to a heap, each time pop one node i from heap(smallest cost) and move into seen list, update
        i's neighbor j which is not in seen list, update j cost as cost(i) + cost(i to j). until heap empty

    def networkDelayTime(times: List[List[int]], n: int, k: int) -> int:
        # Dijkstra
        adj = [{} for _ in range(n)]
        for i, j, t in times:
            adj[i - 1][j - 1] = t
        distance = [float("Inf")] * n
        distance[k - 1] = 0
        seen = set()
        q = [(0, k - 1)]
        while q:
            d, n = heapq.heappop(q)
            if n not in seen:
                seen.add(n)
            for i in adj[n]:
                if i not in seen and d + adj[n][i] < distance[i]:
                    distance[i] = d + adj[n][i]
                    heapq.heappush(q, (distance[i], i))


        # Bellman Ford     t: O(nE)   s: O(n)
        initialize distance list, 0 for src, rest infinite. run n-1 time, n is number of nodes, for each edge, update
            end node dist value if less than source distance + edge weight
        run one more time if still find a smaller update of dist list, there is negative cycle

        distance = [float("Inf")] * n
        distance[src] = 0
        for k in range(n):
            for i, j, t in times:   # i start node, j end node,  t cost
                if distance[j] > distance[i] + t:
                    distance[j ] = distance[i] + t
        """
        return -1 if max(distance) == float("Inf") else max(distance)

Union find (Disjoint set)
    first mark all point's parent as themselves, when check_parent of a node i, while i != parent[i]: i=parent[i] return
    i  at end as parent.   when connect an edge ij, mark parent[i] = check_parent of q,


string
    'Hello, %s' %(name,)    'Hello {}'.format(name,)
    s.find(',')  # -1 if not found
    s.startswith('')  # endswith
    s.replace('old','new',2)  # replace old to new string max 2 times
    s.split(',',2)  # split 2 times into 3 pieces, rsplit
    s.capitalize()  # title  upper lower
    s.strip()      # ljust(30) add space padding on left till 30 char total,  rjust  center  lstrip  rstrip
    ''.join(iterable)
    char(20)   ord('A')
    string match




pop clear del
list
    l=[1,2,'a']    l=[i if i>10 else 2*i for i in range(20) if i%2==0]
    l2 = l.copy()
    l=list(range(3,5))   l=list('ABC')
    c = l.count(1)
    print(1 in l)  # True
    l.index(1)  # exception if can't find
    len(l)
    l=l*2  # [1,2,'a',1,2,'a']   or l*=2
    l=l+['b']
    l.append(20)
    l.insert(-1,'b')
    l.extend(['b','c'])
    l[1]=1
    l.pop()  # remove last item
    l.pop(1) # remove item with index, exception if out of index range
    l.remove('a')   # remove first occurrence or raise error if not exist
    l.clear()
    del l[1:2]
    l=l.reverse()    # l=reversed(l)
    l=l.sort()   # l.sort(reverse=True)   l=sorted(l)
    xxx=min(list1)  # max(list1)  sum(list1)

tuple
    t= (1,2,'a')
    t=tuple([1,2,3])
    a,*b = t   # b=[2, 'a']
    len(t)
    1 in t
    t.index(1)
    t.count(1)

dict
    key need to be immutable (not list or dict)
    d={'name': 'Harry'}
    d=dict([('name','Harry')])
    d=eval('{"name":"Harry"}')
    d=json.loads('{"name":"Harry"}')
    d=dict.fromkeys(['name','age'], 1)
    dict1.get('name', "NA")   # return default if not found
    len(d)
    'name' in d
    list(d.keys())   # .values()  .items()
    d['name']='Harry Potter'
    del d['name']   # error if not exist
    name=d.pop('name',None)  # return None if not found, error if not specified
    (k, v) = my_dict.popitem()  # remove last item
    d.clear()
    del d
    d=sorted(d.items(),key=lambda x:x[1],reverse=False)

    collections.OrderedDict()
    same functions as dict but sort items by key

set
    s=set()
    s={1,2,3}
    1 in set
    s.add(4)
    s.update(s2)  # add set s2 to s
    s = s.union(s2)   # .intersection(s2)     .difference(s2)
    s.difference_update(s2)   # modify set1 to unique item only  {1, 2}
    s1.issubset(s2)   #.issuperset(s2)  .isdisjoint(s2)
    x=s.remove(1)      # raise error if not found
    x=s.discard(1)
    s.clear()
    s1.pop()   # random delete one item
    del s

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