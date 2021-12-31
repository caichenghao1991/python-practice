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
        use DFS with saving state to solve permutation and combination problem (restore state allow traverse through
        previous passed route)
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

    #2D
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

    string comparison problem usually 2 for loop for each char in both string correspondingly
    stock problem can have 2 dp matrix track max profit for k times buy and sell
        complex stock problem with cooldown can draw state machine, for each state create matrix with size of time steps
        track money in out between state and write transfer functions for each state


Divide and Conquer
    T(n) = aT(n/b) + f(n), let k = log_b (a)
    1. f(n) = O(n^p)    p < k      T(n) = O(n^k)
    2. f(n) = O(n^p)    p > k      T(n) = O(f(n))
    3. if exist c >= 0 such that f(n) = O(n^k log_c (n))   T(n) = O(n^k log_(c+1) (n))
    # use memoization+ divide conquer or dynamic programming

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
    array traverse from beginning or end, two array can start both beginning or end or one each.
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