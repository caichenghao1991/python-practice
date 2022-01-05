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
        from beginning, end meet at middle, code easier than start from middle and one go to beginning and one go to end
        giving target and sorted array, use one forward and one backward pointer
        fast slow pointers: fast move two step, slow move one step. meet at point means circle,after first meet, move
            fast pointer to start and both move one step a time, second time meet at start of circle (Floyd Algorithm)
        double pointer can consider both pointer in outer for loop and pick the easier one
        two pointer with one has delay of several steps

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
        time complexity: O(E+V), if using adjacency list O(n^2),  O(N) for tree
        space complexity: max queue size (max items in a layer)  O(V)

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
    python use list instead
    array traverse from beginning or end, two array can start both beginning or end or one each.
    use hash to save time complexity
    sort array might help   .sort() is in place    sorted() is not
    be careful with index of array +1, -1   index in range(len(arr))  [0,len(arr))


linked list
    compare to array, access slower O(n), but insert delete O(1)
    class Node():
        def __init__(self, value=None):
            self.value = value
            self.next = None
    class LinkedList():
        def __init__(self, value):
            self.head = Node(value)  # head is the first element of the linked list
            self.length = 1
            self.tail = self.head  # tail is the end of linked list
    take care of order of updating node and its neighbors
    sometime create dummy node, pointing to the head of linked list and use dummy.next to retrieve linked list
    each step only consider the current node relation (current node's next pointer)

    dummy = head  curr = dummy       or dummy.next = head   curr = dummy.next while curr    or dummy (while curr.next)
    curr = head   while curr  return head (change original linked list)

    def reverse(head, pre=Node()):    # reverse linked list recursive, need pass default pre node
        if not head:
            return pre
        nex = head.next
        head.next = pre
        return reverse(nex, head)
    def reverse(head):                 # reverse linked list non recursive
        pre = Node()
        while head:
            nex = head.next
            head.next = pre
            pre = head
            head = nex
        return pre

    rotate linked list can link last item next pointer to first item

    fast slower (different speed) pointer, two pointer with one has a delay.  run through both linked list once and get
        length and use that info for problem equivalent transfer
    if only 1 step between pre, and current cursor, no need 2 cursors, can use while cur.next: cur.next=cur.next.next
    pay attention for each step whether assign new current node, or assign current node's next pointer

    for double linked list first link addition node with pre and post node, then assign pre_node.next, post_node.pre
        double linked list might (not must) has a tail node available beside head

Queue & Stack
    stack implement via list
    from collections import deque
    q = deque()   # queue    or implement via linked list
    q.append(1)
    print(q[0], q.popleft(), len(q))

    s = []   # stack
    s.append(1)
    print(s.pop())
    print(s[len(s) - 1])   # peek

    monotone increasing/decreasing stack (via list)
        for last item largest decreasing stack from top to bottom: for each new item, if larger than stack top item,
        continue pop top item, till top smaller than new item or empty stack (do calculation for popped item if
        required), push new item in stack. so the item in stack always sorted

    priority queue (implemented via heap(complete binary tree, parent node larger than child node if max heap))
        usually use array for complete binary tree   child index i//2 => parent index   parent index => 2i+1, 2i+2
        O(1) get largest/smallest item, O(log n) insert or remove top item
        li = [2,3,1,5,4]
        heapq.heapify(li)    # min heap, heapify make li[0] smallest item, inside call siftup
        heapq.heappush(li, 6)  # O(logn)    append new item to last then sift up (continue swap with parent if smaller
                               # than parent)
        v=heapq.heappop(li)    # O(logn)   move last item to first then sift down (continue swap with larger child node)
        print(li[0])  # 2    # get smallest O(1)
        heapq.merge(li, li2)   # merge 2 list  O(n)
        heap.nlargest(3,li)  # [5,4,3]   nsmallest

        # max heap
        heapq._heapify_max(li)                  # heapify
        li.append(6)                            # heappush
        heapq._siftdown_max(li, 0, len(li)-1)   # heappush
        v=heapq._heappop_max(ll)                # heap pop

    sliding window queue
        get maximum number of sliding window for each step
        enqueue right item, dequeue most left item, traverse through queue and remove item smaller than right item, and
        left most item in window is the largest item at this step
        O(n) all item is push and pop once although double loop

HashSet, HashMap:
    set and dict
    remove item once done, can save time complexity by eliminate duplicate search on that item
    for geometry coordinates problem can consider slope as dict key

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

    post order traversal is the delete tree node order, also easier for math operation with values and operand as node
        push values in stack, when met operand, pop two values and push the result back in stack

    recursion has top down approach: visit current node first, if current node answer is known, then pass down deduced
        child node answer for calculation when calling recursively on its child nodes, similar to preorder traversal

        def maxDepth(self, root):   # top down
            self.maximum = 0   # define instance variable and inner function to avoid global variable in leetcode
            def f(node, depth):
                if not node.left and not node.right:
                    self.maximum = max(self.maximum, depth + 1)
                if node.left: f(node.left, depth+1)
                if node.right: f(node.right, depth+1)
            if not root: return 0
            f(root, 0)
            return self.maximum

        bottom up approach: get the answer for child node first, then process current node base on child nodes' result,
            similar to post order traversal
        def bottom_down(node):
            if not root:
                return 0
            left_val = top_down(node.left)
            right_val = top_down(node.right)
            return func(left_val, right_val)


    class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

    binary search tree: left child less than parent, right child larger than parent. in order traversal return sorted
        list

    trie
    class Node:
    def __init__(self, char: str):
        self.char = char
        self.children = []
        self.is_finished = False
        self.counter = 1


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
                j = lps[j]
        i = j = 0
        while i < h and j < n:
            if j == -1 or haystack[i] == needle[j]:
                i += 1
                j += 1
            else:
                j = lps[j]
        return i-j if j == n else -1

    print(strStr('ababcababababd','ababd'))   # 9 index of match start


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