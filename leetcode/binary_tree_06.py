from collections import deque
from typing import List
from queue import Queue


class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    @staticmethod
    def build_tree_with_array(arr: List[int]) -> TreeNode:
        q = Queue()
        root = None
        if arr:
            root = TreeNode(arr[0])
            q.put((root, 0))
        while not q.empty():
            t = q.get()
            left, right = 2 * t[1] + 1, 2 * t[1] + 2
            if left < len(arr):
                if arr[left]:
                    l_c = TreeNode(arr[left])
                    t[0].left = l_c
                    q.put((l_c, left))
                else:
                    t[0].left = None
            if right < len(arr):
                if arr[right]:
                    r_c = TreeNode(arr[right])
                    t[0].right = r_c
                    q.put((r_c, right))
                else:
                    t[0].right = None
        return root

    @staticmethod
    # 104. Maximum Depth of Binary Tree   t: O(n) Θ(logn)  s: O(n)  Θ(logn)
    def maxDepth(root: TreeNode) -> int:
        def helper(node, depth):
            if not node:
                return depth
            else:
                return max(helper(node.left, depth + 1), helper(node.right, depth + 1))

        return helper(root, 0)

    @staticmethod
    # 102. Binary Tree Level Order Traversal   t: O(n)  s: O(n) max width
    # bfs + 2 counter /bfs + nested while loop with 1 counter
    # or use second array avoid modification on same array
    def levelOrder(root: TreeNode) -> List[List[int]]:
        """
        q = Queue()  # q = []
        if root:
            # q.append(root)
            q.put(root)
        res, temp = [], []
        c1, c2 = 1, 0

        while not q.empty():  # while len(q):
            n = q.get()  # n = q.pop(0)
            temp.append(n.val)
            c1 -= 1
            if n.left:
                c2 += 1
                q.put(n.left)  # q.append(n.left)
            if n.right:
                c2 += 1
                q.put(n.right)  # q.append(n.right)

            if c1 == 0:
                res.append(temp)
                temp = []
                c1, c2 = c2, 0
        return res
        """

        res, level = [], [root]
        while level and root:
            res.append([i.val for i in level])
            """
            temp = []
            for i in level:
                if i.left:
                    temp.append(i.left)
                if i.right:
                    temp.append(i.right)
            level = temp
            """
            level = [kid for node in level for kid in (node.left, node.right) if kid]
        return res

    @staticmethod
    # 199. Binary Tree Right Side View  dfs track level or bfs get right most
    # t: O(n)  s: O(n)  bfs queue (level) width / dfs height of tree
    def rightSideView(root: TreeNode) -> List[int]:
        """
        res = []
        def dfs(node, level):
            if node:
                if level > len(res):
                    res.append(node.val)
                dfs(node.right, level + 1)
                dfs(node.left, level + 1)

        dfs(root, 1)
        return res
        """
        res = []
        if root:
            level = [root]
            while level:
                res.append(level[-1].val)
                level = [kid for node in level for kid in (node.left, node.right) if kid]
        return res

    # 222. Count Complete Tree Nodes   binary search  / recursion   t: O((logn)^2)  s: O((logn)^2)
    def countNodes(self, root: TreeNode) -> int:
        """
        :type root: TreeNode
        :rtype: int
        """
        """
        if not root:
            return 0
        depth = self.getDepth(root)
        up = pow(2, depth - 1) - 1
        left, right = 0, up
        res = 0
        while left <= right:
            mid = left + (right - left) // 2

            if self.findNodeIndex(mid, depth, root):
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return up + res + 1
        """
        if not root:
            return 0
        left_depth = self.getDepth(root.left)
        right_depth = self.getDepth(root.right)
        print(left_depth, right_depth)
        if left_depth == right_depth:
            return pow(2, left_depth) + self.countNodes(root.right)
        else:
            return pow(2, right_depth) + self.countNodes(root.left)

    @staticmethod
    def findNodeIndex(index, depth, root):
        if depth == 0:
            return False
        steps = 1
        curr = root
        left, right = 0, pow(2, depth - 1) - 1
        while steps < depth:
            mid = left + (right - left) // 2
            if index <= mid:
                curr = curr.left
                right = mid
            else:
                curr = curr.right
                left = mid + 1
            steps += 1
        return curr is not None

    @staticmethod
    def getDepth(root):
        d = 0
        while root:
            root = root.left
            d += 1
        return d

    @staticmethod
    # 98. Validate Binary Search Tree   t: O(n)   s: O(1) if Function Call Stack size is not considered, otherwise O(n)
    def isValidBST(root):
        res = []
        def helper(node, lb, rb):
            if node.val <= lb or node.val >= rb:
                res.append(1)
                return False
            else:
                if node.left:
                    helper(node.left, lb, node.val)
                if node.right:
                    helper(node.right, node.val, rb)
        helper(root, float('-inf'), float('inf'))

        """
        def helper(node, lb, rb, res):
            if node == None:
                return res
            if node.val <= lb or node.val >= rb:
                return False
            else:
                return helper(node.left, lb, node.val, True) and helper(node.right, node.val, rb, True)
        return helper(root, float('-inf'), float('inf'), True)
        """
        return len(res) == 0


def postOrderIterative(root):
    if root is None:
        return

        # Create two stacks
    s1 = []
    s2 = []

    # Push root to first stack
    s1.append(root)

    # Run while first stack is not empty
    while s1:

        # Pop an item from s1 and
        # append it to s2
        node = s1.pop()
        s2.append(node)

        # Push left and right children of
        # removed item to s1
        if node.left:
            s1.append(node.left)
        if node.right:
            s1.append(node.right)

        # Print all elements of second stack
    while s2:
        node = s2.pop()
        print(node.data)


def inOrder(root):
    # Set current to root of binary tree
    current = root
    stack = []  # initialize stack
    done = 0

    while True:

        # Reach the left most Node of the current Node
        if current is not None:

            # Place pointer to a tree node on the stack
            # before traversing the node's left subtree
            stack.append(current)

            current = current.left


        # BackTrack from the empty subtree and visit the Node
        # at the top of the stack; however, if the stack is
        # empty you are done
        elif (stack):
            current = stack.pop()
            print(current.data, end=" ")  # Python 3 printing

            # We have visited the node and its left
            # subtree. Now, it's right subtree's turn
            current = current.right

        else:
            break

    print()


def iterativePreorder(root):
    # Base CAse
    if root is None:
        return

    # create an empty stack and push root to it
    nodeStack = []
    nodeStack.append(root)

    # Pop all items one by one. Do following for every popped item
    # a) print it
    # b) push its right child
    # c) push its left child
    # Note that right child is pushed first so that left
    # is processed first */
    while (len(nodeStack) > 0):

        # Pop the top item from stack and print it
        node = nodeStack.pop()
        print(node.data, end=" ")

        # Push right and left children of the popped node
        # to stack
        if node.right is not None:
            nodeStack.append(node.right)
        if node.left is not None:
            nodeStack.append(node.left)



    # iterative
    def preorderTraversal(self,root):
        stack, res = [root],[]
        while len(stack):
            n = stack.pop()
            if n:
                res.append(n.val)
                stack.append(n.right)
                stack.append(n.left)
        return res

    # recursive
    def preorderTraversal(self, root):
        res = []
        self.helper(root, res)
        return res

    def helper(self, root, res):
        if not root:   # raise error if use none.val
            return
        res.append(root.val)
        self.helper(root.left, res)
        self.helper(root.right, res)


    def inorderTraversal(self, root):
        stack, res = [], []
        curr = root
        while stack or curr:
            if curr:
                stack.append(curr)
                curr = curr.left
            else:
                node = stack.pop()
                res.append(node.val)
                curr = node.right
        return res

    def postorderTraversal(self, root):
        traversal, stack = [], [root]
        while stack:
            node = stack.pop()
            if node:
                # pre-order, right first
                traversal.append(node.val)
                stack.append(node.left)
                stack.append(node.right)

        # reverse result
        return traversal[::-1]

    def levelOrder(self, root):
        if not root:
            return []
        res, level = [], [root]
        while level:
            res.append([i.val for i in level])
            temp = []
            for i in level:
                if i.left:
                    temp.append(i.left)
                if i.right:
                    temp.append(i.right)
            level = temp
            # que = deque([root])
            # while que:
            #     size = len(que)
            #     result = []
            #     for _ in range(size):
            #         cur = que.popleft()
            #         result.append(cur.val)
            #         if cur.left:
            #             que.append(cur.left)
            #         if cur.right:
            #             que.append(cur.right)
            #     results.append(result)
        return res

    def levelOrder(self, root: TreeNode) -> List[List[int]]:
        res = []

        def helper(root, depth):
            if not root: return []
            if len(res) == depth: res.append([])  # start the current depth
            res[depth].append(root.val)  # fulfil the current depth
            if root.left: helper(root.left, depth + 1)  # process child nodes for the next depth
            if root.right: helper(root.right, depth + 1)

        helper(root, 0)
        return res
    
if __name__ == "__main__":
    solution = Solution()
    r = Solution.build_tree_with_array([3, 9, 20, None, None, 15, 7])
    print(Solution.maxDepth(r))
    print(Solution.levelOrder(r))
    print(Solution.rightSideView(r))
    print(solution.countNodes(r))
    r = Solution.build_tree_with_array([2, 1, 3])
    print(Solution.isValidBST(r))

