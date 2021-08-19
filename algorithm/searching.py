from typing import List
import data_structure.binary_search_tree_06 as t
import queue


class Search:
    def __init__(self, **kwargs):
        if "array" in kwargs.keys():
            self.arr = kwargs["array"]
            self.arr.sort()
        if "root" in kwargs.keys():
            self.tree = kwargs["root"]

    def binary_search_first_occurrence(self, value):
        if not hasattr(self, "arr"):
            return -1
        left = 0
        right = len(self.arr) - 1
        res = -1

        while left <= right:
            mid = left + (right - left) // 2
            if self.arr[mid] > value:
                right = mid - 1
            elif self.arr[mid] < value:
                left = mid + 1
            else:
                # res = mid   for search an unique exist item
                res = mid
                right = mid - 1
        return res

    def binary_search_first_greater(self, value):
        if not hasattr(self, "arr"):
            return -1
        left = 0
        right = len(self.arr) - 1
        res = -1

        while left <= right:
            mid = left + (right - left) // 2
            if self.arr[mid] <= value:
                # if self.arr[mid] <= value:  #for first item greater or equal to target
                left = mid + 1
            else:
                res = mid
                right = mid - 1
        return res

    def breadth_first_search(self, node=""):
        if node == "":
            if hasattr(self, "tree"):
                curr = self.tree
            else:
                return []
        else:
            curr = node
        order = []
        q = queue.Queue()
        q.put(curr)
        while q.qsize() > 0:
            node = q.get()
            order.append(node.value)
            if node.left:
                q.put(node.left)
            if node.right:
                q.put(node.right)
        return order

    def breadth_first_search_recursive(self, search_queue: queue.Queue, order: List):
        if not hasattr(self, "tree"):
            return []
        if search_queue.qsize() == 0:
            return order
        else:
            node = search_queue.get()
            order.append(node.value)
            if node.left:
                search_queue.put(node.left)
            if node.right:
                search_queue.put(node.right)
            return self.breadth_first_search_recursive(search_queue, order)

    def depth_first_search_pre_order(self, node: t.TreeNode, order: List):
        order.append(node.value)
        if node.left:
            self.depth_first_search_pre_order(node.left, order)
        if node.right:
            self.depth_first_search_pre_order(node.right, order)
        return order

    def iterative_pre_order(self, node=""):
        if node == "":
            if hasattr(self, "tree"):
                curr = self.tree
            else:
                return []
        else:
            curr = node
        stack = []
        res = []
        stack.append(curr)
        while len(stack) > 0:
            node = stack.pop()
            res.append(node.value)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res

    def iterative_in_order(self, node=""):
        """
        1) Create an empty stack S.
        2) Initialize current node as root
        3) Push the current node to S and set current = current->left until current is NULL
        4) If current is NULL and stack is not empty then
            a) Pop the top item from stack.
            b) Print the popped item, set current = popped_item->right
            c) Go to step 3.
        5) If current is NULL and stack is empty then we are done.
        """
        if node == "":
            if hasattr(self, "tree"):
                curr = self.tree
            else:
                return []
        else:
            curr = node
        stack = []
        res = []
        while True:
            if curr:
                stack.append(curr)
                curr = curr.left
            elif len(stack) > 0:
                curr = stack.pop()
                if type(curr) is t.TreeNode:
                    res.append(curr.value)
                curr = curr.right
            else:
                return res

    def iterative_post_order(self, node=""):
        """
        1. Push root to first stack.
        2. Loop while first stack is not empty
            2.1 Pop a node from first stack and push it to second stack
            2.2 Push left and right children of the popped node to first stack
        3. Print contents of second stack
        """
        if node == "":
            if hasattr(self, "tree"):
                curr = self.tree
            else:
                return []
        else:
            curr = node
        stack1 = []
        stack2 = []
        res = []
        stack1.append(curr)
        while len(stack1) > 0:
            node = stack1.pop()
            stack2.append(node)
            if node.left:
                stack1.append(node.left)
            if node.right:
                stack1.append(node.right)

        while len(stack2) > 0:
            node = stack2.pop()
            res.append(node.value)
        return res


if __name__ == '__main__':
    arr = [5, 10, 1, 3, 5, 6, 2, 7]
    # 1 2 3 5 5 6 7 10
    arr2 = [9, 4, 6, 20, 170, 15, 1]
    tree = t.BinarySearchTree()
    tree.build_tree(arr2)
    '''         9
             4     20
           1  6  15  170
    '''
    search = Search(array=arr, root=tree.root)
    print(search.binary_search_first_occurrence(5))
    print(search.binary_search_first_greater(5))
    print(search.breadth_first_search())
    print(search.breadth_first_search(tree.root.left))
    qu = queue.Queue()
    qu.put(tree.root)
    print(search.breadth_first_search_recursive(qu, []))
    print(search.depth_first_search_pre_order(tree.root, []))
    print(search.iterative_pre_order(tree.root))
    print(search.iterative_in_order())
    print(search.iterative_post_order())
