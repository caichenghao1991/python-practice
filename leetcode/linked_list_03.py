from typing import List


class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Node(object):
    def __init__(self, val, prev, next, child):
        self.val = val
        self.prev = prev
        self.next = next
        self.child = child


class Solution:
    # bottom up build linked list from array
    @staticmethod
    def build_linked_list(arr: List) -> ListNode:
        prev, curr = None, None
        for i in range(len(arr) - 1, -1, -1):
            curr = ListNode(arr[i], prev)
            prev = curr
        return prev

    @staticmethod
    def print_linked_list(start):
        s = []
        while start:
            s.append(start.val)
            start = start.next
        print('->'.join([str(_) for _ in s]))

    # 206. Reverse Linked List 3 pointer
    def reverseList(self, head: ListNode) -> ListNode:
        prev, curr = None, head
        while curr:
            next_node = curr.next
            curr.next = prev
            prev = curr
            curr = next_node
        return prev

    # 92. Reverse Linked List II
    def reverseBetween(self, head: ListNode, left: int, right: int) -> ListNode:
        position = 1
        start = ListNode()
        start.next = head
        dummy = start
        while position < left:
            start = start.next
            position += 1
        position = left

        curr, prev = start.next, start
        while position <= right:
            nex = curr.next
            curr.next = prev
            prev = curr
            curr = nex
            position += 1

        start.next.next = curr
        start.next = prev

        return dummy.next

    # 430. Flatten a Multilevel Doubly Linked List  only need consider 2nd level, flat one level a time
    def flatten(self, head: Node) -> Node:
        curr = head
        while curr:
            if curr.child:
                temp = curr.next
                tail = curr.child
                while tail.next:
                    tail = tail.next
                if temp:
                    temp.prev = tail
                tail.next = temp
                curr.child.prev = curr
                curr.next = curr.child
                curr.child = None
            else:
                curr = curr.next
        return head

    # 142. Linked List Cycle II
    # Floyd's tortoise and hare
    # fast and slow pointer meet at a, then 1 from start 1 from a, meet again at b, b is cycle start
    def detectCycle(self, head: ListNode):
        slow, fast = head, head
        while True:
            if not fast or not fast.next:
                return None
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                break
        slow = head
        while True:
            if fast == slow:
                return slow
            fast = fast.next
            slow = slow.next


if __name__ == '__main__':
    solution = Solution()
    node = solution.build_linked_list([1, 2, 3, 4, 5])
    solution.print_linked_list(node)
    n = solution.reverseList(node)
    solution.print_linked_list(n)

    node = solution.build_linked_list([1, 2, 3, 4, 5])
    n = solution.reverseBetween(node, 2, 4)
    solution.print_linked_list(n)

    c = Node(3, None, None, None)
    a, b = Node(1, None, None, None), Node(2, None, None, c)
    a.next = b
    b.prev = a
    n = solution.flatten(a)
    solution.print_linked_list(n)

    a, b, c = ListNode(1), ListNode(2), ListNode(3)
    a.next, b.next, c.next = b, c, b
    print(solution.detectCycle(a).val)
