class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


class Stack:
    """
        stack implemented with linked list
    """

    def __init__(self):
        self.top = None
        self.bottom = None
        self.length = 0

    def push(self, value):
        node = Node(value)
        if not self.bottom:
            self.bottom = node
            self.top = node
        else:
            node.next = self.top
            self.top = node
        self.length += 1

    def pop(self):
        if self.top:
            node = self.top
            self.top = self.top.next
            if not self.top:
                self.bottom = None
            self.length -= 1
            return node
        return None

    def peek(self):
        return self.top


class Stack2:
    """
        stack implemented with array
    """

    def __init__(self):
        self.arr = []

    def push(self, value):
        self.arr.append(value)

    def pop(self):
        if len(self.arr):
            x = self.arr.pop()
            return x

    def peek(self):
        return self.arr[len(self.arr) - 1]


class Queue:
    def __init__(self):
        self.first = None
        self.last = None
        self.length = 0

    def enqueue(self, value):
        node = Node(value)
        if not self.first:
            self.first = node
            self.last = node
        else:
            self.last.next = node
            self.last = node
        self.length += 1

    def dequeue(self):
        if self.first:
            node = self.first
            self.first = self.first.next
            if not self.first:
                self.last = None
            self.length -= 1
            return node
        return None

    def peek(self):
        return self.first


class BuildStackWithQueue:
    def __init__(self):
        self.stack = Queue()

    def push(self, value):
        """
            first enqueue new item at last of queue
            then repeat new length - 1 times dequeue first item then enqueue it at last
            O(n) time complexity
        """
        self.stack.enqueue(value)
        steps = self.stack.length - 1
        while steps > 0:
            node = self.stack.dequeue()
            self.stack.enqueue(node.value)
            steps -= 1

    def pop(self):
        return self.stack.dequeue()

    def peek(self):
        return self.stack.peek()


if __name__ == '__main__':
    stack = Stack()
    stack.push(1)
    stack.push(2)
    print(stack.peek().value)
    stack.pop()
    print(stack.pop().value)
    stack.pop()  # test pop on empty stack

    stack2 = Stack2()
    stack2.push(1)
    stack2.push(2)
    print(stack2.peek())
    stack2.pop()
    print(stack2.pop())
    stack2.pop()  # test pop on empty stack

    queue = Queue()
    queue.enqueue(1)
    queue.enqueue(2)
    print(queue.peek().value)
    queue.dequeue()
    print(queue.dequeue().value)
    queue.dequeue()  # test dequeue on empty queue

    stack3 = BuildStackWithQueue()
    stack.push(1)
    stack.push(2)
    print(stack.peek().value)
    stack.pop()
    print(stack.pop().value)
    stack.pop()  # test pop on empty stack
