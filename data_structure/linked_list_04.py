class Node:
    def __init__(self, value):
        self.value = value
        self.next = None
        self.prev = None


class LinkedList:
    def __init__(self, value):
        self.head = Node(value)
        self.length = 1
        self.tail = self.head

    def get_index(self, index):
        if type(index) == int and 0 <= index < self.length:
            curr = 0
            node = self.head
            while curr < index:
                node = node.next
                curr += 1
            return node

    def prepend(self, val):
        node = Node(val)
        node.next = self.head
        self.head.prev = node
        self.head = node
        self.length += 1

    def append(self, val):
        node = Node(val)
        self.tail.next = node
        node.prev = self.tail
        self.tail = node
        self.length += 1
    
    def insert_at_index(self, index, val):
        node = Node(val)
        if self.get_index(index):
            post = self.get_index(index)
            post.prev = node
            node.next = post
        if self.get_index(index - 1):
            pre = self.get_index(index - 1)
            pre.next = node
            node.prev = pre
            self.length += 1
            return node
        elif index == 0:
            self.head = node
            self.length += 1
            return node
        return

    def delete(self, index):
        if type(index) == int and 0 < index < self.length:
            node = self.get_index(index - 1)
            node.next = node.next.next
            if node.next.next:
                node.next.next.prev = node
            self.length -= 1
        elif index == 0:
            self.head = self.head.next
            self.length -= 1

    def search(self, val):
        node = self.head
        index = 0
        while node:
            if node.value == val:
                return index
            else:
                node = node.next
                index += 1
        return -1

    def print_list(self):
        node = self.head
        res = []
        while node:
            res.append(str(node.value))
            node = node.next
        print("-->".join(res))

    def size(self):
        return self.length

    def reverse(self):
        prev = None
        curr = self.head
        while curr:
            temp = curr.next
            curr.next = prev
            curr.prev = temp
            prev = curr
            curr = temp
        self.head = prev


if __name__ == '__main__':
    myList = LinkedList(2)
    myList.prepend(1)
    myList.append(3)
    myList.delete(1)
    myList.insert_at_index(1, 4)
    myList.print_list()
    print(myList.search(3))
    
    print(myList.get_index(2).prev.value)
    print(myList.size())
    myList.reverse()
    myList.print_list()

    # print(myList.__str__())
    # print(myList.__dict__)
