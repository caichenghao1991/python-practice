from queue import PriorityQueue
import heapq


class MinHeap:
    def __init__(self):
        self.data = []
        self.size = 0

    def min_heapify(self, pos):
        left = pos * 2 + 1
        right = pos * 2 + 2
        length = len(self.data)
        small = pos
        if left < length and self.data[small] > self.data[left]:
            small = left
        if right < length and self.data[small] > self.data[right]:
            small = right
        if small != pos:
            self.data[small], self.data[pos] = self.data[pos], self.data[small]
            self.min_heapify(small)

    def build_heap(self, array):
        self.data = array
        self.size = len(array)
        for i in range(len(array) // 2, -1, -1):
            self.min_heapify(i)
        return self.data

    def pop(self):
        value = self.data[0]
        self.data[0] = self.data[self.size - 1]
        self.size -= 1
        self.data.pop()
        self.min_heapify(0)
        return value

    def insert(self, value):   
        self.data.append(value)
        self.size += 1
        curr = self.size - 1
        while curr > 0 and self.data[curr] < self.data[(curr - 1) // 2]:
            self.data[curr], self.data[(curr - 1)//2] = self.data[(curr - 1)//2], self.data[curr]
            curr = (curr - 1) // 2


def built_in_methods():
    li = [4, 2, 5, 1, 3]
    # in-built method 1 min heap
    q = PriorityQueue()
    for _ in li:
        q.put(_)
    while not q.empty():
        print(q.get())

    # in-built method 2 min heap
    heapq.heapify(li)
    heapq.heappush(li, 6)
    while len(li) > 0:
        print(heapq.heappop(li))

    # max heap
    li2 = [4, 2, 5, 1, 3]
    li2 = [i * -1 for i in li2]
    heapq.heappush(li2, -6)
    heapq.heapify(li2)
    while len(li2) > 0:
        print(-heapq.heappop(li2))
    

if __name__ == "__main__":
    built_in_methods()
    heap = MinHeap()
    heap.build_heap([4, 2, 5, 1, 3])
    heap.insert(0)
    while heap.size > 0:
        print(heap.pop())
