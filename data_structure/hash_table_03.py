class Node:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.next = None


class HashTable:
    """
        hashtable with fix size buckets and linked list if collision of keys
    """
    def __init__(self, size):
        self.data = [None] * size

    def _hash(self, key):
        hash_value = 0
        for i, c in enumerate(key):
            hash_value += (ord(c) + i) % len(self.data)
        return hash_value

    def set(self, key, value):
        h = self._hash(key)
        if self.data[h]:
            node = self.data[h]
            if key == node.key and node.value != value:
                node.value = value
            elif key != node.key:
                node.next = Node(key, value)
        else:
            self.data[h] = Node(key, value)

    def get(self, key):
        h = self._hash(key)
        curr = self.data[h]
        while curr:
            if curr.key == key:
                return curr.value
            else:
                curr = curr.next
        return None


if __name__ == '__main__':
    hashTable = HashTable(3)
    hashTable.set('a', -1)
    hashTable.set('a', 1)
    hashTable.set('b', 2)
    hashTable.set('c', 3)
    hashTable.set('d', 4)
    print(hashTable.get('a'))
