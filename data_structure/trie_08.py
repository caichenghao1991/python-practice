class Node:
    def __init__(self, char: str):
        self.char = char
        self.children = []
        self.is_finished = False
        self.counter = 1


class Trie:
    def __init__(self):
        self.root = Node("*")

    def add(self, word: str):
        curr = self.root

        for char in word:
            exist = False
            for child in curr.children:
                if child.char == char:
                    child.counter += 1
                    curr = child
                    exist = True
                    break
            if not exist:
                node = Node(char)
                curr.children.append(node)
                curr = node
        curr.is_finished = True

    def find_prefix(self, prefix: str):
        curr = self.root
        for char in prefix:
            exist = False
            for child in curr.children:
                if child.char == char:
                    curr = child
                    exist = True
                    break
            if not exist:
                return 0
        return curr.counter


if __name__ == '__main__':
    trie = Trie()
    trie.add("Harry")
    trie.add("Hi")
    print(trie.find_prefix("H"))
    print(trie.find_prefix("Hi"))
    print(trie.find_prefix("Z"))
    print("")
