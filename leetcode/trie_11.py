class TrieNode:
    def __init__(self):
        self.end = False
        self.children = {}


class Trie(object):

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):   # t: O(n)   s: O(n)
        curr = self.root
        for i in range(len(word)):
            if word[i] not in curr.children:
                child = TrieNode()
                curr.children[word[i]] = child
            curr = curr.children.get(word[i])
            if i == len(word) - 1:
                curr.end = True

    def search(self, word: str) -> bool:   # t: O(n)   s: O(1)
        curr = self.root
        for c in word:
            if c not in curr.children:
                return False
            curr = curr.children[c]

        return curr.end

    def startsWith(self, prefix: str) -> bool:    # t: O(n)   s: O(1)
        curr = self.root
        for c in prefix:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return True


if __name__ == '__main__':
    obj = Trie()
    obj.insert("word")
    print(obj.search("word"))
    print(obj.startsWith("wo"))
