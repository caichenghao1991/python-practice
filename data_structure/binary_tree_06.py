class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, value):
        node = TreeNode(value)
        if not self.root:
            self.root = node
        else:
            curr = self.root
            while True:
                if value < curr.value:
                    if curr.left:
                        curr = curr.left
                    else:
                        curr.left = node
                        return node
                else:
                    if curr.right:
                        curr = curr.right
                    else:
                        curr.right = node
                        return node

    def find(self, value):
        curr = self.root
        while curr:
            if value < curr.value:
                curr = curr.left
            elif value > curr.value:
                curr = curr.right
            else:
                return True
        return False

    def remove(self, value):
        if not self.root:
            return False
        curr = self.root
        parent = None
        while curr:
            if value < curr.value:
                parent = curr
                curr = curr.left
            elif value > curr.value:
                parent = curr
                curr = curr.right
            else:
                # current node has no left child
                if not curr.left:
                    if not parent:
                        self.root = curr.right
                    else:
                        if parent.left == curr:
                            parent.left = curr.right
                        else:
                            parent.right = curr.right

                # current node has no right child
                elif not curr.right:
                    if not parent:
                        self.root = curr.left
                    else:
                        if parent.left == curr:
                            parent.left = curr.left
                        else:
                            parent.right = curr.left

                # current node has both child
                else:
                    # find the left most child in the right subtree if there is one
                    if curr.right.left:
                        left_most = curr.right.left
                        left_most_parent = curr.right
                        while left_most.left:
                            left_most = left_most.left
                            left_most_parent = left_most_parent.left
                        left_most_parent.left = left_most.right
                        left_most.left = curr.left
                        left_most.right = curr.right
                    else:
                        left_most = curr.right
                        left_most.left = curr.left

                    if not parent:
                        self.root = left_most
                    else:
                        if parent.left == curr:
                            parent.left = left_most
                        else:
                            parent.right = left_most
                return True
            
        return False

    def inorder_traversal(self, root):
        if root:
            self.inorder_traversal(root.left)
            print(root.value)
            self.inorder_traversal(root.right)


if __name__ == '__main__':
    tree = BinaryTree()
    tree.insert(9)
    tree.insert(4)
    tree.insert(6)
    tree.insert(20)
    tree.insert(170)
    tree.insert(15)
    tree.insert(1)
    tree.inorder_traversal(tree.root)
    '''
            9
         4     20
       1  6  15  170
    '''
    print(tree.find(20))
    print(tree.remove(9))
