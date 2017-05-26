def main():
    test = Solution()

    a = TreeNode(4)
    b = TreeNode(2)
    c = TreeNode(7)
    d = TreeNode(1)
    e = TreeNode(3)
    f = TreeNode(6)
    g = TreeNode(9)


    a.left = b
    a.right = c

    b.left = d
    b.right = e

    c.left = f
    c.right = g

    test.print_tree(a)
    test.invertTree(a)
    print "*****"
    test.print_tree(a)

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.val)

class Solution(object):
    def __init__(self):
        pass

    def invertTree(self, root):
        """
        Invert a binary tree
        :param root: treeNode
        :return: treeNode
        """
        if root:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
            return root

    def print_tree(self, root):
        if not root:
            return
        print root
        self.print_tree(root.left)
        self.print_tree(root.right)

if __name__ == '__main__':
    main()