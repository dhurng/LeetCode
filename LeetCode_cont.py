from LeetCode import MyStack

def main():
    test = Solution()

    a = TreeNode(6)
    b = TreeNode(2)
    c = TreeNode(0)
    d = TreeNode(4)
    e = TreeNode(3)
    f = TreeNode(5)
    g = TreeNode(8)
    h = TreeNode(7)
    i = TreeNode(9)

    a.left = b
    a.right = g

    b.left = c
    b.right = d

    d.left = e
    d.right = f

    g.left = h
    g.right = i

    test.lowestCommonAncestor(a, e, f)

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        return str(self.val)

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

    def deleteNode(self, node):
        """
        
        :param node: 
        :return: 
        """

    def lowestCommonAncestor(self, root, p, q):
        """
        given bst, find lca of 2 given nodes
        :param root: treenode
        :param p: treenode
        :param q: treenode
        :return: treenode
        """
        p_stack = []
        q_stack = []

        self.lowestCommonAncestor_helper(root, p, p_stack)
        self.lowestCommonAncestor_helper(root, q, q_stack)

        if not p_stack or not q_stack:
            return None

        i = 0
        while i < len(p_stack) and i < len(q_stack):
            if p_stack[i] != q_stack[i]:
                break
            i += 1
        print p_stack[i - 1]
        return p_stack[i - 1]

    def lowestCommonAncestor_helper(self, root, node, stack):
        if not root:
            return
        stack.append(root)
        if node.val < root.val:
            self.lowestCommonAncestor_helper(root.left, node, stack)
        elif node.val > root.val:
            self.lowestCommonAncestor_helper(root.right, node, stack)
        else:
            return root

    def isPalindrome(self, head):
        """
        Given a singly linked list, determine if it is a palindrome.
        :param head: listnode
        :return: bool
        """
        # now with just O(1) space
        curr = runner = head
        while runner and runner.next:
            runner = runner.next.next
            curr = curr.next

        prev = None
        while curr:
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next

        while prev:
            if prev.val != head.val:
                return False
            prev = prev.next
            head = head.next
        # return it to original?
        return True

        # q = []
        # new_head = self.fun_reverse(head, q)
        # print q
        # self.print_list(new_head)
        # i = 0
        # while new_head:
        #     if new_head.val != q[i]:
        #         return False
        #     new_head = new_head.next
        #     i += 1
        # return True

    def fun_reverse(self, head, q):
        """
        reverse a linked list in place
        :param head: listnode
        :return: listnode
        """
        prev = None
        curr = head
        while curr:
            q.append(curr.val)
            next = curr.next
            curr.next = prev
            prev = curr
            curr = next
        return prev

    def print_list(self, node):
        while node:
            print node.val
            node = node.next

    def isPowerOfTwo(self, n):
        """
        Given an integer, write a function to determine if it is a power of two.
        :param n: int
        :return: bool
        """
        if n <= 0:
            return False
        while n > 1:
            if n % 2 == 1:
                return False
            n /= 2
        return True



    def invertTree(self, root):
        """
        Invert a binary tree
        :param root: treeNode
        :return: treeNode
        """
        if root:
            temp = root.left
            root.left = self.invertTree(root.right)
            root.right = self.invertTree(temp)
            return root

    def print_tree(self, root):
        if not root:
            return
        print root
        self.print_tree(root.left)
        self.print_tree(root.right)


class MyQueue(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack1 = []
        self.stack2 = []

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
    #     constact append
        self.stack1.push(x)

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
    #     linear time
        while self.stack1:
            self.stack2.append(stack1.pop)
        res = self.stack2.pop()
        self.stack1, self.stack2 = self.stack2, self.stack1
        return res

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
    #     ultimately the same without the pop

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        return True if not self.stack1 else False

if __name__ == '__main__':
    main()