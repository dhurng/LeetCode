from LeetCode import MyStack
import Queue

def main():
    test = Solution()

    # pattern = "abba"
    # string = "dog dog dog dog"
    # print test.wordPattern(pattern, string)

    a = TreeNode(3)
    b = TreeNode(9)
    c = TreeNode(20)
    d = TreeNode(15)
    e = TreeNode(7)
    f = TreeNode(10)

    a.left = b
    a.right = c

    c.left = d
    c.right = e

    b.left = f

    test.levelOrder(a)

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


    def max_height(self, root):
        if not root:
            return 0
        return max(self.max_height(root.left), self.max_height(root.right)) + 1

    def levelOrder(self, root):
        if not root:
            return []
        q = Queue.Queue()
        q.put(root)

        while not q.empty():
            temp = q.get()
            print temp
            if temp.left:
                q.put(temp.left)
            if temp.right:
                q.put(temp.right)


    def levelOrder_2(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        q = []
        res = []
        q.append(root)
        dummy = TreeNode(-99999999)
        q.append(dummy)
        self.lvl(root, dummy, q, res)

        for i in res:
            for j in i:
                print j
            print "*"
        return res

    def lvl(self, node, dummy, q, res):
        list = []
        while len(q) != 0:
            temp = q.pop(0)

            if temp.val != dummy.val:
                if temp.left:
                    q.append(temp.left)
                if temp.right:
                    q.append(temp.right)
                list.append(temp)
            else:
                res.append(list)
                list = []
                if len(q) > 0:
                    q.append(dummy)


    def wordPattern(self, pattern, str):
        """
        Given pattern and string, find if string follows same pattern
        :param pattern: str
        :param str: str
        :return: bool
        """
        if not pattern and not str:
            return True
        if not pattern or not str:
            return False

        map = {}
        j = 0
        for i in pattern:
            pos = self.get_substring(str, j)
            substring = str[j:pos]
            print "SUB", substring
            j = pos + 1

            if i not in map.keys():
                map[i] = substring

            else:
                if map.get(i) != substring:
                    return False
        print map
        return True

    def get_substring(self, word, j):
        k = j + 1
        while k < len(word):
            if word[k] == " ":
                break
            k += 1
        return k

    def moveZeroes(self, nums):
        """
        Given nums, move all 0s to end while keeping order
        :param nums: list[int]
        :return: none doing it in place
        """
        if not nums:
            return []
        i = 0
        for j in xrange(1, len(nums)):
            if nums[i] == 0:
                if nums[j] != 0:
                    nums[i] = nums[j]
                    nums[j] = 0
                else:
                    continue
            i += 1
        print nums

    def firstBadVersion(self, n):
        """
        suppose [1,2,...,n] want to find first bad one,
        following is rest is bad, given api isBadVersion
        :param n: int
        :return: int
        """
        beg = 1
        end = n - 1
        while beg <= end:
            mid = (beg + end) / 2
            if isBadVersion(mid):
                end = mid - 1
            else:
                beg = mid + 1
        return beg

    def missingNumber(self, nums):
        """
        Given array containing n distinct nums taken from 0,1,2,...,n
        find one that is missing from array
        :param nums: list[int]
        :return: int
        """
        sums = sum(nums)
        n = len(nums)

        ideal_sum = n * (n + 1) / 2
        print ideal_sum - sums
        return ideal_sum - sums

    def isUgly(self, num):
        """
        Check if ugly = pos num whose prime only 2,3,5
        :param num: 
        :return: 
        """
        if num <= 0:
            return False
        ugliness = [2,3,5]
        for i in ugliness:
            while num % i == 0:
                num //= i
        return num == 1


    def addDigits(self, num):
        """
        Given non-neg int, repeatedly add all digits until result has only 1 digit
        :param num: int
        :return: int
        """
        # or if you know math tricks
        # return num if num == 0 else num % 9 or 9
        if num <= 0:
            return 0

        sum = num
        while sum > 9:
            sum = self.addDigits_helper(sum)
        print sum
        return sum

    def addDigits_helper(self, num):
        sum = 0
        while num > 0:
            right = num % 10
            sum += right
            num //= 10
        return sum

    def binaryTreePaths(self, root):
        """
        given binary tree, return all the paths
        :param root: treeNode
        :return: list[treeNodes]
        """
        if not root:
            return []
        res = []
        self.binaryTreePaths_helper(root, "", res)

        return res

    def binaryTreePaths_helper(self, node, ls, res):
        if not node.left and not root.right:
            res.append(ls + str(root.val))
        if root.left:
            self.binaryTreePaths_helper(root.left, ls + str(root.val) + '->', res)
        if root.right:
            self.binaryTreePaths_helper(root.right, ls + str(root.val) + '->', res)

        res.append(node.val)
        # deal with lefts here?
        # at leaf so stop and begin from root again
        if not node.left and not node.right:
            self.binaryTreePaths_helper(node.left, res)
            self.binaryTreePaths_helper(node.right, res)

    def isAnagram(self, s, t):
        """
        check if t is anagram of s, assume only lowercase, what if unicode?
        :param s: str
        :param t: str
        :return: bool
        """
    #     just need to know if same letter and number of them
        my_dict = {}
        if not s and not t:
            return True
        if not s or not t:
            return False
        if len(s) != len(t):
            return False
        for i in xrange(len(s)):
            if s[i] not in my_dict.keys():
                my_dict[s[i]] = 1
            else:
                my_dict[s[i]] += 1
        for j in range(len(t)):
            if t[j] in my_dict.keys():
                my_dict[t[j]] -= 1
            else:
                return False
        for k in my_dict:
            if my_dict[k] != 0:
                return False
        return True


    def deleteNode(self, node):
        """
        delete node(except tail), given only access to that node
        :param node: listnode
        :return: none, modify in place
        """
        node.val = node.next.val
        node.next = node.next.next


    def deleteNode(self, node):
        node.val = node.next.val
        node.next = node.next.next

    def lowestCommonAncestor(self, root, p, q):
        """
        given bst, find lca of 2 given nodes
        :param root: treenode
        :param p: treenode
        :param q: treenode
        :return: treenode
        """
        # using no space
        if root.val > max(p.val, q.val):
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < min(p.val, q.val):
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root

        # p_stack = []
        # q_stack = []
        #
        # self.lowestCommonAncestor_helper(root, p, p_stack)
        # self.lowestCommonAncestor_helper(root, q, q_stack)
        #
        # if not p_stack or not q_stack:
        #     return None
        #
        # i = 0
        # while i < len(p_stack) and i < len(q_stack):
        #     if p_stack[i] != q_stack[i]:
        #         break
        #     i += 1
        # print p_stack[i - 1]
        # return p_stack[i - 1]

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

    def lowestCommonAncestor_notbst(self, root, n1, n2):
        if not root:
            return None
        if root == n1 or root == n2:
            return root
        left = self.lowestCommonAncestor_notbst(root.left, n1, n2)
        right = self.lowestCommonAncestor_notbst(root.right, n1, n2)
        if left and right:
            return root
        if not left and not right:
            return None
        return left if left else right

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