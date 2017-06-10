from LeetCode import MyStack
import Queue
import sys


def main():
    test = Solution()

    a = TreeNode(10)
    b = TreeNode(5)
    c = TreeNode(-3)
    d = TreeNode(3)
    e = TreeNode(2)
    f = TreeNode(11)
    g = TreeNode(3)
    h = TreeNode(-2)
    i = TreeNode(1)

    a.left = b
    a.right = c

    b.left = d
    b.right = e

    c.right = f

    d.left = g
    d.right = h

    e.right = i

    test.print_tree(a)
    print "****"
    test.pre_order_NoRec(a)

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None
        self.prev = None
        self.child = None

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

    def avg(self, nums):
        """
        :param nums: Actualy a stream of data, but just put it into the list 
        :return: list[int]
        """
        pass

    def pre_order_NoRec(self, root):
        """
        no recursion preorder 
        :param root: treenode
        :return: none just print
        """
        if not root:
            return
        stack = []
        stack.append(root)
        while len(stack) != 0:
            curr = stack.pop()
            print curr
            if curr.right:
                stack.append(curr.right)
            if curr.left:
                stack.append(curr.left)


    def find_height(self, root):
        if not root:
            return 0
        left = self.find_height(root.left)
        right = self.find_height(root.right)
        return max(right, left) + 1

    def pre_order(self, root):
        if not root:
            return None
        print root
        self.pre_order(root.left)
        self.pre_order(root.right)

    def in_order(self, root):
        if not root:
            return None
        self.pre_order(root.left)
        print root
        self.pre_order(root.right)

    def post_order(self, root):
        if not root:
            return None
        self.pre_order(root.left)
        self.pre_order(root.right)
        print root


    def search_BST(self, root, trgt):
        if not root:
            return None
        if root.val == trgt:
            return root
        elif root.val < trgt:
            return self.search_BST(root.right, trgt)
        else:
            return self.search_BST(root.left, trgt)

        # while root:
        #     if root.val == trgt:
        #         return root
        #     elif root.val < trgt:
        #         root = root.left
        #     else:
        #         root = root.right
        # return root

    def check_cycle(self, head):
        """
        Determine is list is cyclic or not
        :param head: listnode
        :return: bool
        """
        if not head:
            return False
        curr = head
        runner = curr.next

        while runner and runner.next:
            runner = runner.next.next
            curr = curr.next
            if curr is runner:
                return True
        return False


    def flatten_linked(self, head, tail):
        """
        Flatten a multi layered linked list into 1
        :param head: listnode
        :param tail: listnode
        :return: listnode
        """
        curr = head
        while curr:
            if curr.child:
                self.append(tail, curr.child)
            curr = curr.next
        return head

    def append(self, tail, node):
        tail.next = node
        node.prev = tail

        curr = node
        while curr.next:
            curr = curr.prev
        tail = curr

        # OR
        #
        # uses an extra data struct
        # curr = head
        # stack = []
        # while curr:
        #     if curr.child and curr.next:
        #         # save the next
        #         stack.append(curr.next)
        #         curr.next = curr.child
        #         curr.child = None
        #         curr = curr.next
        #     elif not curr.child and curr.next:
        #         curr = curr.next
        #     #     curr has no child and no next
        #     else:
        #         if len(stack) != 0:
        #             popped = stack.pop()
        #             curr.next = popped



    def mth_last(self, head, m):
        if not head:
            return None
        curr = head
        runner = curr.next

        count = 0
        while runner and count < m:
            runner = runner.next
            count += 1
        while runner:
            curr = curr.next
            runner = runner.next

        print curr
        return curr

    def reverse_sent(self, sent):
        res = sent.strip()
        res = res.split(" ")

        for i,j in enumerate(res):
            res[i] = j[::-1]

        print ' '.join(res)


    def pathSum(self, root, sum):
        """
        Each node has val, find number of paths that sum to given value
        does not need to start or end at root or leaf but trav downwards
        :param root: treenode
        :param sum: int
        :return: int
        """


        print "root", root

    def countSegments(self, s):
        """
        Count num of segments in str, where seg is contiguous seq of non space char
        :param s: str
        :return: int
        """
        if not s:
            return 0
        count = 0
        i = 0
        j = i + 1
        while j < len(s):
            if s[j] == " " and s[i] != " ":
                count += 1
                i = j
            else:
                i = j
            j += 1
        if i != j and s[i] != " ":
            count += 1

        print count
        return count

    def addStrings(self, num1, num2):
        """
        2 non-neg ints rep as sring, return sum
        :param num1: str
        :param num2: str
        :return: str
        """
        map = {'0': 0, '1':1, '2':2 , '3': 3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}
        if not num1 or not num2:
            return num1 or num2

        if len(num1) < len(num2):
            shorter = num1
            longer = num2
        else:
            shorter = num2
            longer = num1

        diff = len(longer) - len(shorter)
        new = []
        for i in range(diff):
            new.append("0")
        for j in shorter:
            new.append(j)

        shorter = ''.join(new)

        print longer, shorter

        k = len(longer) - 1
        carry = 0
        res = []

        while k >= 0:
            if shorter[k] in map and longer[k] in map:
                sum = map[shorter[k]] + map[longer[k]] + carry
                if sum > 9:
                    carry = 1
                    res.append(str(sum % 10))
                else:
                    carry //= 10
                    res.append(str(sum))
            k -= 1

        if carry > 0:
            res.append(str(carry))

        res = ''.join(res[::-1])
        return res


    def thirdMax(self, nums):
        """
        non-empty arr of int, return 3rd max number in arr, else max number
        must be linear runtime
        :param nums: list[int]
        :return: int
        """
        # nums = set(nums)
        # print nums
        # if len(nums) < 3:
        #     return max(nums)
        # nums.remove(max(nums))
        # nums.remove(max(nums))
        # return max(nums)


        max = -sys.maxint
        max_2 = -sys.maxint
        max_3 = -sys.maxint

        for i in nums:
            if i > max:
                max = i

        for j in nums:
            if j > max_2 and j != max:
                max_2 = j

        for k in nums:
            if k > max_3 and k != max and k != max_2:
                max_3 = k

        if max_3 == -sys.maxint:
            return max

        return max_3

    def fizzBuzz(self, n):
        """
        Output string representation of numbers from 1 to n, but multiples of 3 are Fizz
        5 is Buzz, and 3 and 5 are FizzBuzz
        :param n: int
        :return: str
        """
        res = []
        for i in range(1, n + 1):
            if i % 3 == 0 and i % 5 == 0:
                print "FizzBuzz"
                res.append("FizzBuzz")
            elif i % 3 == 0:
                print "Fizz"
                res.append("Fizz")
            elif i % 5 == 0:
                print "Buzz"
                res.append("Buzz")
            else:
                res.append(str(i))
        print res
        return res


    def longestPalindrome(self, s):
        """
        Given a string, find lenght of longest palindrome that can be built
        it is case sensative
        :param s: str
        :return: int
        """
        if not s:
            return 0

        map = {}
        for i in s:
            if i not in map:
                map[i] = 1
            else:
                map[i] += 1
        print map

        count_odd = 0
        res = 0
        for j in map.values():
            if j % 2 == 0:
                res += j
            else:
                res += j - 1
                count_odd = 1

        print "res", res + count_odd
        return res

    def toHex(self, num):
        """
        Given an int, convert to hex, for neg use two's complement
        :param num: int
        :return: str
        """
        res = []
        dict = {10: "a", 11: "b", 12: "c", 13: "d", 14: "e", 15: "f"}
        if num == 0:
            return "0"
        if num < 0:
            num = num + 2 ** 32

        while num > 0:
            digit = num % 16
            num = (num - digit) / 16
            if digit > 9 and digit < 16:
                digit = dict[digit]
            else:
                digit = str(digit)
            res.append(digit)
        return "".join(res[::-1])

    def sumOfLeftLeaves(self, root):
        """
        Find sum of all left leaves in given binary tree
        :param root: treenode
        :return: int
        """
        if not root:
            return 0
        if root.left and not root.left.left and not root.left.right:
            return root.left.val + self.sumOfLeftLeaves(root.right)
        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

    def findNthDigit(self, n):
        """
        find nth DIGIT of inifinite int sequence 0 < n < 2^31
        :param n: int
        :return: int
        """
        n -= 1
        for digits in range(1, 11):
            first = 10 ** (digits - 1)
            if n < 9 * first * digits:
                return int(str(first + n / digits))[n%digits]
            n -= 9 * first * digits

    def findTheDifference(self, s, t):
        """
        two strings, t is random string of s but add 1 more letter
        find the letter
        :param s: str
        :param t: str
        :return: str
        """
        if not s:
            return t[-1]
        map = {}
        for i in s:
            if i not in map:
                map[i] = 1
            else:
                map[i] += 1
        print map
        for j in t:
            if j in map:
                map[j] -= 1
            else:
                return j

        for k, l in map.iteritems():
            if l == -1:
                return k


    def firstUniqChar(self, s):
        """
        find first non-repeating character and return index else -1
        :param s: str
        :return: int
        """
        if not s:
            return -1
        map = {}
        for i in s:
            if i not in map:
                map[i] = 1
            else:
                map[i] += 1

        for j, k in enumerate(s):
            if map[k] == 1:
                print j, k
                return j
        return -1

    def canConstruct(self, ransomNote, magazine):
        """
        check if ransomnote can be constructed from magazine
        :param ransomNote: str
        :param magazine: str
        :return: bool
        """
        if not ransomNote:
            return True
        if not magazine:
            return False
        if len(magazine) < len(ransomNote):
            return False

        map = {}
        for i in magazine:
            if i not in map:
                map[i] = 1
            else:
                map[i] += 1
        print map

        for j in ransomNote:
            if j not in map:
                return False
            else:
                map[j] -= 1

        for k in map.values():
            if k < 0:
                return False
        return True

    def isPerfectSquare(self, num):
        """
        Given positive int num, check if num is perfect square
        :param num: int
        :return: bool
        """
        if num < 0: return False
        if num <= 1: return True
        half = num // 2
        s = set([half])
        while half * half != num:
            half = (half + (num // half)) // 2
            if half in s:
                return False
            s.add(half)
        return True

    def intersect(self, nums1, nums2):
        """
        Given 2 arrays, compute intersection can have dups
        :param nums1: list[int]
        :param nums2: list[int]
        :return: list[int]
        """
        dict1 = {}
        for i in nums1:
            if i not in dict1:
                dict1[i] = 1
            else:
                dict1[i] += 1

        print dict1
        res = []

        for i in nums2:
            if i in dict1 and dict1[i] > 0:
                res.append(i)
                dict1[i] -= 1
        return res


    def intersection(self, nums1, nums2):
        """
        Given 2 arrays, compute intersection
        :param nums1: list[int]
        :param nums2: list[int]
        :return: list[int]
        """
        a = set(nums1)
        b = set(nums2)
        res = set()

        for i in a:
            if i in b:
                res.add(i)
        print "res", res
        return res

    def reverseVowels(self, str):
        """
        reverse only the vowels
        :param s: str
        :return: str
        """
        if not str:
            return ""
        vowels = ["a", "e", "i", "o", "u", "A", "E","I","O","U"]
        beg = 0
        end = len(str) - 1

        s = []
        for i in str:
            s.append(i)
        print s

        while beg <= end:
            if s[beg] in vowels:
                if s[end] in vowels:
                    temp = s[beg]
                    s[beg] = s[end]
                    s[end] = temp
                    beg += 1
                    end -= 1
                else:
                    end -= 1
            else:
                beg += 1
        print ''.join(s)

    def reverseString(self, s):
        """
        reverse a string
        :param s: str
        :return: str
        """
        print s
        return s[::-1]

    def isPowerOfFour(self, num):
        """
        Determine if power of 4
        :param num: int
        :return: bool
        """
        """
        num within 32 bit and turn into binary and xor would be 
        1010101010101010101010101010101
        any power of 4 is also power of 2
        
        or take log of num on base 4 and if res is int then true
        
        or keep dividing by 4 if n%4 is not 0 
        """
        # return num != 0 and num &(num-1) == 0 and num & 1431655765== num
        if n <= 0:
            return False
        while n != 1:
            if n % 4 != 0:
                return False
            n /= 4
        return True


    def isPowerOfThree(self, n):
        """
        Determine if power of 3
        :param n: int
        :return: bool
        """
        #  1162261467 is the largest pow of 3 in the int
        return n > 0 and 1162261467 % n == 0

    def canWinNim(self, n):
        """
        Each turn to remove 1 - 3 stones, one to remove last wins
        determine if you can win by number of stones in heap
        :param n: int
        :return: bool
        """
        return True if n % 4 == 1 else False

    def hasPathSum(self, root, sum):
        if not root:
            return None
        if not root.left and not root.right:
            if sum == root.val:
                return True
            else:
                return False
        return self.hasPathSum(root.left, sum - root.val) or self.hasPathSum(root.right, sum - root.val)

        # if self.hasPathSum(root.left, sum - root.val):
        #     return True
        # if self.hasPathSum(root.right, sum - root.val):
        #     return True
        # return False


    def levelOrderBottom(self, root):
        """
        bottom up level traversal of binary tree
        :param root: treenode
        :return: list[list[int]]
        """
        if not root:
            return None
        q = Queue.Queue()
        stack= []
        q.put(root)

        while not q.empty():
            temp = q.get()
            if temp.left:
                q.put(temp.left)
            if temp.right:
                q.put(temp.right)
            stack.append(temp)


        while len(stack) != 0:
            print stack.pop()

    def max_height(self, root):
        if not root:
            return 0
        return max(self.max_height(root.left), self.max_height(root.right)) + 1

    def levelOrder(self, root):
        """
        level traversal of binary tree
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        q = Queue.Queue()
        res = []
        q.put(root)
        dummy = TreeNode(-99999999)
        q.put(dummy)
        self.lvl(root, dummy, q, res)

        for i in res:
            for j in i:
                print j
            print "*"
        return res

    def lvl(self, node, dummy, q, res):
        list = []
        while not q.empty():
            temp = q.get()

            if temp.val != dummy.val:
                if temp.left:
                    q.put(temp.left)
                if temp.right:
                    q.put(temp.right)
                list.append(temp)
            else:
                res.append(list)
                list = []
                if not q.empty():
                    q.put(dummy)


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
            # return last position
            pos = self.get_substring(str, j)
            substring = str[j:pos]
            print "SUB", substring
            j = pos + 1

            if i not in map.keys() and substring not in map.values():
                map[i] = substring
            else:
                if map.get(i) != substring:
                    return False
        print map
        return True if pos == len(str) else False

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

"""
Given an integer arr nums, find sum of elements between i and j (i <= j) incl
"""
class NumArray(object):
    def __init__(self, nums):
        """
        assume arr does not change
        Many calls to sumrange func
        :type nums: List[int]
        """
        self.nums = nums
        self.map = {-1:0}
        for pos,val in enumerate(nums):
            self.map[pos] = self.map[pos - 1] + val
        print self.map

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.map[j] - self.map[i - 1]

        # sum = 0
        # while i <= j:
        #     sum += self.nums[i]
        #     i += 1
        # print "sum", sum
        # return sum

# a = [-2, 0, 3, -5, 2, -1]
# tester = NumArray(a)
# print tester.nums
# print "**"
# print tester.sumRange(0,2)
# print "*"
# print tester.sumRange(2,5)
# print "*"
# print tester.sumRange(0,5)

if __name__ == '__main__':
    main()