"""
Leet code problems
"""

def main():
    test = Solution()

    print test.isIsomorphic("", "")

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

    def isIsomorphic(self, s, t):
        """
        isomorphic - char in s can be replaced to get t, preserve order
        no 2 char can map to same character but can map to itself
        assume same length
        :param s: str
        :param t: str
        :return: bool
        """
        map = {}
        map2 = {}
        for pos, char in enumerate(s):
            if char not in map:
                map[char] = [pos]
            else:
                map[char].append(pos)

        for pos, char in enumerate(t):
            if char not in map2:
                map2[char] = [pos]
            else:
                map2[char].append(pos)

        return sorted(map.values()) == sorted(map2.values())

    def countPrimes(self, n):
        """
        Count the number of prime numbers less than a non-negative number, n.
        :param n: int
        :return: int
        """
        if n <= 2:
            return 0
        prime = [True] * n
        prime[0] = prime[1] = False
        for i in xrange(2, n):
            if prime[i]:
                for j in xrange(i << 1, n , i):
                    prime[j] = False
        count = filter(lambda x: x, prime)
        print len(count)
        return len(count)

        # if n <= 2:
        #     return 0
        # i = 2
        # count = 0
        # while i < n:
        #     if self.countPrimes_helper(i):
        #         count += 1
        #     i += 1
        # print count
        # return count

    # takes too long
    def countPrimes_helper(self, n):
        for i in xrange(2, n):
            if n % i == 0:
                return False
        return True

    def removeElements(self, head, val):
        """
        Remove all elements from a linked list of integers that have value val.
        :param head: listnode
        :param val: int
        :return: listnode
        """
        temp = ListNode(0)
        temp.next = head

        prev, curr = temp, temp.next
        while curr:
            if curr.val == val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        return temp.next

    def isHappy(self, n):
        """
        number is happy: starting with positive int, replace number by the sum
        of the squares of its digits, repeat process until the number equals 1
        or loop endlessly in cycle when does not include 1. 
        :param n: int
        :return: bool
        """
        if not n or n < 0:
            return False
        mem = set()
        while n != 1:
            print n
            print "MEM", mem
            n = self.isHappy_helper(n)
            if n not in mem:
                mem.add(n)
            else:
                return False
        return True

    def isHappy_helper(self, n):
        sum = 0
        while n > 0:
            sum += (n % 10) ** 2
            n /= 10
        return sum

    def rob(self, nums):
        """
        dont take 2 adjacent ints from a list, and get the most sum
        :param nums: list[int]
        :return: int
        """
        last = 0
        now = 0
        for i in nums:
            last, now = now, max(last + i, now)
        return now

    def reverseBits(self, n):
        """
        Reverse bits of given 32 bits unsigned int
        :param n: int
        :return: int
        """
        res = 0
        for i in xrange(32):
            res = (res << 1) + (n & 1)
            n >>= 1
        return res

    def rotate(self, nums, k):
        """
        rotate list nums by k steps
        :param nums: list[int]
        :param k: int
        :return: none do in place
        """
        if k is None or k <= 0:
            return
        k = k % len(nums)
        end = len(nums) - 1
        self.rotate_helper(nums, 0, end - k)
        self.rotate_helper(nums, end - k + 1, end)
        self.rotate_helper(nums, 0, end)

        print nums

    def rotate_helper(self, nums, beg, end):
        while beg < end:
            temp = nums[beg]
            nums[beg] = nums[end]
            nums[end] = temp

            beg += 1
            end -= 1


    def titleToNumber(self, s):
        """
        Return number as in excel sheet
        :param s: str
        :return: int
        """
        nums = [int(x) for x in range(1, 10)]
        res = 0
        for i in s:
            res = 26 * res + int(i, 36) - 9
        return res


    def majorityElement(self, nums):
        """
        Given an array of size n, find the majority element. 
        The majority element is the element that appears more than n/2 times.
        :param nums: list[int]
        :return: int 
        """
        map = {}
        for i in nums:
            if i not in map.keys():
                map[i] = 1
            else:
                map[i] += 1
        print map
        freq_val = 0
        freq_key = 0
        for val,key in map.iteritems():
            if key > freq_key:
                freq_val = val
                freq_key = key
        return freq_val

    def convertToTitle(self, n):
        """
        given positive integer, return corresponding column title in Excel Sheet
        :param n: int
        :return: str
        """
        capitals = [chr(x) for x in range(ord('A'), ord('Z') + 1)]
        res = []
        while n > 0:
            res.append(capitals[(n - 1) % 26])
            n = (n - 1) // 26
        res.reverse()
        return ''.join(res)

    def getIntersectionNode(self, headA, headB):
        """
        find the node at the intersection of two singly linked lists begins
        :param headA: listnode
        :param headB: listnode
        :return: listnode
        """
        if not headA or not headB:
            return None

        curr_a = headA
        curr_b = headB
        a_len = 0
        while curr_a:
            curr_a = curr_a.next
            a_len += 1

        b_len = 0
        while curr_b:
            curr_b = curr_b.next
            b_len += 1

        diff = abs(a_len - b_len)
        print diff

        longer = headA
        shorter = headB
        if b_len > a_len:
            longer = headB
            shorter = headA

        while diff > 0:
            longer = longer.next
            diff -= 1

        while longer and shorter:
            if longer is shorter:
                print "same", longer
                return True
            longer = longer.next
            shorter = shorter.next
        return False


    def hasCycle(self, head):
        """
        check if there is a cycle in the list
        :param head: listnode
        :return: bool
        """
        if not head:
            return False
        curr = head
        runner = curr
        while runner and runner.next:
            runner = runner.next.next
            curr = curr.next
            if curr == runner:
                return True
        return False

    def singleNumber(self, nums):
        """
        return which number only appears once
        :type nums: List[int]
        :rtype: int
        """
        res = 0
        for n in nums:
            res ^= n
        return res


    def isPalindrome2(self, s):
        """
        check if palindrome
        :param s: str
        :return: bool
        """
        if s == " ":
            return True
        i = 0
        j = len(s) - 1

        while i < j:
            while i < j and not s[i].isalpha():
                i += 1
            while i < j and not s[j].isalpha():
                j -= 1
            if s[i].lower() != s[j].lower():
                return False
            i += 1
            j -= 1
        return True


    def hasPathSum(self, root, sum):
        """
        given binary tree and sum, determine if tree has root-leaf path
        so that adding up all values along path equals to the sum
        :param root: treeNode
        :param sum: int
        :return: bool
        """
        return self.hasPathSum_helper(root, 0, sum)

    def hasPathSum_helper(self, node, curr_sum, trget_sum):
        if not node:
            return False
        if not node.left and not node.right:
            return curr_sum + node.val == trget_sum

        return self.hasPathSum_helper(node.left, curr_sum + node.val, trget_sum) or \
            self.hasPathSum_helper(node.right, curr_sum + node.val, trget_sum)

    def minDepth(self, root):
        """
        find min depth of binary tree, which is number of nodes along the shortest path
        from root down to nearest leaf node
        :param root: treeNode
        :return: int
        """
        if not root:
            return 0
        if not root.left:
            return self.minDepth(root.right) + 1
        if not root.right:
            return self.minDepth(root.left) + 1
        return min(self.minDepth(root.left), self.minDepth(root.right)) + 1

    def isBalanced(self, root):
        """
        check if tree is height balanced
        :param root: treeNode
        :return: bool
        """
        if not root:
            return True
        return abs(self.isBalanced_helper(root.left) - self.isBalanced_helper(root.right)) <= 1 and \
                self.isBalanced(root.left) and self.isBalanced(root.right)

    def isBalanced_helper(self, node):
        if not node:
            return 0
        return max(self.isBalanced_helper(node.left), self.isBalanced_helper(node.right)) + 1


    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return
        beg = 0
        end = len(nums) - 1
        mid = (beg + end) / 2

        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid + 1:])

        return root

    def levelOrderBottom(self, root):
        """
        given binary tree, return bottom-up traversal, left to right
        :param root: treeNode
        :return: List
        """
        if not root:
            return []
        curr_level = [root]
        while curr_level:
            next_level = []
            for i in curr_level:
                print i.val
                if i.left:
                    next_level.append(i.left)
                if i.right:
                    next_level.append(i.right)
            curr_level = next_level

        print curr_level
        print next_level

        # if not root:
        #     return []
        # res = []
        # self.levelOrderBottom_helper(root, 0, res)
        # return res.reverse()

    def levelOrderBottom_helper(self, node, level, res):
        if node:
            if len(res) < level + 1:
                res.append([])
            res[level].append(node.val)
            self.levelOrderBottom_helper(node.left, level + 1, res)
            self.levelOrderBottom_helper(node.right, level + 1, res)

    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1

    def isSymmetric(self, root):
        """
        check if mirror of itself
        :param root: treeNode
        :return: bool
        """
        if not root:
            return True
        return self.isSymmetric_helper(root.left, root.right)

    def isSymmetric_helper(self, left, right):
        if not left and not right:
            return True
        if not left or not right:
            return False
        if left.val == right.val:
            outer = self.isSymmetric_helper(left.left, right.right)
            inner = self.isSymmetric_helper(left.right, right.left)
            return outer and inner
        else:
            return False

    def isSameTree(self, p, q):
        """
        given 2 binary trees, check if equal
        :param p: treeNode
        :param q: treeNode
        :return: bool
        """
        if p and q:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return p is q

    def create_list(self, treeNode, res):
        if not treeNode:
            return
        res.append(treeNode.val)
        if treeNode.left:
            res.append(self.create_list(treeNode.left, res))
        if treeNode.right:
            res.append(self.create_list(treeNode.right, res))

    def merge(self, nums1, m, nums2, n):
        """
        Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        m, n = m - 1, n - 1
        while m >= 0 and n >= 0:
            if nums1[m] > nums2[n]:
                nums1[m + n + 1] = nums1[m]
                m -= 1
            else:
                nums1[m + n + 1] = nums2[n]
                n -= 1
        if n != -1:  # nums2 is still left
            nums1[:n + 1] = nums2[:n + 1]

    def binaryTreePaths(self, root):
        """
        return all paths from root to leaf
        :param root: treeNode
        :return: treeNode
        """
        if not root:
            return []
        res = []
        self.binaryTreePaths_helper(root, "", res)
        return res

    def binaryTreePaths_helper(self, root, ls, res):
        if not root.left and not root.right:
            res.append(ls + str(root.val))
        if root.left:
            self.binaryTreePaths_helper(root.left, ls + str(root.val) + "->", res)
        if root.right:
            self.binaryTreePaths_helper(root.right, ls + str(root.val) + "->", res)


    def isAnagram(self, s, t):
        """
        given two strings, deterine if t is anagram of s
        contain only lowercase
        :param s:
        :param t:
        :return:
        """
        my_dict = {}
        if (not s) ^ (not t):
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

    def deleteDuplicates(self, head):
        """
        given sorted linked list, delete all duplicates
        :param head: listnode
        :return: listnode
        """
        if not head:
            return None
        if not head.next:
            return head

        curr = head
        runner = curr.next

        while curr and runner:
            if curr.val == runner.val:
                curr.next = runner.next
                runner.next = None
                # runner = curr.next
            else:
                curr = runner
            runner = curr.next

        return head

    def print_list(self, listnode):
        curr = listnode
        while curr:
            print curr
            curr = curr.next

    def climbStairs(self, n):
        """
        climbing staircase, takes n steps to reach the top, each time you either climb
        1 or 2 steps. how many distinct ways to climb the top?
        :param n: int
        :return: int
        """
        # if n == 1:
        #     return 1
        # if n == 2:
        #     return 2
        # return self.climbStairs(n - 1) + self.climbStairsn(n - 2)
    #     memoization
        steps = [1,1]
        while len(steps) < n + 1:
            steps.append(steps[-1] + steps[-2])
        return steps[n]


    def mySqrt(self, x):
        """
        computer and return sqrt of x
        :param x: int
        :return: int
        """
        a = x
        b = (a + 1) // 2
        while b < a:
            a = b
            b = (a + x // a) // 2
        return a

    def addBinary(self, a, b):
        """
        given two binary strings, return their sum (also a binary string)
        :param a: str
        :param b: str
        :return: str
        """
        if not a or not b:
            return a or b
        res = ""
        carry = 0
        fill_len = max(len(a), len(b))

        a = a.zfill(fill_len)
        b = b.zfill(fill_len)

        i = fill_len - 1
        while i >= 0:
            sum = int(a[i]) + int(b[i]) + carry
            if sum == 2:
                res = res + "0"
                carry = 1
            elif sum > 2:
                res = res + "1"
                carry = 1
            else:
                res = res + str(sum)
                carry = 0
            i -= 1
        if carry == 1:
            res = res + "1"
        print "total", res[::-1]
        return res[::-1]


    def plusOne(self, digits):
        """
        given non-neg int represented as non-empty array of digits, plus one to int
        no leading 0s except 0 itself
        :param digits: list[int]
        :return: list[int]
        """
        end = len(digits) - 1
        while 0 <= end:
            if digits[end] < 9:
                digits[end] += 1
                return digits
            else:
                digits[end] = 0
            end -= 1

        new_number = []
        new_number.append(1)

        for i in digits:
            new_number.append(i)
        print new_number
        return new_number

    def lengthOfLastWord(self, s):
        """
        given string of upper/lower/empty return the length of last word
        if not then return 0
        :param s: str
        :return: int
        """
        # check if only whitespace
        if not s:
            return 0
        s = s.strip()
        beg = 0
        end = len(s) - 1
        while beg <= end:
            if s[end] != ' ':
                end -= 1
            else:
                break
        print s[end + 1:]
        return len(s[end + 1:])

    def maxSubArray(self, nums):
        """
        find contiguous subarray within an array(at least 1 number) w/largest sum
        :param n: int
        :return: str
        """
        if not nums:
            return 0
        curr_sum = nums[0]
        max_sum = nums[0]
        for n in nums[1:]:
            curr_sum = max(n, curr_sum + n)
            max_sum = max(curr_sum, max_sum)
        return max_sum


    def searchInsert(self, nums, target):
        """
        given sorted array and target, return index if target is found,
        else return where it would be if in order
        :param nums: list[int]
        :param target: int
        :return: int
        """
        if target > nums[len(nums) - 1]:
            return len(nums)
        if target < nums[0]:
            return 0
        beg = 0
        end = len(nums) - 1
        while beg <= end:
            mid = (beg + end) / 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                end = mid - 1
            else:
                beg = mid + 1
        return beg

    def strStr(self, haystack, needle):
        """
        returns index of first occurence of needle in hay or -1
        :param haystack: str
        :param needle: str
        :return: int
        """
        for i in range(len(haystack) - len(needle) + 1):
            if haystack[i:i + len(needle)] == needle:
                return i
        return -1


    def removeElement(self, nums, val):
        """
        given array and val, remove all instance of value in place
        return new length, order can be changed
        :param nums: list[int]
        :param val: int
        :return: int
        """
        if not nums:
            return 0
        start = 0
        end = len(nums) - 1

        while start <= end:
            if nums[start] == val:
                nums[start] = nums[end]
                nums[end] = nums[start]
                end = end - 1
            else:
                start += 1
        print nums
        return start

    def removeDuplicates(self, nums):
        """
        remove duplicates in place so each element appears once and return length
        no extra space, doesnt matter what you leave beyond new length ;]
        :param nums: list[int]
        :return: int
        """
        if not nums:
            return 0
        i = 0
        for j in range(1, len(nums)):
            if nums[i] != nums[j]:
                i += 1
                nums[i] = nums[j]
        print nums[:j]
        return len(nums[:j])

    def romanToInt(self, s):
        """
        given roman numeral convert to int
        familiarize how roman numerals work!
        :param s: str
        :return: int
        """
        roman = {'M': 1000,'D': 500 ,'C': 100,'L': 50,'X': 10,'V': 5,'I': 1}
        res = 0
        for i in range(len(s) - 1):
            if roman[s[i]] < roman[s[i + 1]]:
                res -= roman[s[i]]
            else:
                res += roman[s[i]]
        return res + roman[s[-1]]

    def mergeTwoLists(self, l1, l2):
        """
        merge 2 sorted linked lists and return as new list
        :param l1:
        :param l2:
        :return:
        """
        if not l1 or not l2:
            return l1 or l2
        new = ListNode(0)
        head = new
        while l1 and l2:
            print "run"
            if l1.val < l2.val:
                new.next = l1
                new = new.next
                l1 = l1.next
            else:
                new.next = l2
                new = new.next
                l2 = l2.next
        if not l1 and l2:
            print "no more 1"
            new.next = l2
        if not l2 and l1:
            print "no more 2"
            new.next = l1
        return head.next

        # recursive
        # if not l1 or not l2:
        #     return l1 or l2
        # if l1.val < l2.val:
        #     l1.next = self.mergeTwoLists(l1.next, l2)
        #     return l1
        # else:
        #     l2.next = self.mergeTwoLists(l1, l2.next)
        #     return l2

    def isValid(self, str):
        """
        Check if string has correct open and close brackets/parenthesis
        :param str: str
        :return: bool
        """
        if not str:
            return True
        map = {'}':'{', ']':'[', ')':'('}
        stack = []
        for char in str:
            if char in map.values():
                stack.append(char)
            elif char in map.keys():
                open = ""
                if len(stack) > 0:
                    open = stack.pop()
                if map[char] != open:
                    return False
        return False if len(stack) > 0 else True

    def sort_color(self, colors):
        if colors is None:
            return
        red = 0
        white = 0
        blue = 0
        for i in range(len(colors)):
            if colors[i] == 0:
                red += 1
            elif colors[i] == 1:
                white += 1
            elif colors[i] == 2:
                blue += 1
        print "red", red
        print "white", white
        print "blue", blue

        for j in range(len(colors)):
            if red > 0:
                colors[j] = 0
                red -= 1
            elif white > 0:
                colors[j] = 1
                white -= 1
            elif blue > 0:
                colors[j] = 2
                blue -= 1

        return colors

    def lcp(self, s1, s2):
        i = 0
        while i < len(s1) and i < len(s2):
            if s1[i] == s2[i]:
                i += 1
            else:
                break
        return s1[:i]

    def longestCommonPrefix(self, strs):
        """
        find longet common prefix string from array of strings
        :param strs: List[str]
        :return: str
        """
        if not strs:
            return ""
        return reduce(self.lcp, strs)

    def isPalindrome(self, x):
        """
        determine if int is palindrome without extra space
        :param x: int
        :return: bool
        """
        rev = self.no_space_reverse(x)
        return True if x - rev == 0 else False

    def no_space_reverse(self, x):
        if x > 0:
            return int(str(x)[::-1])
        else:
            return int('-' + str(x)[:0:-1])

    def reverse(self, x):
        """
        reverse an integer, assume 32-bit signed int and return 0 if overflow
        :param x: int
        :return: int
        """
    #     for python dont have max and min so no overflow
        neg = False
        res = 0
        if x < 0:
            neg = True
            x = x * -1
        while x > 0:
            right = x % 10
            res = (res * 10) + right
            x = x/10
        if neg:
            return res * -1
        return res

    def twoSum(self, nums, target):
        """
        Return indices of two numbers so they add to target
        :param nums: List[int]
        :param target: int
        :return: list of positions
        """
        if len(nums) <= 1:
            return False
        buff_dict = {}
        for i, val in enumerate(nums):
            if val in buff_dict:
                return [buff_dict[val], i]
            else:
                buff_dict[target - val] = i

if __name__ == '__main__':
    main()