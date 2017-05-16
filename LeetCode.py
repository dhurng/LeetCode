"""
Leet code problems
"""

def main():
    test = Solution()

    print test.isAnagram("rat", "tar")

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        return str(self.val)

class Solution(object):
    def __init__(self):
        pass

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
        """
        :param s1: str
        :param s2: str
        :return:
        """
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
