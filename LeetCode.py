"""
Leet code problems
"""

def main():
    test = Solution()

    nums = [1, 1, 2]
    print test.removeDuplicates(nums)

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

    def removeElement(self, nums, val):
        """
        given array and val, remove all instance of value in place
        return new length
        :param nums: list[int]
        :param val: int
        :return: int 
        """

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
