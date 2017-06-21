from LeetCode import MyStack
import Queue
import sys


def main():
    test = Solution()

    # grid = [[0 for i in range(4)] for j in range(4)]
    grid = [[0,1,2,3],
            [4,5,6,7],
            [8,9,10,11]]
    # grid[1][2] = 'x'
    # grid[1][3] = 'x'
    # grid[2][4] = 'x'
    # grid[3][5] = 'x'
    print test.is_rotiational("waterbottle", "erbottlewat")


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

    def __repr__(self):
        return str(self.val)

class Actor(object):
    def __init__(self, name):
        self.name = name
        self.actors = set()
        self.bacon = -1

    def add_actor(self, actor):
        self.actors.add(actor)
        actor.actors.add(self)

    def __repr__(self):
        return self.name

class Solution(object):
    def __init__(self):
        self._largest = -sys.maxint

    def is_rotiational(self, str1, str2):
        """
        Use substring method/needle in haystack given 2 strings, check if s2
        is rotation of s1 using only 1 call
        :param input_str: str
        :return: bool
        """
        if not str1 and not str2:
            return True
        if not str1 or not str2:
            return False
    #     append itself twice to see and use a list vs concat
        total = []
        for i in str2:
            total.append(i)
        for i in str2:
            total.append(i)
        total = ''.join(total)

        # check below, only returns the position but if its not -1 its true
        pos = self.needle_haystack(str1, total)
        return True if pos != -1 else False



    def zero_matrix(self, grid):
        """
        if element in mxn matrix is 0, then entire row and column are 0
        :param grid: list[list[int]]
        :return: list[list[int]]
        """
        print grid
    #     since we cannot just look for a 0 and set it due to conflict and everything would be set to 0
    #     keep track of row and column
        row = len(grid)
        col = len(grid[0])
        #     use 2 arrays to keep track of rows/cols with 0
        arr1 = [False] * row
        arr2 = [False] * col

        for r in xrange(row):
            for c in xrange(col):
                if grid[r][c] == 0:
                    print "found you at ", r, c
                    arr1[r] = True
                    arr2[c] = True
        print arr1, arr2

        for i in range(len(arr1)):
            if arr1[i]:
                self.null_row(grid, i)

        for j in range(len(arr2)):
            if arr2[j]:
                self.null_col(grid, j)

        print grid

    def null_row(self, grid, row):
        for j in range(len(grid[0])):
            grid[row][j] = 0

    def null_col(self, grid, col):
        for i in range(len(grid)):
            grid[i][col] = 0


    def rotate_matrix(self, grid):
        """
        rotate a matrix by 90 degrees
        :param grid: list[list[int]]
        :return: list[list[int]]
        """
        print grid
        if len(grid) == 0 or len(grid) != len(grid[0]):
            print "no bueno"
            return False

        n = len(grid)
        for layer in range(n/2):
            # go from first section of layer to the end
            for i in range(layer, n - 1 - layer, 1):
                grid[layer][i], grid[n - 1 - i][layer], grid[i][n - 1 - layer], \
                grid[n - 1 - layer][n - 1 - i] = grid[n - 1 - i][layer], \
                grid[n - 1 - layer][n - 1 - i], grid[layer][i], grid[i][n - 1 - layer]
        print grid
        return grid

    def strstr(self, str1, str2):
        """
        implement substring runtime is O(ab)
        :param str1: str
        :param str2: str
        :return: int
        """
        if not str1 or not isinstance(str1, str):
            return False
        if not str2 or not isinstance(str2, str):
            return False

        for i in xrange(len(str1) - len(str2) + 1):
            if str1[i:i + len(str2)] == str2:
                return i
        return -1

    def str_compression(self, input_str):
        """
        compress string to have count follow it
        :param str: str
        :return: str
        """
        if not input_str:
            return ""
        counter = 1
        res = []
        i = 0
        for j in range(1, len(input_str)):
            if input_str[i] != input_str[j]:
                res.append(input_str[i])
                res.append(str(counter))
                counter = 1
                i = j
            else:
                counter += 1
        res.append(input_str[i])
        res.append(str(counter))

        res = ''.join(res)

        if len(res) > len(input_str):
            return input_str
        return res


    def one_away(self, str1, str2):
        """
        see if 2 strings are either insert, delete, or replace away
        :param str1: str
        :param str2: str
        :return: bool
        """
        if str1 == str2:
            return True
        if abs(len(str1) - str(str2)) > 1:
            return False
        count = 0
        i, j = 0, 0
        while i < str1 and j < str2:
            if str1[i] != str2[j]:
                if count == 1:
                    return False

                if len(str1) > len(str2):
                    i += 1
                elif len(str1) < len(str2):
                    j += 1
                else:
                    i += 1
                    j += 1
                count += 1
            else:
                i += 1
                j += 1

        if i < str1 and j < str2:
            count += 1

        return count == 1

        # or
        # uses extra map space
        # if len(str1) > len(str2):
        #     long = str1
        #     short = str2
        # else:
        #     long = str2
        #     short = str1
        # print long, short
        #
        # if len(long) - len(short) > 1:
        #     return False
        # map = dict()
        # for i in short:
        #     if i not in map:
        #         map[i] = 1
        #     else:
        #         map[i] += 1
        # print map
        #
        # diff = 0
        # for j in long:
        #     if j not in map:
        #         diff += 1
        #     else:
        #         map[j] -= 1
        # print diff
        #
        # if -1 in map.values():
        #     return False
        #
        # if diff > 1:
        #     return False
        #
        # return True


    def palindrome_permu(self, str):
        """
        given string check if permutation of palindrome
        :param str: str
        :return: bool
        """
        if not str:
            return True
        str = str.lower()
        str = ''.join(str.split())

        map = dict()
        for i in str:
            if i not in map:
                map[i] = 1
            else:
                map[i] += 1
        print map

        odd_count = 0
        for j in map.values():
            if j % 2 != 0:
                odd_count += 1
        if odd_count > 1:
            return False

        print "it is"
        return True

    def find_deepest_node(self, root):
        """
        find deepest node in bst using bfs 
        :param root: treenode
        :return: treenode
        """
        if not root:
            return
        res = []
        q = Queue.Queue()
        q.put(root)
        while not q.empty():
            curr = q.get()
            res.append(curr)
            if curr.left:
                q.put(curr.left)
            if curr.right:
                q.put(curr.right)
        return res[-1]

    def count_island(self, grid):
        """
        count number of islands
        :param grid: 
        :return: 
        """
        print grid
        print "*****"

        if not grid:
            return 0

        row = len(grid)
        col = len(grid[0])

        count = 0

        for i in xrange(row):
            for j in xrange(col):
                if grid[i][j] == 'x':
                    # counting number of islands not the number of x's
                    self.visit_island(i, j, grid)
                    count += 1
        print count
        return count

    def visit_island(self, row, col, grid):
        # verify it is island and check its neighbors
        if row >= 0 and row < len(grid) and col >= 0 and col < len(grid[0]) and grid[row][col] == 'x':
            # marked all connected islands visited
            grid[row][col] = 'm'
            self.visit_island(row + 1, col, grid)
            self.visit_island(row - 1, col, grid)
            self.visit_island(row, col + 1, grid)
            self.visit_island(row, col - 1, grid)

    def is_valid_bst(self, root):
        """
        validate bst
        :param root: treenode 
        :return: bool
        """
        min = float('-inf')
        max = float('inf')
        return self.is_valid_bst_helper(root, min, max)

    def is_valid_bst_helper(self, root, min, max):
        if not root:
            return True
        if root.val <= min or root.val >= max:
            return False
        return self.is_valid_bst_helper(root.left, min, root.val) and self.is_valid_bst_helper(root.right, root.val, max)

    def merge_sort(self, nums):
        """
        merge sort 
        :param nums: list[int]
        :return: list[int]
        """
        if len(nums) < 2:
            return nums
        mid = len(nums) // 2
        left = nums[:mid]
        right = nums[mid:]

        self.merge_sort(left)
        self.merge_sort(right)

        i, j, k = 0, 0, 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                nums[k] = left[i]
                i += 1
            else:
                nums[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            nums[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            nums[k] = right[j]
            j += 1
            k += 1

        print "merging ", nums


    def needle_haystack(self, needle, haystack):
        """
        needle in haystack question, index of match
        :param needle: str
        :param haystack: str
        :return: int 
        """
        if needle == "":
            return 0
        for i in range(len(haystack) - len(needle) + 1):
            for j in range(len(needle)):
                if haystack[i + j] != needle[j]:
                    break
                if j == len(needle) - 1:
                    return i
        return -1

    def print_reverse(self, nums):
        rev = nums[::-1]
        print rev
        res = ""
        for i in rev:
            res += str(i) + " "
        print res

    def lonely(self, nums):
        """
        find lonely number when all others are in pairs
        :param nums: list[int]
        :return: int
        """
        # bit manipulation way
        res = 0
        for i in nums:
            res ^= i
        print res


        # if not nums:
        #     return
        # # use dict way
        # map = dict()
        # for i in nums:
        #     if i not in map:
        #         map[i] = 1
        #     else:
        #         map[i] += 1
        # print map
        #
        # for key,val in map.iteritems():
        #     if val == 1:
        #         print "lonely: ", key
        #         return key


    def remove_duplicates(self, nums):
        """
        should do this in place
        :param nums: list[int]
        :return: int
        """
        if not nums:
            return 0
        curr = 0
        for i in range(1, len(nums)):
            if nums[curr] == nums[i]:
                curr += 1
                nums[curr] = nums[i]
        return curr + 1


    def free_time(self, sorted_list):
    #   sorted_list = sorted(list_of_people)
    #   [[9,11], [9,12], [12.5,13.5], [13,14], [17, 24]]
    #   res = [0, 9] [12, 12.5], [14,17]
        start = 0
        res = []
        for time in sorted_list:
            beg = time[0]
            end = time[1]
            if start > beg and start < end:
                start = end
            elif start != beg:
                res.append([start, beg])
                start = end

        if end != 24:
            res.append([start, 24])

        print res
        return res

    def longest_sub_distinct(self, word):
        """
        find the longest substring with the number of distinct letters
        :param word: str
        :param size: int
        :return: int
        """
        if not word:
            return 0
        print word

        dis = set()
        longest = 0
        curr = 0
        for i in word:
            if i not in dis:
                if len(dis) == 2:
                    dis.clear()

                    if curr > longest:
                        longest = curr
                    curr = 1
                dis.add(i)

            if i in dis:
                curr += 1
            if curr > longest:
                longest = curr
        print longest
        return longest


    def rpn_calc(self, ops):
        """
        create simple rpn calculator/polish calc
        :param ops: str
        :return: int
        """
        if not ops:
            return 0
        ops = ops.split(' ')
        print ops
        stack = []
        operations = ['-', '+', '*', '/']
        for val in ops:
            if val in operations and len(stack) != 0:
                op1 = stack.pop()
                op2 = stack.pop()
                if val == '-':
                    res = op2 - op1
                if val == '+':
                    res = op2 + op1
                if val == '*':
                    res = op2 * op1
                if val == '/':
                    res = op2 / op1
                stack.append(res)
            else:
                stack.append(float(val))
        print stack.pop()
        # return stack.pop()

    def combination_of_string(self, c_str):
        """
        Find all combinations of a string recursively
        :param c_str: str
        :return: list[str]
        """
        if len(c_str) < 2:
            return [c_str]
        res = []
        for i, c in enumerate(c_str):
            print c_str[:i] + c_str[i + 1:]
            for r in self.combination_of_string(c_str[:i] + c_str[i + 1:]):
                res.append([c])
        return res

    def permutations_of_string(self, p_str):
        """
        Print all permutations of a string recursively
        :param orig_str: str
        :return: list[str]
        """
        # base case single letter
        if len(p_str) < 2:
            return p_str

        # remove head like part of merge sort recursive part
        perms = self.permutations_of_string(p_str[1:])
        head = p_str[0]
        res = []

        for perm in perms:
            for i in range(len(perm) + 1):
                # put head in every location
                res.append(perm[:i] + head + perm[i:])
        print res
        return res


    def binary_search_rec(self, nums, trgt):
        """
        Binary Search on sorted array of int to find index of given int
        :param nums: list[int] 
        :param trgt: int
        :return: int
        """
        if not nums or not trgt:
            print "you fail"
            return None
        # check for any weird conditions, beg < end
        # not in the list, not sorted list etc
        beg = 0
        end = len(nums) - 1
        return self.binary_search_rec_helper(nums, beg, end, trgt)

    def binary_search_rec_helper(self, nums, beg, end, trgt):
        mid = (beg + end) / 2
        if nums[mid] == trgt:
            return mid
        elif nums[mid] < trgt:
            return self.binary_search_rec_helper(nums, mid + 1, end, trgt)
        else:
            return self.binary_search_rec_helper(nums, beg, mid - 1, trgt)


    def factorial(self, n):
        res = [0] * n
        self.factorial_helper(n, res, 0)
        print res
        return res

    def factorial_helper(self, n, res, level):
        if n <= 1:
            res[level] = 1
            return 1
        res[level] = n * self.factorial_helper(n - 1, res, level + 1)
        return res

    def moving_avg(self, l, N):
        """
        feed or have a list of ints with block size return average of all block sizes
        :param nums: Actualy a stream of data, but just put it into the list 
        :return: list[int]
        """
        print l, N
        sum = 0
        result = list(0 for x in l)

        for i in range(N):
            sum += l[i]
            result[i] = sum / (i + 1)

        print result, sum

        # calculate the rest
        for i in range(N, len(l)):
            print "s", sum, i, l[i -N], l[i]
            sum = sum - l[i - N] + l[i]
            result[i] = sum / N

        print result
        return result


    def int_to_string(self, num):
        stack = []

        neg = ""
        if num < 0:
            num *= -1
            neg = "-"

        while num > 0:
            right = num % 10

            stack.append(chr(right + ord('0')))

            num //= 10

        res = ''.join(stack[::-1])
        print type(res), res

    def string_to_int(self, str_int):
        if not str_int:
            return

        # deals with less operations
        flag = 0
        if str_int[0] == '-':
            flag = 1
        final = 0
        for i in xrange(flag, len(str_int)):
            final = final * 10 + (ord(str_int[i]) - ord('0'))
        if flag == 1:
            final *= -1
        print final

        print "** OR **"

        # can also implement neg func
        res = 0
        i = 0
        end = len(str_int) - 1
        while end >= 0:
            place = 10 ** i
            res += (ord(str_int[end]) - 48) * place
            end -= 1
            i += 1
        print res
        print type(res)

        # for i, j in enumerate(str_int):
        #     hun = 10 ** i
        #     res += int(i) * hun
        # print res

    # also dont use str
    def reverse_better(self, str):
        # try writing this quickly and correct errors as you go
        if not str:
            return ""

        # change to arr prior to call so you wont have to convert each time
        arr = self.reverse_inplace(str)
        i = 0
        # make sure this doesnt start at some spaces
        while arr[i] == " ":
            i += 1

        for j in range(1, len(arr)):
            if arr[j] == " ":
                if arr[i] != " ":
                    print self.reverse_inplace(arr[i:j])
                    i = j + 1

        if i != j:
            print self.reverse_inplace(arr[i:j + 1])


    def reverse_inplace(self, str):
        # another way without too many python funcs
        arr = list(str)
    #     cannot use in place for python or java
        for i in range(len(str)/2):
            arr[i], arr[len(str) - 1 - i] = arr[len(str) - 1 - i], arr[i]
        return arr

    def remove_char(self, str, rmv):
        """
        remove specified chars in a word
        :param str: str
        :param rmv: str
        :return: str
        """
        # uses O(n + m) space and runs O(n)
        res = ""
        nono = set(rmv)

        print str
        print nono

        for i in str:
            if i not in nono:
                # actually much slower than just join
                res += i
        print res

    def sixBacon(self, kb, actor):
        """
        6 degrees of bacon game
        :param kb: actor
        :param actor: actor
        :return: int
        """
        # can precompute bacon number O(n) for edges since expect n << m (nodes)
        if not actor.actors:
            print "No connections"
            return 0

        q = Queue.Queue()
        q.put(actor)
        while not q.empty():
            curr_actor = q.get()
            if curr_actor is kb:
                print "FOUND HIM AT ", curr_actor.bacon
                return curr_actor.bacon

            print curr_actor, curr_actor.bacon, curr_actor.actors

            for i in curr_actor.actors:
                if i.bacon == -1:
                    i.bacon = curr_actor.bacon + 1
                    q.put(i)
        return False


    def tree_rotate_right(self, root):
        """
        Balance BST, more nodes in left than right
        :param root: treenode
        :return: treenode
        """
        # better to implement as method of TreeNode class O(1) since constant ops
        if not root:
            return
        new_root = root.left
        root.left = new_root.right
        new_root.right = root
        print new_root
        return new_root

    def unbalanced_bst(self, root):
        """
        Given unbalanced bst with more nodes in left than right, reorg to improve
        balance but also keep as bst
        :param root: treenode
        :return: treenode
        """
        # Brute force way
        arr = []
        #     since bst can use inorder trav so just O(n)
        self.unbalanced_bst_helper(root, arr)
        print arr
    #     convert to bst
        res = self.unbalance_bst_helper_2(arr)
        self.print_tree(res)

    def unbalance_bst_helper_2(self, arr):
        if not arr:
            return None
        beg = 0
        end = len(arr) - 1
        mid = (beg + end) / 2

        root = arr[mid]
        root.left = self. unbalance_bst_helper_2(arr[:mid])
        root.right = self.unbalance_bst_helper_2(arr[mid+1:])

        return root

    def unbalanced_bst_helper(self, root, arr):
        if not root:
            return
        self.unbalanced_bst_helper(root.left, arr)
        arr.append(root)
        self.unbalanced_bst_helper(root.right, arr)


    def bt_to_heap(self, root):
        """
        Given set of int in unordered bin tree, us array sorting routine to 
        turn tree into heap that uses balance binary tree as ds
        :param nums: treenode
        :return: treenode
        """
        arr = []
        self.bt_to_heap_help(root, arr)

        print arr

        sort_arr = sorted(arr, key=self.sort_key)

        print sort_arr

        for i in range(len(sort_arr)):
            left = 2*i + 1
            right = left + 1

            if left < len(sort_arr):
                sort_arr[i].left = TreeNode(sort_arr[left])
            if right < len(sort_arr):
                sort_arr[i].right = TreeNode(sort_arr[right])
            else:
                sort_arr[i].left = None
                sort_arr[i].right = None

        for i in sort_arr:
            print "NODE", i
            print "l", i.left
            print "r", i.right
            print

    def sort_key(self, treenode):
        return treenode.val


    def bt_to_heap_help(self, root, arr):
        if not root:
            return
        arr.append(root)
        self.bt_to_heap_help(root.left, arr)
        self.bt_to_heap_help(root.right, arr)

    def lca_bst(self, root, p, q):
        """
        find lowest common anc in bst
        :param node1: treenode
        :param node2: treenode
        :return: treenode
        """
        if not root:
            return
        if root.right and root.val < min(p.val, q.val):
            return self.lowestCommonAncestor(root.right, p, q)
        elif root.left and root.val > max(p.val, q.val):
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root


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
        self.in_order(root.left)
        print root
        self.in_order(root.right)

    def post_order(self, root):
        if not root:
            return None
        self.post_order(root.left)
        self.post_order(root.right)
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
        if node.next:
            node.val = node.next.val
            node.next = node.next.next
        else:
            print "Cannot delete the last node!"

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