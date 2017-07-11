import sys
import Queue

def main():
    test = Solution()

    # graph = {0: [1],
    #         1: [2],
    #         2: [3],
    #         3: [],
    #         4: [6],
    #         5: [4],
    #         6: [5]}
    # test.route_graph(graph, 1, 3)

    # a = Project("a", 1)
    # b = Project("b", 1)
    # c = Project("c", 1)
    # d = Project("d", 2)
    # e = Project("e", 0)
    # f = Project("f", 0)
    #
    # a.add(d)
    # f.add(b)
    # f.add(a)
    # b.add(d)
    # d.add(c)

    a = TreeNode(4)
    b = TreeNode(2)
    c = TreeNode(6)
    d = TreeNode(1)
    e = TreeNode(3)
    f = TreeNode(5)

    a.left = b
    a.right = c

    b.left = d
    b.right = e

    c.left = f

    res = []
    test.root_leaf_sum(a, 9, res)
    print res


class ListNode(object):
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

    def __repr__(self):
        return str(self.data)


class TreeNode(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right
        self.size = 0

    def __repr__(self):
        return str(self.data)

class GraphNode(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def __repr__(self):
        return str(self.data)

class Project(object):
    # quickly giving it values
    def __init__(self, data, in_count):
        self.data = data
        self.out = []
        self.in_count = in_count

    def add(self, project):
        self.out.append(project)

    def __repr__(self):
        return self.data

class Solution(object):
    def __init__(self):
        pass

    def paths_sum(self, root, sum):
        """
        find paths that add to sum, do not need root to leaf
        :param root: treenode
        :param sum: int
        :return: list[treenode]
        """
        if not root:
            return []
        my_map = dict()
        self.paths_sum_helper(root, sum, 0, my_map)

    def paths_sum_helper(self, root, sum, curr_sum, my_map):
        if not root:
            return 0

    #     count paths with sum at curr
        curr_sum += root.data
        sum = curr_sum - sum
        total = my_map.get(sum)

    #     if curr_sum == trgt sum then 1 path starts at root
        if curr_sum == sum:
            total += 1

        self.incrtable(my_map, curr_sum, 1)
        total += self.paths_sum_helper(root.left, sum, curr_sum, my_map)
        total += self.paths_sum_helper(root.right, sum, curr_sum, my_map)
        self.incrtable(my_map, curr_sum, -1)

        return total

    def incrtable(self, my_map, key, diff):
        new = my_map.get(key) + diff
        if new == 0:
            del my_map[key]
        else:
            my_map[key] = new

    def root_leaf_sum(self, root, sum, res):
        if not root:
            return False
        if not root.left and not root.right:
            if root.data == sum:
                res.append(root)
                return True
            else:
                return False
        # if its true, it adds itself into back of list
        if self.root_leaf_sum(root.left, sum - root.data, res):
            res.append(root.data)
            return True
        if self.root_leaf_sum(root.right, sum - root.data, res):
            res.append(root.data)
            return True
        return False

    def find_node(self, root, trgt):
        if not root:
            return
        print root
        if root is trgt:
            print "found"
            return root
        return self.find_node(root.left, trgt) or self.find_node(root.right, trgt)

    def check_subtree(self, n1, n2):
        """
        check if t2 is subtree of large n1
        :param n1: treenode
        :param n2: treenode
        :return: bool
        """
        res = self.search2(n1, n2)
        print "res", res
        # if res:
        #     nice = self.is_iden(res, n2)
        # print nice
        return res

    def is_iden(self, n1, n2):
        if not n1 and not n2:
            return True
        if not n1 or not n2:
            return False
        if n1 is not n2:
            return False
        if self.is_iden(n1.left, n2.left) and self.is_iden(n1.right, n2.right):
            return True

    def search2(self, root, root2):
        if not root:
            return False

        if root is root2 and self.is_iden(root, root2):
            return True
        # dfs way
        else:
            return self.search2(root.left, root2) or self.search2(root.right, root2)

        # q = Queue.Queue()
        # q.put(root)
        #
        # while not q.empty():
        #     popped = q.get()
        #     print popped
        #     if popped is root2:
        #         return popped
        #
        #     if popped.left:
        #         q.put(popped.left)
        #     if popped.right:
        #         q.put(popped.right)

    def print_null(self, root, res):
        if not root:
            res.append(root)
            return
        res.append(root)
        self.print_null(root.left, res)
        self.print_null(root.right, res)

    def first_ca(self, root, n1, n2):
        """
        Find first common ancestor of 2 nodes in binary tree not bst
        :param root: treenode
        :param n1, n2: treenodes
        :return: treenode
        """
        if not root:
            return
        # includes if it is both
        if root is n1 or root is n2:
            return root
        left = self.first_ca(root.left, n1, n2)
        right = self.first_ca(root.right, n1, n2)
        if left and right:
            return root
        if not left and not right:
            return
        return left if left else right


    def build_order(self, projects):
        """
        given lists of projects and dependencies, find the build order
        will just use unique strings to rep projects
        :param projects: list[str]
        :param depend: list[tuples(str,str)]
        :return: list[str]
        """
        print projects
        res = []
        for i in projects:
            if i.in_count == 0:
                res.append(i)
        print "res", res
        i = 0
        while i < len(res):
            curr = res[i]
            print curr
            if not curr:
                return None
            for j in curr.out:
                j.in_count -= 1
                # because we know that before is already 0 just cont loop
                if j.in_count == 0:
                    res.append(j)
            i += 1
        print res
        # runs O(d + p) p for proj, d for pair of dep
        return res

    def successor(self, root):
        """
        find the next node, in-order successor) of given node in bst
        assume each node has link to parent
        :param root: treenode
        :return: treenode
        """
    #     if there is right sub then return leftmost child
    #   else if node is right child keep going up
    #   either way return the parent
        if not root:
            return
        if root.right:
            return self.left_most(root.right)
        else:
            q = root
            # reference to parent
            x = q.parent
            while x and x.left is not q:
                q = x
                x = x.parent
            return x

    def left_most(self, root):
        if not root:
            return
        while root.left:
            root = root.left
        print root
        return root

    def copy_bst(self, root, arr):
        if not root:
            return
        self.copy_bst(root.left, arr)
        arr.append(root)
        self.copy_bst(root.right, arr)

    def validate_bst(self, root):
        """
        check if binary tree is bst
        :param root: treenode
        :return: bool
        """
        min = -sys.maxint
        max = sys.maxint
        return self.validate_bst_helper(root, min, max)

    def validate_bst_helper(self, root, min, max):
        if not root:
            return True
        if root.data <= min or root.data >= max:
            return False
        left = self.validate_bst_helper(root.left, min, root.data)
        right = self.validate_bst_helper(root.right, root.data, max)
        if left and right:
            return True
        return False

    def check_balanced(self, root):
        """
        check if binary tree is balanced
        :param root: 
        :return: 
        """
        # if not root:
        #     return True
        # diff = abs(self.find_height(root.left) - self.find_height(root.right))
        # return True if diff < 2 and self.check_balanced(root.left) and self.check_balanced(root.right) else False

        # fast fail method with O(n) and O(h) for height
        return True if self.find_height(root) != -sys.maxint else False

    def find_height(self, root):
        if not root:
            return 0

        left = self.find_height(root.left)
        if left == -sys.maxint:
            return -sys.maxint

        right = self.find_height(root.right)
        if right == -sys.maxint:
            return -sys.maxint

        diff = abs(right - left)
        if diff > 1:
            return -sys.maxint
        else:
            return max(self.find_height(root.left), self.find_height(root.right)) + 1

    def list_of_depths(self, root):
        """
        Given binary tree, create an ll of all nodes at each depth
        depth d so d linked lists
        :param root: treenode
        :return: linklist
        """
        # used preorder
        lists = []
        self.list_of_depths_helper(root, lists, 0)

        for i in lists:
            while i:
                print i
                i = i.next
            print " "
        return lists

    def list_of_depths_helper(self, node, lists, level):
        if not node:
            return
        if len(lists) == level:
            # dummy
            list = ListNode(-sys.maxint)
            lists.append(list)
        else:
            list = lists[level]
        #     add to the end
        while list.next:
            list = list.next
        list.next = ListNode(node.data)
        self.list_of_depths_helper(node.left, lists, level + 1)
        self.list_of_depths_helper(node.right, lists, level + 1)

    def min_tree(self, nums):
        """
        Given sorted unique int arr, create bst with minimal height
        :param nums: list[int]
        :return: treenode
        """
    #   just use the mid as root and break it apart
        if not nums:
            return None
        beg = 0
        end = len(nums) - 1
        mid = (beg + end) / 2

        root = TreeNode(nums[mid])
        root.left = self.min_tree(nums[:mid])
        root.right = self.min_tree(nums[mid + 1:])
        return root

    def print_tree(self, treenode):
        if not treenode:
            return
        print treenode
        self.print_tree(treenode.left)
        self.print_tree(treenode.right)


    def route_graph(self, graph, n1, n2):
        """
        Given a directed graph see if there is a route between 2 nodes
        :param graph: dict{int, list[int]}
        :param n1: int
        :param n2: int
        :return: bool
        """
        print graph
        if not n1 or not n2:
            return False

        if n1 == n2:
            return True

        # bfs
        q = Queue.Queue()
        q.put(n1)
        while not q.empty():
            popped = q.get()
            print popped, n2
            if popped == n2:
                print "got it"
                return True
            for i in graph[popped]:
                q.put(i)
        print "nope"
        return False

        # try bidirectional search
        # n1_q = Queue.Queue()
        # n2_q = Queue.Queue()
        #
        # n1_q.put(n1)
        # n2_q.put(n2)
        # while not n1_q.empty() and not n2_q.empty():
        #     popped_n1 = n1_q.get()
        #     popped_n2 = n2_q.get()
        #     print popped_n1, popped_n2
        #
        #     if popped_n1 != popped_n2:
        #         if len(graph[popped_n1]) != 0:
        #             for i in graph[popped_n1]:
        #                 print i
        #                 n1_q.put(i)
        #         if len(graph[popped_n2]) != 0:
        #             for j in graph[popped_n2]:
        #                 print j
        #                 n2_q.put(j)


    def loop_detection(self, head):
        """
        given circular list, return node at the beginning of loop
        :param head: listnode
        :return: listnode
        """
        if not head:
            return
        curr = head
        runner = head
        while runner and runner.next:
            curr = curr.next
            runner = runner.next.next
            if curr is runner:
                # move slower back to head
                curr = head
                break
        while curr and runner:
            if curr is runner:
                print "start of loop here: ", curr
                return curr
            curr = curr.next
            runner = runner.next
        return False

    def intersection(self, n1, n2):
        """
        determine if 2 ll intersect, and return reference not by value
        :param n1: listnode
        :param n2: listnode
        :return: listnode
        """
        # or run through to not only get lengths but also tails to see if there is inter
        # rather than pad it then just traverse longer until same length

        if not n1 or not n2:
            return False
        dummy1 = n1
        dummy2 = n2
        n1_count, n2_count = 0, 0
        while dummy1.next:
            n1_count += 1
            dummy1 = dummy1.next
        while dummy2.next:
            n2_count += 1
            dummy2 = dummy2.next

        print dummy1, dummy2
        n1_count += 1
        n2_count += 1

        # dont bother since they dont intersect and have same end
        if dummy1 is not dummy2:
            return False

        if n1_count != n2_count:
            if n1_count > n2_count:
                big = n1
                small = n2
            else:
                big = n2
                small = n1

        for i in range(abs(n1_count - n2_count)):
            big = big.next

        while big and small:
            if big is small:
                print big
                return big
            big = big.next
            small = small.next

        # or
        #
        # if not n1 or not n2:
        #     return False
        #
        # dummy1 = n1
        # dummy2 = n2
        #
        # n1_count = 0
        # n2_count = 0
        # while dummy1:
        #     n1_count += 1
        #     dummy1 = dummy1.next
        # while dummy2:
        #     n2_count += 1
        #     dummy2 = dummy2.next
        #
        # if n1_count != n2_count:
        #     if n1_count > n2_count:
        #         big = n1
        #         small = n2
        #     else:
        #         big = n2
        #         small = n1
        #
        # print n1, n2
        # for i in range(abs(n1_count - n2_count)):
        #     pad = ListNode(0)
        #     pad.next = small
        #     small = pad
        #
        # while big and small:
        #     if big is small:
        #         print big
        #         return big
        #     big = big.next
        #     small = small.next

    def running_water(self, nums):
        """
        find the total number of water that can be contained 
        :param nums: list[int]
        :return: int
        """
        if not nums or len(nums) < 3:
            return 0
        print nums
        vol = 0

        left, right = 0, len(nums) - 1
        l_max, r_max = nums[left], nums[right]

        while left < right:
            # find max bars on the sides
            l_max, r_max = max(nums[left], l_max), max(nums[right], r_max)
            # always lmax
            if l_max <= r_max:
                vol += l_max - nums[left]
                left += 1
            else:
                vol += r_max - nums[right]
                right -= 1
        print vol
        return vol


    def palindrome_ll(self, head):
        """
        check if palindrome
        :param head: listnode
        :return: bool
        """
        if not head:
            return True
        curr = head
        runner = curr

        dummy = head

        while runner and runner.next:
            curr = curr.next
            runner = runner.next.next

        # reverse starting at the mid
        res = self.reverse_ll(curr)

        while res and dummy:
            if res.data != dummy.data:
                return False
            res = res.next
            dummy = dummy.next
        return True


    def reverse_ll(self, node, prev=None):
        if not node:
            return prev
        next = node.next
        node.next = prev
        return self.reverse_ll(next, node)

    def sum_list(self, n1, n2):
        """
        add 2 numbers rep by ll, stored in reverse order
        :param n1: listnode
        :param n2: listnode
        :return: listnode
        """
        if not n1 or not n2:
            return n1 or n2
        res = ListNode(sys.maxint)
        dummy = res
        carry = 0
        while n1 or n2:
            if not n1:
                n1 = ListNode(0)
            if not n2:
                n2 = ListNode(0)
            sum = n1.data + n2.data + carry
            # print sum
            if sum > 9:
                sum = sum % 10
                carry = 1
                new = ListNode(sum)
                res.next = new
                res = res.next
            elif sum < 10:
                new = ListNode(sum)
                res.next = new
                res = res.next
                carry = 0
            n1 = n1.next
            n2 = n2.next
        if carry == 1:
            print "still carry"
            new = ListNode(1)
            res.next = new
        print "***"
        while dummy.next:
            print dummy.next
            dummy = dummy.next


    def partition(self, head, part):
        """
        partition ll around partition, if equal can go on the greater side doesn
        need to be sorted
        :param head: listnode
        :return: listnode
        """
        if not head:
            return None
        if not part:
            return head

        curr = head
        ll_less = ListNode(sys.maxint)
        ll_big = ListNode(sys.maxint)

        temp2 = ll_less
        temp = ll_big

        while curr:
            if curr.data < part:
                ll_less.next = curr
                ll_less = ll_less.next
            else:
                ll_big.next = curr
                ll_big = ll_big.next
            curr = curr.next

        ll_less.next = temp.next

        return temp2.next


    def delete_node(self, node):
        """
        delete the node given only the node 
        :param node: listnode
        :return: none
        """
        if not node or not node.next:
            return
        node.data = node.next.data
        node.next = node.next.next

    def kth_to_last(self, head, k):
        """
        find kth to last node in ll
        :param head: listnode
        :param k: int
        :return: listnode
        """
        if not head:
            return None
        curr = head
        runner = curr
        for i in range(k):
            runner = runner.next
        while runner:
            curr = curr.next
            runner = runner.next
        print curr
        return curr

        # or recursively
        #
        # if not head:
        #     return 0
        # index = self.kth_to_last(head.next, k) + 1
        # if index == k:
        #     print head.data
        # return index

    def remove_dupll(self, head):
        """
        remove duplicates in unsorted ll without using extra space
        :param node: listnode
        :return: listnode
        """
        if not head:
            return None
        curr = head
        while curr:
            runner = curr
            while runner.next:
                if runner.next.data == curr.data:
                    runner.next = runner.next.next
                else:
                    runner = runner.next
            curr = curr.next
        return head

    def print_list(self, node):
        if not node:
            return None
        while node:
            print node
            node = node.next

class StackPlates(object):
    """
    implement a stack that has threshold 
    """
    def __init__(self):
        # 0 index
        self.threshold = 1
        self.curr_stack = 0
        self.stacks = [[]]

    def push(self, item):
        if len(self.stacks[self.curr_stack]) > self.threshold:
            new_stack = []
            self.stacks.append(new_stack)
            self.curr_stack += 1
        self.stacks[self.curr_stack].append(item)

    def pop(self):
        # what if keep popping and the curr stack is empty?
        if len(self.stacks[self.curr_stack]) != 0:
            if len(self.stacks[self.curr_stack]) == 1:
                del self.stacks[self.curr_stack]
                self.curr_stack -= 1
            return self.stacks[self.curr_stack].pop()
        else:
            del self.stacks[self.curr_stack]
            self.curr_stack -= 1
            return self.stacks[self.curr_stack].pop()

    def pop_at(self, index):
        # shouldnt be empty
        if len(self.stacks) != 0:
    #         0 index
            if index > len(self.stacks) or index < 0:
                print "out of bounds index"
                return False
            else:
                # can add a condition to check the entire stack
                return self.stacks[index].pop()

    def peek(self):
        if len(self.stacks[self.curr_stack]) != 0:
            return self.stacks[self.curr_stack][-1]
        print "empty!"

    def empty(self):
        return True if len(self.stacks[self.curr_stack]) == 0 else False

class MinStack(object):
    """
    implement min stack all ops are constant
    or use another stack to keep min
    """
    def __init__(self):
        self.curr_min = sys.maxint
        self.stack = []

    def push(self, item):
        self.curr_min = min(self.curr_min, item)
        item = (item, self.curr_min)
        self.stack.append(item)

    def pop(self):
        if len(self.stack) != 0:
            return self.stack.pop()

    def peek(self):
        if len(self.stack) != 0:
            return self.stack[-1]

    def min(self):
        if len(self.stack) != 0:
            latest = self.stack[-1]
            return latest[1]

    def empty(self):
        return True if len(self.stack) == 0 else False

    def print_s(self):
        print self.stack

class MyQueue(object):
    """
    implement a q using 2 stacks
    """
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def get(self):
        while self.s1:
            self.s2.append(self.s1.pop())
        if len(self.s2) != 0:
            res = self.s2.pop()
            self.s1, self.s2 = self.s2, self.s1
            return res
        print "empty"

    def put(self, item):
        self.s1.append(item)

    def empty(self):
        return True if len(self.s1) == 0 else False

class SortStack(object):
    """
    have a sorted stack
    """
    def __init__(self):
        self.stack = []
        self.temp_stack = []

    def push(self, item):
        if len(self.stack) == 0:
            self.stack.append(item)

        elif self.stack[-1] < item:
            while self.stack and self.stack[-1] < item:
                self.temp_stack.append(self.stack.pop())
            self.stack.append(item)

            while self.temp_stack:
                self.stack.append(self.temp_stack.pop())
        else:
            self.stack.append(item)

    def pop(self):
        return self.stack.pop()

    def peek(self):
        return self.stack[-1]

    def empty(self):
        return True if len(self.stack) == 0 else False

class AnimalNode(object):
    def __init__(self, type, name=None, order=-1, next=None):
        self.type = type
        self.name = name
        self.next = next
        self.order = order

    def __repr__(self):
        return self.type + " : " + self.name + " : " + str(self.order)

class AnimalShelter(object):
    """
    fifo, oldest of both dogs and cats or one of each
    
    did this as a stack to return most recent for q
    just return and remove the head
    
    for the dq then also search for the counter that is 0
    """
    def __init__(self):
        self.dog_q = ListNode("dog_dummy")
        self.cat_q = ListNode("cat_dummy")

        self.head_dog_slow = self.dog_q
        self.head_cat_slow = self.cat_q

        self.head_dog = self.dog_q
        self.head_cat = self.cat_q
        self.counter = 0

    def enq(self, animal):
        if animal.type == "dog":
            animal.order = self.counter
            self.counter += 1

            self.dog_q.next = animal
            self.dog_q = self.dog_q.next

        if animal.type == "cat":
            animal.order = self.counter
            self.counter += 1

            self.cat_q.next = animal
            self.cat_q = self.cat_q.next

    def dq(self):
        if self.cat_q.order == (self.counter - 1):
            print "Cat"
            self.dq_cat()
        if self.dog_q.order == (self.counter - 1):
            print "Dog"
            self.dq_dog()

    def dq_dog(self):
        if self.dog_q:
            while self.head_dog.next.next:
                self.head_dog = self.head_dog.next

            print self.head_dog.next
            self.head_dog.next = None
            self.head_dog = self.head_dog_slow
        else:
            print "empty"

    def dq_cat(self):
        if self.cat_q:
            while self.head_cat.next.next:
                self.head_cat = self.head_cat.next

            print self.head_cat.next
            self.head_cat.next = None
            self.head_cat = self.head_cat_slow
        else:
            print "empty"

    def printer(self, animal):
        while animal:
            print animal
            animal = animal.next

class BST(object):
    def __init__(self, root=None):
        self.root = root
        self.curr = self.root

    def __repr__(self):
        return self.root

    def insert(self, data):
        if not data:
            return
        self.insert_helper(self.root, data)

    def insert_helper(self, root, data):
        add_node = TreeNode(data)
        if not root:
            return add_node
        else:
            if root.data < data:
                if not root.right:
                    root.right = add_node
                else:
                    self.insert_helper(root.right, data)
            else:
                if not root.left:
                    root.left = add_node
                else:
                    self.insert_helper(root.left, data)
        # iterative
        # while root:
        #     p = root
        #     if root.data <= data:
        #         root = root.right
        #     else:
        #         root = root.left
        # if p.data <= data:
        #     p.right = add_node
        # else:
        #     p.left = add_node
        # return self.root

    def find(self, node):
        if not node or not self.root:
            return
        return self.find_helper(self.root, node)

    def find_helper(self, curr, node):
        if not curr or curr.data == node:
            return curr
        if curr.data < node:
            return self.find_helper(curr.right, node)
        if curr.data > node:
            return self.find_helper(curr.left, note)


    def min_val(self, node):
        curr = node
        while curr.left:
            curr = curr.left
        print curr
        return curr

    def delete(self, node):
        if not node:
            return False
        self.delete_helper(self.root, node)

    def delete_helper(self, root, key):
        # doesnt exist so return it
        if not root:
            return root
        if root.val > key:
            root.left = self.delete_helper(root.left, key)
        elif root.val < key:
            root.right = self.delete_helper(root.right, key)
        else:
    #       1 or no children
            if not root.left:
                return root.right
            if not root.right:
                return root.left
    #         with 2 child, get inorder successor
    #       leftmost of right subtree
            temp = self.min_val(root.right)
            root.val = temp.data
    #         delete successor
            root.right = self.delete_helper(root.right, temp.val)
        return root

if __name__ == '__main__':
    main()