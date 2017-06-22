import sys
def main():
    test = Solution()

    a = ListNode("a")
    b = ListNode("b")
    c = ListNode("c")
    d = ListNode("d")
    e = ListNode("e")
    f = ListNode("f")

    a.next = b
    b.next = c
    c.next = d
    d.next = e
    e.next = c

    test.loop_detection(a)

class ListNode(object):
    def __init__(self, data, next=None):
        self.data = data
        self.next = next

    def __repr__(self):
        return str(self.data)

class Solution(object):
    def __init__(self):
        pass

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

if __name__ == '__main__':
    main()