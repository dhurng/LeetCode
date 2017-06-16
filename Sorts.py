"""
Popular Sorting Algorithms
"""
import time

def main():
    sorter = Sorter()

    start = time.time()

    list = [3, 2, 5, 1, 4, 0]

class Sorter(object):
    def __init__(self):
        pass

    def merge_sort(self, nums):
    #     basic merge sort
        self.merge_sort_helper(nums, 0, len(nums) - 1)

    def merge_sort_helper(self, nums, beg, end):
        if beg < end:
            mid = (beg + end) // 2
            self.merge_sort_helper(nums, beg, mid)
            self.merge_sort_helper(nums, mid + 1, end)
            self.merge_all(nums, beg, mid, end)

    def merge_all(self, nums, beg, mid, end):
        left = nums[beg:mid]
        right = nums[mid:end + 1]
        # ending indicator
        left.append(999999)
        right.append(999999)

        i = j = 0
        for k in range(beg, end + 1):
            if left[i] <= right[j]:
                nums[k] = left[i]
                i += 1
            else:
                nums[k] = right[j]
                j += 1


    def quick_sort(self, nums):
    #     most of the times middle works
    #     median of 3 values
        self.quick_sort_help(nums, 0, len(nums) - 1)

    def quick_sort_help(self, beg, end):
        if beg < end:
            p = self.partition(nums, beg, end)
            self.quick_sort_help(nums, beg, p - 1)
            self.quick_sort_help(nums, p + 1, end)

    def get_pivot(self, nums, beg, end):
        mid = (beg + mid) // 2
        pivot = end
        if nums[beg] < nums[mid]:
            if nums[mid] < nums[end]:
                pivot = mid
        elif nums[beg] < nums[end]:
            pivot = beg
        return pivot

    def partition(self, nums, beg, end):
        pivotIndex = self.get_pivot(nums, beg, end)
        pivotVal = nums[pivotIndex]
        nums[pivotIndex], nums[beg] = nums[beg], nums[pivotIndex]
        border = beg

        for i in range(beg, end + 1):
            if nums[beg] < pivotVal:
                nums[beg], nums[border] = nums[border], nums[beg]
        nums[beg], nums[border] = nums[border], nums[beg]

        return border

    def bubble_sort(self, nums):
        for i in range(0, len(nums) - 1):
            for j in range(0, len(nums) - 1 - i):
                if nums[j] > nums[j + 1]:
                    nums[j], nums[j + 1] = nums[j + 1], nums[j]
        print nums

    def selection_sort(self, nums):
        # second to last
        for i in range(0, len(nums) - 1):
            minIndex = i
            for j in range(i + 1, len(nums)):
                if nums[j] < nums[minIndex]:
                    minIndex = j
            if minIndex != i:
                nums[i], nums[minIndex] = nums[minIndex], nums[i]

        print nums


if __name__ == '__main__':
    main()