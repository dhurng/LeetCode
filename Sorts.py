"""
Popular Sorting Algorithms
"""


def main():
    sorter = Sorter()

    list = [3, 2, 5, 1, 4, 0]

    sorter.bubble_sort(list)


class Sorter(object):
    def __init__(self):
        pass

    def bubble_sort(self, nums):
        print nums


if __name__ == '__main__':
    main()