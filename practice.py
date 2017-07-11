"""
practice problems of interview questions
"""

def palindrome_substring(input):
    """
    find all palindromes in substring
    :return: 
    """
    # try iterating and doing middle out, runtime around O(n^2)
    pass

def count_island(graph):
    """
    count number of islands rep in binary in 2d graph
    :return: 
    """
    # use bfs
    pass

def trap_water(arr):
    """
    find amount of water that can be trapped
    :return: 
    """
    # keep track of left and right maxs as you add from both sides
    pass

def scheduler(arr):
    """
    find the times when everyone is free
    :return: 
    """
    # merge all the lists and then use pointers to keep track
    pass

def skyline():
    pass

def moving_avg(arr, block):
    """
    find the average of block sizes
    :return: 
    """
    pass

def search_word_del(input, delim):
    """
    find the words given a list of delimeters
    :param input: 
    :param delim: 
    :return: 
    """
    pass

def pancake_sort():
    """
    use pancake sort
    :return: 
    """
    pass

def distributed_qs():
    """
    write a distributed quicksort
    :return: 
    """
    pass

def rpn_calc():
    """
    write rpn calc without dictionary
    :return: 
    """
    pass

def rotational_ndigits(n):
    """
    find all rotational numbers with n digits
    :return: list[int]
    """
    is_rsn("086980")

def is_rsn(input):
    if not input:
        print "empty"
        return False
    rsn = {'0': '0', '1': '1', '6': '9', '8': '8', '9': '6'}
    beg = 0
    end = len(input) - 1
    while beg <= end:
        print input[beg], input[end]
        if input[beg] not in rsn:
            print "False"
            return False
        else:
            if rsn[input[beg]] != input[end]:
                print "False"
                return False
        beg += 1
        end -= 1
    print "True"
    return True

rotational_ndigits(4)

