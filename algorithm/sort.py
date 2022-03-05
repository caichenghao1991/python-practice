import data_structure.heap_priority_queue_09 as h


def bubble_sort(array):
    """
        O(n^2) time complexity, best Omega(n), O(1) space complexity
        from left to right compare adjacent pairs and switch if in wrong order,
        after one iteration the largest/smallest value is at the right end,
        then repeat each time one more value at right end is sorted
    """
    for i in range(len(array)):
        for j in range(0, len(array) - i - 1):
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
    return array


def selection_sort(array):
    """
        O(n^2) time complexity, best Omega(n^2), O(1) space complexity
        from left to right find the smallest value among all the value,
        then switch with the left end value, after one iteration the
        smallest value is at the left end, then repeat each time one
        more value at left end is sorted
    """
    for i in range(0, len(array)):
        index = i
        for j in range(i + 1, len(array)):
            if array[j] < array[index]:
                index = j
        arr[i], arr[index] = arr[index], arr[i]
    return array


def insertion_sort(array):
    """
        O(n^2) time complexity, best case O(n), O(1) space complexity
        from left to right increasing the left sorted size by one, consider
        the item i beside the sorted part, pointer from right to left, if number greater than
        than i, shift to the right one step, finally place i at pointer location
        good for almost sorted small data
    """
    for i in range(1, len(array)):
        val = array[i]
        j = i - 1
        while j >= 0 and array[j] > val:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = val
    return array



def merge(a, b):
    c = []
    h, j = 0, 0
    while j < len(a) and h < len(b):
        if a[j] < b[h]:
            c.append(a[j])
            j += 1
        else:
            c.append(b[h])
            h += 1

    if j == len(a):
        for i in b[h:]:
            c.append(i)
    else:
        for i in a[j:]:
            c.append(i)

    return c

def merge_sort(lists):
    if len(lists) <= 1:
        return lists
    middle = len(lists) / 2
    left = merge_sort(lists[:middle])
    right = merge_sort(lists[middle:])
    return merge(left, right)


def quick_sort(array):
    """
        O(n logn) time complexity, worst case O(n^2), O(logn) space complexity,

        most popular, worst case O(n^2), pivot need random. O(logn) space
        """
    quick_sort_helper(array, 0, len(array) - 1)


def quick_sort_helper(array, left, right):
    if left < right:  # check index - 1 > left
        index = partition(array, left, right)
        quick_sort_helper(array, left, index - 1)
        quick_sort_helper(array, index + 1, right)


def partition(array, left, right):
    pivot = array[right]
    pointer = left
    for i in range(left, right):
        if array[i] <= pivot:
            array[i], array[pointer] = array[pointer], array[i]
            pointer += 1
    array[pointer], array[right] = array[right], array[pointer],
    return pointer


def bucket_sort(array):
    """
        O(n^2) average theta(n+k) time complexity, O(n) space complexity
        create empty buckets, insert value belong to those bucket range
        sort values inside bucket, concat all bucket values
    """
    min_val = min(array)
    max_val = max(array)
    num_bucket = 10
    rang = (max_val - min_val) / num_bucket
    bucket = []
    for i in range(num_bucket):
        bucket.append([])

    for i in range(len(array)):
        # append boundary to lower bucket
        diff = (array[i] - min_val) / rang - int((array[i] - min_val) / rang)
        if diff == 0 and array[i] != min_val:
            bucket[int((array[i] - min_val) / rang) - 1].append(array[i])
        else:
            bucket[int((array[i] - min_val) / rang)].append(array[i])

    for i in range(num_bucket):
        if len(bucket[i]) != 0:
            insertion_sort(bucket[i])

    j = 0
    for lst in bucket:
        if lst:
            for i in lst:
                array[j] = i
                j += 1
    return array


def counting_sort(array, rang):
    """
        O(n+k) time complexity, O(k) space complexity
        count each ordered unique values occurrence times, then sum the previous
        occurrence times to get the index of each unique values' ending index of the
        result array. loop though the original array, find the corresponding end index of
        value i for that value in the original array v, then fill the output array with v
        at index i, minus one for v 's occurrence times
    """
    # array: 1 0 3 1 3 1
    count = [0] * rang
    output = [0 for _ in range(len(array))]
    # count each digit occurrence
    # 1 3  0  2
    for i in array:
        count[i] += 1
    # get the ending index of each value
    # 1  4  4  6
    for i in range(1, len(count)):
        count[i] = count[i] + count[i - 1]
    # shift one to the right  is the corresponding starting index of each value
    for i in range(len(array)-1, -1, -1):
        # for each value at index i, this value occurrence in the output should be at
        # the index of corresponding counter list value - 1
        output[count[arr[i]] - 1] = arr[i]
        count[arr[i]] -= 1  # minus one for that counter value

    return output







def radix_sort(s):
    """
        O(nk) time complexity, O(n+k) space complexity
        use counting sort in each digit of numbers from least significant to
        most significant and sort is stable (inside counting sort start from
        right of the array which is the biggest at right most)

    """
    i = 0  # 记录当前正在排拿一位，最低位为1
    max_num = max(s)  # 最大值
    j = len(str(max_num))  # 记录最大值的位数
    while i < j:
        bucket_list = [[] for _ in range(10)]  # 初始化桶数组
        for x in s:
            bucket_list[int(x / (10 ** i)) % 10].append(x)  # 找到位置放入桶数组
        print(bucket_list)
        s.clear()
        for x in bucket_list:  # 放回原序列
            for y in x:
                s.append(y)
        i += 1


def min_heapify(pos, array):
    left = pos * 2 + 1
    right = pos * 2 + 2
    length = len(array)
    small = pos
    if left < length and array[small] > array[left]:
        small = left
    if right < length and array[small] > array[right]:
        small = right
    if small != pos:
        array[small], array[pos] = array[pos], array[small]
        min_heapify(small, array)
    return array


def heap_sort(array):
    heap = h.MinHeap()
    array = heap.build_heap(array)
    for i in range(len(array)):
        array[i:] = min_heapify(0, array[i:])
        min_heapify(0, array[i:])
    return array


if __name__ == '__main__':
    arr = [1, 5, 3, 9, 2, 6, 8, 7, 4]
    bubble_sort(arr)
    print(arr)
    arr = [1, 5, 3, 9, 2, 6, 8, 7, 4]
    selection_sort(arr)
    print(arr)
    arr = [1, 5, 3, 9, 2, 6, 8, 7, 4]
    insertion_sort(arr)
    print(arr)
    arr = [1, 5, 3, 9, 2, 6, 8, 7, 4]
    merge_sort(arr)
    print(arr)
    arr = [1, 5, 3, 9, 2, 6, 8, 7, 4]
    quick_sort(arr)
    print(arr)
    arr = [1, 5, 3, 9, 2, 6, 8, 7, 4, 1.6, 2.3]
    bucket_sort(arr)
    print(arr)
    arr = [1, 0, 3, 1, 3, 1]
    print(counting_sort(arr, 4))
    arr = [170, 45, 75, 90, 802, 24, 2, 66]
    radix_sort(arr)
    print(arr)
    arr = [170, 45, 75, 90, 802, 24, 2, 66]
    print(heap_sort(arr))

