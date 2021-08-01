class Array:
    arr = []

    def __init__(self, arr):
        if type(arr) is str:
            self.arr = [char for char in arr]
            self.flag = False
        elif type(arr) is list:
            self.arr = arr
            self.flag = True
        else:
            self.flag = False
            raise Exception("Illegal input parameter")

    @classmethod
    def get_class_name(cls):
        return cls.__name__

    def reverse_2pointer(self):
        left, right = 0, len(self.arr) - 1
        while left < right:
            self.arr[left], self.arr[right] = self.arr[right], self.arr[left]
            left, right = left + 1, right - 1
        if not self.flag:
            return ''.join(self.arr)
        return self.arr

    def reverse_recursive(self):
        def helper(left, right):
            if left < right:
                self.arr[left], self.arr[right] = self.arr[right], self.arr[left]
                helper(left + 1, right - 1)

        helper(0, len(self.arr) - 1)
        if not self.flag:
            return ''.join(self.arr)
        return self.arr

    def merge_sorted(self, *args):
        '''
            in place merging
            merging from right to left so can merge in place, otherwise change from second list
            will overwrite values in the first list
            last line if first list has remaining, don't need to do anything, since the
            modified first list will have same smallest value as first list
            only need to add second list remaining at the front of first list if any
        '''
        if len(args) == 1:
            p1, p2 = len(self.arr), len(args[0].arr)
            self.arr = self.arr + args[0].arr
            while p1 > 0 and p2 > 0:
                if self.arr[p1 - 1] > args[0].arr[p2 - 1]:
                    self.arr[p1 + p2 -1] = self.arr[p1 - 1]
                    p1 -= 1
                else:
                    self.arr[p1 + p2 -1] = args[0].arr[p2 - 1]
                    p2 -= 1
            self.arr[:p2] = args[0].arr[:p2]
        return self.arr


arr1 = Array(['a', 'b', 'c'])
print(arr1.get_class_name())
print(arr1.reverse_2pointer())
print(arr1.reverse_recursive())
print(arr1.arr[::-1])

arr2 = Array('Hello Harry')
print(arr2.reverse_recursive())
print(arr2.reverse_2pointer())
# arr3 = Array({})

arr3 = Array([1, 3, 5])
arr4 = Array([2, 4, 6, 7])
print(arr3.merge_sorted(arr4))
