def factorial(num):
    if num <= 1:
        return 1
    return num * factorial(num - 1)


def factorial2(num, total=1):
    # tail recursion O(1) space complexity compare to regular O(n)
    if num == 0:
        return total
    else:
        # no need to wait result for computation, direct move to next function
        return factorial2(num-1, total*num)


def fibonacci(index):
    # 0, 1, 1, 2, 3, 5, 8, 13, 21, ...    O(2^n)
    if index <= 1:
        return index
    return fibonacci(index - 1) + fibonacci(index - 2)


def fibonacci2(index):
    # O(n)
    arr = [0, 1]
    for i in range(2, index+1):
        arr = arr + [(arr[i-1] + arr[i-2])]
    return arr[index]


if __name__ == '__main__':
    print(factorial(5))
    print(factorial2(5))
    print(fibonacci(7))
    print(fibonacci2(7))