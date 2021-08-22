def fib(num: int):
    # use closure to avoid global variable
    # 0  1  1  2  3  5  8  13
    res = [None] * (num + 1)

    def helper(index: int):
        if res[index]:
            return res[index]
        else:
            if index < 2:
                return index
            else:
                return helper(index - 1) + helper(index - 2)

    return helper(num)


def fib2(num: int):
    prev, curr = 0, 1
    index = 2
    while index <= num:
        prev, curr = curr, prev + curr
        index += 1
    return curr


# 9820255076106227

if __name__ == '__main__':
    print(fib(7))
    print(fib2(7))

