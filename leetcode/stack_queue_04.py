class Solution:

    # 20. Valid Parentheses  stack
    @staticmethod
    def isValid(s: str) -> bool:
        start = ['(', '{', '[']
        close = [')', '}', ']']
        stack = []
        for c in s:
            if c in start:
                stack.append(c)
            else:
                if len(stack) != 0 and start.index(stack[-1]) == close.index(c):
                    stack.pop()
                else:
                    return False
        return len(stack) == 0

    # 1249. Minimum Remove to Make Valid Parentheses   stack
    @staticmethod
    def minRemoveToMakeValid(s: str) -> str:
        arr = list(s)
        stack = []
        for i in range(len(arr)):
            if arr[i] == ')':
                if not len(stack):
                    arr[i] = ""
                else:
                    stack.pop()
            elif arr[i] == '(':
                stack.append(i)
        for i in stack:
            arr[i] = ""
        return "".join(arr)


if __name__ == '__main__':
    solution = Solution()
    print(Solution.isValid("({}[])"))
    print(Solution.minRemoveToMakeValid(")I(love y)ou)"))
