class Solution(object):
    # 844. Backspace String Compare   two pointer
    @staticmethod
    def backspaceCompare(s: str, t: str) -> bool:
        p1, p2 = len(s), len(t)
        while p1 >= 0 and p2 >= 0:
            back = 1
            while back > 0:
                p1 -= 1
                if p1 >= 0 and s[p1] == '#':
                    back += 1
                else:
                    back -= 1
            back = 1
            while back > 0:
                p2 -= 1
                if p2 >= 0 and t[p2] == '#':
                    back += 1
                else:
                    back -= 1
            if p1 >= 0 and p2 >= 0 and s[p1] != t[p2]:
                return False

        return p1 < 0 and p2 < 0

    # 3. Longest Substring Without Repeating Characters    sliding window + hash
    @staticmethod
    def lengthOfLongestSubstring(s: str) -> int:
        dic = {}
        res, start = 0, 0
        for i in range(len(s)):
            if s[i] in dic and start <= dic[s[i]]:
                start = dic[s[i]] + 1
            else:
                res = max(res, i - start + 1)
            dic[s[i]] = i
        return res

    # 680. Valid Palindrome II     two pointer  iterative
    @staticmethod
    def validPalindrome(s: str) -> bool:
        left, right = 0, len(s) - 1
        flag = True
        while left < right:
            if s[left] != s[right]:
                l1 = left + 1
                r1 = right
                while l1 < r1:
                    if s[l1] != s[r1]:
                        flag = False
                        break
                    l1 += 1
                    r1 -= 1
                if not flag:
                    r2 = right - 1
                    l2 = left
                    while l2 < r2:
                        if s[l2] != s[r2]:
                            return False
                        l2 += 1
                        r2 -= 1
                return True
            left += 1
            right -= 1
        return True

    @staticmethod   # recursive
    def validPalindrome2(s: str) -> bool:
        def helper(s2, left: int, right: int, flag: bool) -> bool:
            while left < right:
                if s2[left] != s2[right]:
                    if not flag:
                        return helper(s2, left + 1, right, True) or helper(s2, left, right - 1, True)
                    else:
                        return False
                else:
                    left += 1
                    right -= 1
            return True

        return helper(s, 0, len(s) - 1, False)


if __name__ == '__main__':
    print(Solution.backspaceCompare('ab#c', 'ad#c'))
    print(Solution.lengthOfLongestSubstring("abcabcbb"))
    print(Solution.validPalindrome("abca"))
