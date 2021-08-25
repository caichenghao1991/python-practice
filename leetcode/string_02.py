class Solution(object):
    # 844. Backspace String Compare   two pointer
    def backspaceCompare(self, s: str, t: str) -> bool:
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

    #3. Longest Substring Without Repeating Characters    sliding window + hash
    def lengthOfLongestSubstring(self, s: str) -> int:
        dic = {}
        res, start = 0, 0
        for i in range(len(s)):
            if s[i] in dic and start <= dic[s[i]]:
                start = dic[s[i]] + 1
            else:
                res = max(res, i - start + 1)
            dic[s[i]] = i
        return res

    # 680. Valid Palindrome II
    def validPalindrome(self, s:str) -> bool:
        l, r = 0, len(s) - 1
        flag = True
        while l < r:
            if s[l] != s[r]:
                l1 = l + 1
                r1 = r
                while l1 < r1:
                    if s[l1] != s[r1]:
                        flag = False
                        break
                    l1 += 1
                    r1 -= 1
                if not flag:
                    r2 = r - 1
                    l2 = l
                    while l2 < r2:
                        if s[l2] != s[r2]:
                            return False
                        l2 += 1
                        r2 -= 1
                return True
            l += 1
            r -= 1
        return True

    def validPalindrome2(self, s:str) -> bool:
        def helper(s, l, r, flag):
            while l < r:
                if s[l] != s[r]:
                    if not flag:
                        return helper(s, l + 1, r, True) or helper(s, l, r - 1, True)
                    else:
                        return False
                else:
                    l += 1
                    r -= 1
            return True

        return helper(s, 0, len(s) - 1, False)

if __name__ == '__main__':
    solution = Solution()
    print(solution.backspaceCompare('ab#c', 'ad#c'))
    print(solution.lengthOfLongestSubstring("abcabcbb"))
    print(solution.validPalindrome("abca"))