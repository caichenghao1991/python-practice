from typing import List


class Solution:
    # 746. Min Cost Climbing Stairs  top down /bottom up dynamic programming  t: O(n)   s: O(n)  or O(1)
    def minCostClimbingStairs(self, cost: List[int]) -> int:
        cost.append(0)
        """
        # bottom up
        prev, curr = cost[0], cost[1]
        for i in range(2, len(cost)):
            prev, curr = curr, min(curr, prev) + cost[i]
        return curr
        """
        # top down
        dp = [None] * (len(cost) + 1)
        return self.stairs(cost, len(cost) - 1, dp)

    def stairs(self, cost, index, dp):
        if index == 0 or index == 1:
            return cost[index]
        if dp[index]:
            return dp[index]
        else:
            dp[index] = min(self.stairs(cost, index - 2, dp), self.stairs(cost, index - 1, dp)) + cost[index]
            return dp[index]

    # 688. Knight Probability in Chessboard   top down /bottom up dynamic programming
    # t: O(kn^2)   s: O(kn^2)  or O(n^2)
    def knightProbability(self, n: int, k: int, row: int, column: int)-> float:
        """
        # top down
        dp = [[[None] * n for i in range(n)] for j in range(k+1)]
        return self.helper(n, k, row, column, dp)
        """
        # bottom up
        dp_prev, dp_curr = [[1] * n for _ in range(n)], [[0] * n for _ in range(n)]
        for x in range(k):
            for i in range(n):
                for j in range(n):
                    for c, d in [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]:
                        ro, co = i + d, j + c
                        if 0 <= ro < n and 0 <= co < n:
                            dp_curr[i][j] += dp_prev[ro][co] * 0.125
            dp_prev, dp_curr = dp_curr, [[0] * n for _ in range(n)]

        return dp_prev[row][column]

    def helper(self, n, k, row, column, dp):
        direction = [(1, -2), (2, -1), (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2)]
        if row < 0 or row >= n or column < 0 or column >= n:
            return 0
        if k == 0:
            return 1
        if dp[k][row][column]:
            return dp[k][row][column]
        total = 0
        for i, j in direction:
            total += self.helper(n, k - 1, row + j, column + i, dp)
        dp[k][row][column] = total * 0.125
        return dp[k][row][column]


if __name__ == '__main__':
    solution = Solution()
    print(solution.minCostClimbingStairs([10, 15, 20]))
    print(solution.knightProbability(3, 1, 0, 0))
