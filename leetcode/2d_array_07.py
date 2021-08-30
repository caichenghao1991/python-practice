from collections import deque
from typing import List


class Solution:
    # 200. Number of Islands   bfs / dfs
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0
        res = 0
        queue = deque([])
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    # self.dfs_num_islands(grid, i , j)
                    queue.append((i, j))
                    grid[i][j] = "0"
                    while queue:
                        x = queue.popleft()
                        for d in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                            m, n = x[0] + d[0], x[1] + d[1]
                            if 0 <= m < len(grid) and 0 <= n < len(grid[0]) and grid[m][n] == "1":
                                queue.append((m, n))
                                grid[m][n] = "0"
                    res += 1
        return res

    def dfs_num_islands(self, grid2, y, x):
        direction = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        if y < 0 or x < 0 or y >= len(grid2) or x >= len(grid2[0]) or grid2[y][x] != '1':
            return
        grid2[y][x] = "0"
        for d in direction:
            self.dfs_num_islands(grid2, y + d[0], x + d[1])

    @staticmethod
    # 994. Rotting Oranges   bfs
    def orangesRotting(grid: List[List[int]]) -> int:
        good = 0

        rotten = deque()

        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 2:
                    rotten.append((i, j))  # rotten.append((i, j, steps))
                if grid[i][j] == 1:
                    good += 1
        """
        steps = 0
        c = 0
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        while rotten:
            x, y, steps = rotten.popleft()
            for dx, dy in dirs:
                if 0 <= x + dx < len(grid) and 0 <= y + dy < len(grid[0]) and grid[x + dx][y + dy] == 1:
                    rotten.append((x + dx, y + dy, steps + 1))
                    grid[x + dx][y + dy] = 2
                    c += 1
        return -1 if c < good else steps
        """
        steps = -1
        c1, c2, c3 = len(rotten), 0, 0
        if good == 0:
            return 0
        while rotten:
            x = rotten.popleft()
            c1 -= 1
            for d in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                m, n = x[0] + d[0], x[1] + d[1]
                if 0 <= m < len(grid) and 0 <= n < len(grid[0]) and grid[m][n] == 1:
                    rotten.append((m, n))
                    grid[m][n] = 2
                    c2 += 1
                    c3 += 1
            if c1 == 0:
                c1, c2 = c2, 0
                steps = steps + 1
        return -1 if c3 < good else steps

    # 663. Walls and Gates    dfs / bfs
    def wallsAndGates(self, rooms):
        door = []
        for i in range(len(rooms)):
            for j in range(len(rooms[0])):
                if rooms[i][j] == 0:
                    door.append((i, j))
        for y, x in door:
            self.dfs(rooms, y, x, 1)

    def dfs(self, rooms, y, x, steps):
        for dy, dx in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            m, n = y + dy, x + dx
            if 0 <= m < len(rooms) and 0 <= n < len(rooms[0]) and rooms[m][n] != -1 and 0 != rooms[m][n] and steps <\
                    rooms[m][n]:
                rooms[m][n] = steps
                self.dfs(rooms, m, n, steps + 1)


if __name__ == "__main__":
    solution = Solution()
    g = [["1", "1", "1", "1", "0"], ["1", "1", "0", "1", "0"], ["1", "1", "0", "0", "0"], ["0", "0", "0", "0", "0"]]
    print(solution.numIslands(g))
    print(Solution.orangesRotting([[2, 1, 1], [1, 1, 0], [0, 1, 1]]))
    room = [[2147483647, -1, 0, 2147483647], [2147483647, 2147483647, 2147483647, -1], [2147483647, -1, 2147483647, -1],
            [0, -1, 2147483647, 2147483647]]
    solution.wallsAndGates(room)
    print(room)
