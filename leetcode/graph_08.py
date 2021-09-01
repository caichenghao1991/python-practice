from collections import deque
from typing import List
import heapq


class Solution:
    # 1376. Time Needed to Inform All Employees  dfs/ bfs
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        adj = [[] for _ in range(n)]
        for i in range(len(manager)):
            if manager[i] != -1:
                adj[manager[i]].append(i)
        # return self.dfs_num_min(headID, adj, informTime)

        q = deque()
        q.append((headID, informTime[headID]))
        total = 0
        while q:
            emp, t = q.popleft()
            total = max(total, t)
            for _ in adj[emp]:
                q.append((_, t + informTime[_]))
        return total

    def dfs_num_min(self, emp, adj, time):
        if not time[emp]:
            return 0
        else:
            longest = 0
            for i in adj[emp]:
                longest = max(longest, self.dfs_num_min(i, adj, time))
        return longest + time[emp]

    # 207. Course Schedule   topological sort (dfs/kahn's algorithm)
    def canFinish(self, numCourses, prerequisites):
        adj = [[] for _ in range(numCourses)]
        """
        degree = [0] * numCourses
        for i, j in prerequisites:
            adj[j].append(i)
            degree[i] += 1
        q = deque()
        for i in range(len(degree)):
            if not degree[i]: 
                q.append(i)
        counter = 0
        while q:         
            c = q.popleft()
            counter += 1
            for i in adj[c]:
                degree[i] -= 1
                if not degree[i]:
                    q.append(i)
        return counter == numCourses
        """
        for i, j in prerequisites:
            adj[j].append(i)
        visited = [0] * numCourses
        flag = []  # or use flag=[] self.flag = 0
        # can't use flag = 0, since integer is immutable, pass in function will give a new local duplicate copy

        for i in range(numCourses):
            if len(flag) > 0:
                break  # early stop if the loop is found
            if visited[i] == 0:
                self.dfs(i, adj, visited, flag)

        return len(flag) == 0

    def dfs(self, start, adj, visited, flag):
        if len(flag) > 0:  # self.flag == 1
            return  # early stop if the loop is found
        if visited[start] == 1:
            flag.append("1")  # loop is found
            return
        if visited[start] == 0:  # node is not visited yet, visit it
            visited[start] = 1  # color current node as gray
            for i in adj[start]:  # visit all its neighbours
                self.dfs(i, adj, visited, flag)
            visited[start] = 2  # no loop start from this node

    @staticmethod
    # 743. Network Delay Time   Dijkstra  / Bellman Ford
    def networkDelayTime(times: List[List[int]], n: int, k: int) -> int:
        # Dijkstra
        adj = [{} for _ in range(n)]
        for i, j, t in times:
            adj[i - 1][j - 1] = t
        distance = [float("Inf")] * n
        distance[k - 1] = 0
        seen = set()
        q = [(0, k - 1)]
        while q:
            d, n = heapq.heappop(q)
            if n not in seen:
                seen.add(n)
            for i in adj[n]:
                if i not in seen and d + adj[n][i] < distance[i]:
                    distance[i] = d + adj[n][i]
                    heapq.heappush(q, (distance[i], i))

        print(distance)

        """
        # Bellman Ford
        distance = [float("Inf")] * n
        distance[k - 1] = 0
        for k in range(n):
            for i, j, t in times:
                if distance[j - 1] > distance[i - 1] + t:
                    distance[j - 1] = distance[i - 1] + t
        """
        return -1 if max(distance) == float("Inf") else max(distance)


if __name__ == '__main__':
    solution = Solution()
    print(solution.numOfMinutes(6, 2, [2, 2, -1, 2, 2, 2], [0, 0, 1, 0, 0, 0]))
    print(solution.canFinish(2, [[1, 0], [0, 1]]))
    print(Solution.networkDelayTime([[1, 2, 1], [2, 3, 2], [1, 3, 4]], 3, 1))
