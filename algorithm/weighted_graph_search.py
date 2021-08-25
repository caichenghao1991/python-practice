import queue as q


class Graph:

    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    @staticmethod
    def find_min_vertex(dist, queue):
        minimum = float("Inf")
        index = -1
        for i in range(len(dist)):
            if dist[i] < minimum and i in queue:
                minimum = dist[i]
                index = i
        return index

    def print_path(self, parent, i):
        if parent[i] == -1:
            print(str(i) + "->", end="")
            return
        self.print_path(parent, parent[i])
        print(str(i) + "->", end="")

    def dijkstra(self, src):
        """
            time complexity: O(V^2) with priority queue O(V + E logV ) .

        """
        dist = [float("Inf")] * self.vertices
        parent = [-1] * self.vertices
        dist[0] = 0
        queue = [i for i in range(self.vertices)]
        ct = 0  # in case vertices can't be reach via any edges
        while queue and ct < self.vertices:
            curr = self.find_min_vertex(dist, queue)
            queue.remove(curr)

            for next_node in range(self.vertices):
                if self.graph[curr][next_node] and next_node in queue:
                    if dist[curr] + self.graph[curr][next_node] < dist[next_node]:
                        dist[next_node] = dist[curr] + self.graph[curr][next_node]
                        parent[next_node] = curr
            ct += 1

        for i in range(1, len(dist)):
            print("\n%d --> %d \t\t%d \t\t\t\t\t" % (src, i, dist[i]))
            self.print_path(parent, i)
        return dist, parent

    def dijkstra_pq(self, src):
        dist_pq = q.PriorityQueue()
        dist_pq.put((0, 0))
        parent = [-1] * self.vertices
        visited = set()
        dist = [float("Inf")] * self.vertices
        dist[0] = 0

        while not dist_pq.empty():
            _, curr = dist_pq.get()
            visited.add(curr)

            for next_node in range(self.vertices):
                if self.graph[curr][next_node] and next_node not in visited:
                    if dist[curr] + self.graph[curr][next_node] < dist[next_node]:
                        dist[next_node] = dist[curr] + self.graph[curr][next_node]
                        parent[next_node] = curr
                        dist_pq.put((dist[next_node], next_node))

        for i in range(1, len(dist)):
            print("\n%d --> %d \t\t%d \t\t\t\t\t" % (src, i, dist[i]))
            self.print_path(parent, i)
        return dist, parent


if __name__ == '__main__':
    graph = Graph(9)
    graph.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
                   [4, 0, 8, 0, 0, 0, 0, 11, 0],
                   [0, 8, 0, 7, 0, 4, 0, 0, 2],
                   [0, 0, 7, 0, 9, 14, 0, 0, 0],
                   [0, 0, 0, 9, 0, 10, 0, 0, 0],
                   [0, 0, 4, 14, 10, 0, 2, 0, 0],
                   [0, 0, 0, 0, 0, 2, 0, 1, 6],
                   [8, 11, 0, 0, 0, 0, 1, 0, 7],
                   [0, 0, 2, 0, 0, 0, 6, 7, 0]]

    print(graph.dijkstra(0)[0])
    print(graph.dijkstra_pq(0)[0])  # [0, 4, 12, 19, 21, 11, 9, 8, 14]
