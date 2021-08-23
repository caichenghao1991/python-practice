class Graph:

    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [[0 for _ in range(vertices)] for _ in range(vertices)]

    def print_path(self, parent, i):
        if parent[i] == -1:
            print(str(i) + "->", end="")
            return
        self.print_path(parent, parent[i])
        print(str(i) + "->", end="")

    def dijkstra(self, src):
        dist = [float("Inf")] * self.vertices
        parent = [-1] * self.vertices
        dist[0] = 0
        queue = [i for i in range(self.vertices)]

        while queue:
            minimum = float("Inf")
            index = -1
            for i in range(len(dist)):
                if dist[i] < minimum and i in queue:
                    minimum = dist[i]
                    index = i
            curr = index

            queue.remove(curr)

            for next_node in range(self.vertices):
                if self.graph[curr][next_node] and next_node in queue:
                    if dist[curr] + self.graph[curr][next_node] < dist[next_node]:
                        dist[next_node] = dist[curr] + self.graph[curr][next_node]
                        parent[next_node] = curr

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
