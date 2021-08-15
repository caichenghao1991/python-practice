class Graph:
    def __init__(self):
        self.num_nodes = 0
        self.adjacentList = {}

    def add_vertex(self, node):
        if node not in self.adjacentList:
            self.adjacentList[node] = []
            self.num_nodes += 1

    def add_edge(self, node1, node2):
        if node1 not in self.adjacentList:
            self.add_vertex(node1)
        if node2 not in self.adjacentList:
            self.add_vertex(node2)
        self.adjacentList[node1].append(node2)
        self.adjacentList[node2].append(node1)

    def print_graph(self):
        for k in self.adjacentList.keys():
            values = k + ":"
            for v in self.adjacentList.get(k):
                values = values + v + '-'
            print(values[:-1])


if __name__ == '__main__':
    graph = Graph()
    graph.add_vertex('0')
    graph.add_vertex('1')
    graph.add_vertex('2')
    graph.add_vertex('3')
    graph.add_vertex('4')
    graph.add_vertex('5')
    graph.add_vertex('6')
    graph.add_edge('3', '1')
    graph.add_edge('3', '4')
    graph.add_edge('4', '2')
    graph.add_edge('4', '5')
    graph.add_edge('1', '2')
    graph.add_edge('1', '0')
    graph.add_edge('0', '2')
    graph.add_edge('6', '5')
    graph.print_graph()
