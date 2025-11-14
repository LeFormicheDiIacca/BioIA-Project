

def cost_assignment(graph, edges_metadata, assignment_function, print_assignment=False):
    for edge in graph.edges():
        #If no metadata is available, a big value is assigned as cost
        try:
            metadata = edges_metadata[edge]
            cost = assignment_function(metadata)
        except KeyError:
            cost = 1000000
            metadata = None
            print(f"Error: edge {edge} has no metadata. Cost set to {cost}.")
        graph[edge[0]][edge[1]]['cost'] = cost
        if print_assignment:
            print(f"Assignment for edge {edge[0]}->{edge[1]} cost: {cost}")
            print(f"Metadata of edge {edge[0]}->{edge[1]}:\n{metadata}")

#The function that we should evolve in the future
#We can evolve various functions depending on the thing that the ants need to plan (road, railways, optic fiber lines, etc)
def test_cost_assignment(edge_metadata):
    cost = 0
    for (k,v) in edge_metadata.items():
        cost = cost + v
    return cost