


#The function that we should evolve in the future
#We can evolve various functions depending on the thing that the ants need to plan (road, railways, optic fiber lines, etc)
def test_cost_assignment(edge_metadata):
    cost = 0
    for (k,v) in edge_metadata.items():
        cost = cost + v
    return cost