import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from tqdm import tqdm

def H(Graph, color_map):
    E, cost = Graph.edges(), 0
    for (v, w) in E:
        if color_map[v] == color_map[w]:
            cost += 1
    return cost

def gen_map(colors, n):
    color_map = []
    for i in range(n):
        color_map.append(np.random.choice(colors))
    return np.array(color_map)

def color_picker(n):
    options = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    selection, opt_cop = [], options.copy()
    for i in range(n):
        c = np.random.choice(opt_cop)
        selection.append(c)
        opt_cop.remove(c)
    return selection

def new_color(colors, x):
    colors.remove(x)
    return np.random.choice(colors)

def highest_degree(Graph):
    temp, max = Graph.degree(), 0
    for (v,d) in temp:
        if d > max:
            max = d
    return max

def avg_degree(Graph):
    temp, total = Graph.degree(), 0
    for (v,d) in temp:
        total += d
    return total/Graph.order()

def anneal(Graph, map, colors):
    init_temp = 30
    final_temp = 0
    alpha = .1
    beta = 0.95

    current_temp = init_temp
    current_state = map.copy()
    solution = current_state.copy()

    while(current_temp > final_temp):
        vertex = np.random.randint(0, Graph.order()-1)    # pick random vertex

        orig_color = current_state[vertex]
        current_state[vertex] = new_color(colors.copy(), orig_color)    # pick dif color from orig and re-color

        cost = H(Graph, current_state) - H(Graph, solution)    # compute cost
        if cost <= 0:
            if np.random.uniform(0, 1) < np.exp(-cost/current_temp):
                solution = current_state.copy()
        elif cost > 0:
            if np.random.uniform(0, 1) < np.exp(-beta*cost):
                solution = current_state.copy()
        else:
            current_state[vertex] = orig_color
        current_temp -= alpha
    return solution


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
p = comm.Get_size()

# rank 0 creates graph object and generates a random color mappping
# then send both to the other processors, including color option
#

if rank == 0:
    n = 100 # number of nodes
    
    G = nx.random_degree_sequence_graph([3 for i in range(n)]) # degree sequence
    colors = color_picker(3)  # numb of possible colors (max 7)
    map = gen_map(colors, G.order())

    Graph = [G, map, colors] # graph object && color mapping
    data = [H(G,map)]
    best = map

    for dest in range(1, p):
        comm.send(Graph, dest=dest)
else:
    Graph = comm.recv(source=0)

#print(f"{rank} {Graph[1]}")

# Parallel
# 
# - gen color map
# - send to rank 0
# - rank 0 compute cost and find lowest
# - rank 0 sends lower map to other nodes
# - repeat

iterations = 100
 
for i in range(iterations):
    Graph[1] = anneal(Graph[0], Graph[1], Graph[2])

    if rank != 0:
        # send solution to rank 0
        comm.send(Graph[1], dest=0)
        Graph[1] = comm.recv(source=0)
    else:
        # compre rank 0 with best
        if H(Graph[0], Graph[1]) < H(Graph[0], best):
            best = Graph[1]

        # receive everyone's solution
        maps = [Graph[1]]
        for r in range(1, p):
            maps.append( comm.recv(source=r) )
        
        # find the lowest solution
        maps_costs = []
        min_index = 0
        for j in range(p):
            maps_costs.append( H(Graph[0], maps[j]) )
            if maps_costs[j] < maps_costs[min_index]: 
                min_index = j
        
        # send the lowest found is any, otherwise send best
        if H(Graph[0], maps[min_index]) < H(Graph[0], best): 
            Graph[1] = maps[min_index]
            data.append( H(Graph[0], Graph[1]) )
            for dest in range(1, p):
                comm.send(maps[min_index], dest=dest)
        else:
            Graph[1] = best
            data.append( H(Graph[0], Graph[1]) )
            for dest in range(1, p):
                comm.send(best, dest=dest)
        
#print(f"{rank} {Graph[1]} cost: {H(Graph[0], Graph[1])}")

if rank == 0:
    y, x = np.array(data), np.array( [i for i in range(iterations+1)] )
    plt.plot(x, y)
    plt.xlabel("Runs"), plt.ylabel("Cost")
    plt.title(f"{p} nodes     vertices: {G.order()}  deg: {avg_degree(G)}")
    plt.show()


MPI.Finalize