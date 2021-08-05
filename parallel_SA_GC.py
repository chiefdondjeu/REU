import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
import sys

####
## Functions
####

# cost function
def H(Graph, color_coloring):
    cost = 0
    for (v, w) in Graph.edges():
        if color_coloring[v] == color_coloring[w]:
            cost += 1
    return cost

# generates a random color coloring
def gen_coloring(colors, n):
    return np.array( [np.random.choice(colors) for i in range(n)] )

# returns n number of colors
def color_picker(n):
    options = np.array(['b', 'g', 'r', 'c', 'm', 'y', 'k'])
    return np.array( [options[i] for i in np.random.choice(options.size, n, replace=False)] )

# removes color x from list of colors and return another color other than x
def new_color(colors, x):
    return np.random.choice( np.setdiff1d(colors, x) )

def highest_degree(Graph):
    return np.amax( np.array(Graph.degree())[:,1] )

def avg_degree(Graph):
    return np.average( np.array(Graph.degree())[:,1] )

def anneal(Graph, coloring, colors):
    init_temp = 50
    final_temp = 0
    alpha = .1
    beta = 0.95

    current_temp = init_temp
    current_state = coloring.copy()
    solution = current_state.copy()

    while(current_temp > final_temp):
        # pick random vertex
        vertex = np.random.randint(0, Graph.order()-1)
        
        # choose dif color from orig and re-color
        orig_color = current_state[vertex]
        current_state[vertex] = new_color(colors, orig_color)
        
        # compute cost
        cost = H(Graph, current_state) - H(Graph, solution)
        if cost <= 0:
            if np.random.uniform(0, 1) < np.exp(-cost/current_temp):
                solution = current_state.copy()
        elif cost > 0:
            if np.random.uniform(0, 1) < np.exp(-beta*cost):
                solution = current_state.copy()
        else:
            current_state[vertex] = orig_color
        # decrease temp
        current_temp -= alpha
        
    return solution

####
## Setup our parallel world
####

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
p = comm.Get_size()

# rank 0 creates networkx graph object and generates a random color coloringpping
# then send both to the other processors, including color option
#

if rank == 0:
    n = 100 # number of nodes
    deg = 3
    nCol = 3 # number of colors
    
    G = nx.random_degree_sequence_graph([deg for i in range(n)]) # degree sequence
    #G = nx.gnm_random_graph(n, n)
    
    colors = color_picker(nCol)  # numb of possible colors (max 7)
    
    coloring = gen_coloring(colors, G.order())
    coloringS = coloring.copy() # for serial

    Graph = [G, coloring, colors] # main data
    data = np.array([H(G,coloring)]) # costs of each iterations
    best = coloring

    print(f"\nparallel...", end=" ")
    sys.stdout.flush()

else:
    Graph = None
Graph = comm.bcast(Graph, root=0)
#print(f"{rank} {Graph[1]}")


# Parallel
# 
# - all gen a solution
# - send to rank 0
# - rank 0 compute cost and find lowest
# - rank 0 sends lowest cost coloring to other nodes
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
        colorings = [Graph[1]]
        for r in range(1, p):
            colorings.append( comm.recv(source=r) )
        
        # find the lowest solution
        colorings_costs = []
        min_index = 0
        for j in range(p):
            colorings_costs.append( H(Graph[0], colorings[j]) )
            if colorings_costs[j] < colorings_costs[min_index]: 
                min_index = j
        
        # send the lowest found is any, otherwise send best
        if H(Graph[0], colorings[min_index]) < H(Graph[0], best): 
            Graph[1] = colorings[min_index]
            data = np.append(data, H(Graph[0], Graph[1]) )
            for dest in range(1, p):
                comm.send(colorings[min_index], dest=dest)
        else:
            Graph[1] = best
            data = np.append(data, H(Graph[0], Graph[1]) )
            for dest in range(1, p):
                comm.send(best, dest=dest)
        
#print(f"{rank} {Graph[1]} cost: {H(Graph[0], Graph[1])}")

if rank == 0:
    print(f"done\n\nserial...", end=" ")
    sys.stdout.flush()

    ####
    ## Serial
    ####

    dataS = np.array([H(G,coloringS)])

    for i in range(iterations):
        sol = anneal(G, coloringS, colors)
        sol_cost = H(G, sol)
        if sol_cost < H(G,coloringS):
            coloringS = sol
        dataS = np.append(dataS, H(G,coloringS))

    print(f"...done")
    sys.stdout.flush()

    ####
    ## plots
    ####

    X = np.arange(iterations+1)
    
    plt.plot(X, data, 'g', label=f"{p} nodes")
    plt.plot(X, dataS, 'r', label="serial")

    plt.title(f"Parallel Vs. Serial SA")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()

    plt.show()

    # y, x = np.array(data), np.array( [i for i in range(iterations+1)] )
    # plt.plot(x, y)
    # plt.xlabel("Runs"), plt.ylabel("Cost")
    # plt.title(f"{p} nodes     vertices: {G.order()}  deg: {avg_degree(G)}")
    # plt.show()


MPI.Finalize