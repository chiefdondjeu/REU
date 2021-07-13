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
    return color_map

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

        cost = H(G, current_state) - H(G, solution)    # compute cost
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


# n = 100
# deg = 3

# #G = nx.gnm_random_graph(n, n)
# G = nx.random_degree_sequence_graph([deg for i in range(n)]) # degree sequence

# print(f"\n{G.order()} nodes") # G.order number of nodes
# print(f"highest degree: {highest_degree(G)}")
# print(f"avg degree: {avg_degree(G)}\n")

# colors = color_picker(3)  # numb of possible colors (max 7)
# numb_iter = 100

# color_map = gen_map(colors, G.order())
# min_cost = H(G, color_map)

# cost_data = [min_cost]

# print("annealing...")
# for i in tqdm(range(numb_iter)):
#     avg_run = np.array([])
#     for j in range(10):
#         temp = list(anneal(G, color_map, colors))
#         temp_cost = H(G,temp)
#         avg_run = np.append(avg_run, temp_cost)
#         if temp_cost < min_cost:
#             color_map = temp.copy()
#             min_cost = temp_cost
#     cost_data.append(np.average(avg_run))

# print(f"min cost: {H(G, color_map)}")

# if G.order() <= 10:
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, node_color=color_map, with_labels=True)
#     plt.show()

# plt.plot([i for i in range(numb_iter+1)], cost_data)
# plt.title(f"{G.order()} nodes    deg:{avg_degree(G)}")
# plt.xlabel("Runs")
# plt.ylabel("Cost")
# plt.show()


comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f"{rank} here")

if rank == 0: # parent nod
    n = 100
    G = nx.random_degree_sequence_graph([3 for i in range(n)]) # degree sequence
    colors = color_picker(3)  # numb of possible colors (max 7)
    color_map = gen_map(colors, G.order())

else: # other nodes
    #color_map = comm.recv(source=0)
    print(f"{rank} {color_map}")

comm.Bcast(color_map, root=0)