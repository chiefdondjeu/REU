import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

####
## Functions
####

# cost function
def H(Graph, color_map):
    cost = 0
    for (v, w) in Graph.edges():
        if color_map[v] == color_map[w]:
            cost += 1
    return cost

# generates a random color map
def gen_map(colors, n):
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


#####
## CONFIGURATION
####

nCol = 3
colors = color_picker(nCol) # numb of possible colors (max 7)
iterations = 100

vert_count = 100
edge_count = vert_count
deg = 3


####
## Generate Graph
####

# G = nx.gnm_random_graph(vert_count, edge_count)
G = nx.random_degree_sequence_graph([deg for i in range(vert_count)]) # degree sequence

print(f"\n{G.order()} Vertices\nHighest deg: {highest_degree(G)} | Avg deg: {avg_degree(G)}\n")

if G.order() <= 10: # G.order number of nodes
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)

####
## Anneal
####

coloring = gen_map(colors, G.order())
min_cost = H(G, coloring)

cost_data = np.array([min_cost])
avg = 3

print(f"initial cost {H(G,coloring)}\n")
print("annealing...")

# REFINED
for i in tqdm(range(iterations)):
    avg_run = np.array([])
    for j in range(avg):
        sol = anneal(G, coloring, colors)
        sol_cost = H(G, sol)
        avg_run = np.append(avg_run, sol_cost)
        if sol_cost < min_cost:
            coloring = sol
            min_cost = sol_cost
    cost_data = np.append(cost_data, np.average(avg_run))

# # NOT REFINED
# raw_data = np.array([min_cost])
# for i in tqdm(range(iterations)):
#     sol = anneal(G, coloring, colors)
#     sol_cost = H(G, sol)
#     raw_data = np.append(raw_data, sol_cost)
#     if sol_cost < min_cost:
#         coloring = sol
#         min_cost = sol_cost
#     cost_data = np.append(cost_data, min_cost)


print(f"min cost {H(G,coloring)}")

if G.order() <= 10:
    nx.draw(G, pos, node_color=coloring, with_labels=True)

####
## Plot
####

plt.plot([x for x in range(iterations+1)], cost_data)
plt.title(f"{G.order()} vertices -- deg {avg_degree(G)} -- {nCol} colors")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()