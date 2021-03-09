#%%
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter


#%%
fname = "./processed_data/rideaustin_productivity.csv"
data = pd.read_csv(fname, dtype={"timebin": int}, parse_dates=["completed_on"])

# for interactive env can do next two lines
# data.sort_values('end_taz', inplace=True)
# data.head()

# %% valid tazs
tazindata = data.end_taz.unique()
# len(tazindata)

# %% read adjacency info
fname = "./processed_data/taz_adjacency.csv"
taz_adjacency = np.genfromtxt(fname, delimiter=",", dtype=int)
tazindata_ = set(tazindata)
taz_adjacency = [
    [i, j] for i, j in taz_adjacency if (i in tazindata_) and (j in tazindata_)
]

# %% build spatial adjacency graph with networkx
gspatial = nx.Graph()
gspatial.add_edges_from(taz_adjacency)
conn_comps = nx.connected_components(gspatial)
conn_comps = sorted(conn_comps, key=len, reverse=True)
conn_largest = conn_comps[0]

# %% keep tazs in largest connected component
tazindata = [x for x in tazindata if x in conn_largest]
tazindata_ = set(tazindata)
taz_adjacency = [
    [i, j] for i, j in taz_adjacency if (i in tazindata_) and (j in tazindata_)
]


# %% filter data in valid tazs
data = data[data.end_taz.isin(tazindata_)]

# %% build the spatiotemporal graph
links = []
num_timebins = 168  # look at the column timebin
istemporal = []  # 0 for spatial, 1 for temporal
# spatial links in each time slice
for t in range(num_timebins):
    for i, j in taz_adjacency:
        v = "{}-{}".format(i, t + 1)
        w = "{}-{}".format(j, t + 1)
        links.append([v, w])
        istemporal.append(0)
# now add temporal links
for x in tazindata:
    for t in range(num_timebins - 1):
        v = "{}-{}".format(x, t + 1)
        w = "{}-{}".format(x, t + 2)
        links.append([v, w])
        istemporal.append(1)
    # for periodic time
    v = "{}-{}".format(x, num_timebins)
    w = "{}-{}".format(x, 1)
    links.append([v, w])
    istemporal.append(1)
g = nx.Graph()
g.add_edges_from(links)
# nx.number_connected_components(g) # should be one!


# %% vertex info
nodes = list(g.nodes())
node2vertex = {x: i for i, x in enumerate(nodes)}
df = pd.DataFrame({"node": nodes})
df["vertex"] = [node2vertex[node] for node in nodes]
df["taz"] = [int(node.split("-")[0]) for node in nodes]
df["hour"] = [int(node.split("-")[1]) for node in nodes]

# function for time labels
def timelabeller(hour):
    w = (hour - 1) // 24
    t = (hour - 1) % 24
    wdays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    return f"{wdays[w]} {t:02d}:00"


df["timelabel"] = df.hour.apply(timelabeller)

# counter number of nodes
cnts = Counter(data.node)
df["node_counts"] = df.node.apply(lambda x: cnts[x])

# save file
fname = "./processed_data/vertex_data.csv"
df.to_csv(fname, index=False)
print(f"...saved vertex info in {fname}")

# %%
spatiotemporal_graph = pd.DataFrame(
    {
        "vertex1": [node2vertex[v[0]] for v in links],
        "vertex2": [node2vertex[v[1]] for v in links],
        "temporal": [b for b in istemporal],
    }
)
fname = "./processed_data/spatiotemporal_graph.csv"
spatiotemporal_graph.to_csv(fname, index=False)
print(f"...saved spatiotemporal graph in {fname}")


#%% read split data
# fname = "./processed_data/splitlevels.csv"
splitlevels = pd.read_csv("processed_data/splits_opt_pt.csv")


# %% sort data in vertices
vertexdata = [[] for _ in range(len(nodes))]
for value, node in zip(data.productivity, data.node):
    vertex = node2vertex[node]
    vertexdata[vertex].append(value)

# %% bin counts
N = len(vertexdata)
M = splitlevels.shape[0]

attempts = np.zeros((N, M), dtype=int)  # data in cell
successes = np.zeros((N, M), dtype=int)  # data in left split

for j in range(M):
    left, mid, right = splitlevels.iloc[j, :3]
    for i, vdata in enumerate(vertexdata):
        for x in vdata:
            if left <= x < right:
                attempts[i, j] += 1
                if left <= x < mid:
                    successes[i, j] += 1


# %% now create splitdata files
directory = "./productivity_splits"
print(f"...saving data splits in {directory}/")

for j in range(M):
    sj, aj = successes[:, j], attempts[:, j]
    splitinfo = ", ".join(f"{x:.2f}" for x in splitlevels.iloc[j, :])
    print("- split " + splitinfo)
    print(f"  successes: {sj.sum()}  attempts {aj.sum()}")

    fname = f"{directory}/{j:02d}.csv"
    dat = np.column_stack([sj, aj])

    print(f"  saving to file {fname}...")
    np.savetxt(fname, dat, fmt="%d", delimiter=",")


# %%
