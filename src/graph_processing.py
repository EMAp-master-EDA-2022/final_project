# OSMnx graph of Rio de Janeiro present many highway classes and many missing 
# values for "maxspeed" of the streets, this code will simplify that and input 
# "maxspeed" values when necessary.
# It will also create the document with the penalization function of maxspeed.

import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # simplification rules of highways classes
    highway_mapper = {
        "residential" : "residential",
        "primary" : "primary",
        "secondary" : "secondary",
        "tertiary" : "tertiary",
        "trunk": "motorway",
        "motorway" : "motorway",
        "living_street" : "living_street",
        "busway" : "busway",
        "trunk_link" : "motorway",
        "motorway_link" : "motorway",
        "primary_link" : "primary",
        "secondary_link" : "secondary",
        "tertiary_link" : "tertiary",
        "unclassified" : "tertiary",
    }

    highway_order_importance = [
        'motorway', 'primary', 'secondary', 'tertiary', 'residential', 'busway', 'living_street'
        ]

    # get available info from the network
    RJ = ox.graph_from_place('Rio de Janeiro', network_type='drive')

    df = {"maxspeed": [], "highway": [], "u": [], "v": [], "k" : []}
    for u, v, k, e in RJ.edges(data = True, keys = True):
        if "maxspeed" in e.keys():
            df["maxspeed"].append(e["maxspeed"])
        else:
            df["maxspeed"].append(None)
        df["highway"].append(e["highway"])
        df["u"].append(u)
        df["v"].append(v)
        df["k"].append(k)
    df = pd.DataFrame(df)

    # deal with lists as values
    def remove_highway_list(x):
        if type(x) == str:
            return highway_mapper[x]
        x_ = [highway_mapper[v] for v in x]
        for v in highway_order_importance:
            if v in x_:
                return highway_mapper[v]

    remove_maxspeed_list = lambda x : max(x) if type(x) == list else x
    
    df["highway"] = df.highway.apply(remove_highway_list)
    df["maxspeed"] = df.maxspeed.apply(remove_maxspeed_list)

    # for each class, get the most common maxspeed
    maxspeed_mapper = {}

    for highway in highway_order_importance:
        maxspeed_mode = df[((df.highway == highway) & (~df.maxspeed.isna()))].maxspeed.mode()
        if maxspeed_mode.shape[0] > 0:
            maxspeed_mapper[highway] = float(maxspeed_mode.values[0])
        else:
            maxspeed_mapper[highway] = 0

    # update the maxspeed with the most common value for the highway type
    df["maxspeed"] = df.highway.apply(lambda x : maxspeed_mapper[x])

    # update the graph
    for i, row in df.iterrows():
        u, v, k, maxspeed, highway = row.u, row.v, row.k, row.maxspeed, row.highway
        RJ.edges[u, v, k]["maxspeed"] = maxspeed
        RJ.edges[u, v, k]["highway"] = highway

    # save with pickle
    nx.write_gpickle(RJ, "RJ.pkl")

    # create penalization function for each highway type
    np.random.seed(0)

    highway_penalizer = {
        'motorway' : 0.9,
        'primary' : 0.7,
        'secondary' : 0.5, 
        'tertiary' : 0.5,
        'residential': 0.7,
        'busway' : 1,
        'living_street' : 1
    }

    def f_penalization(h, m, p_):
        p = p_ + np.random.uniform(0, 0.1)
        if h not in [7, 8, 11, 12, 17, 18]:
            return 1
        if h == 7 or h == 11 or h == 17:
            return 1 - m / 60 * p
        if h == 8 or h == 12 or h == 18:
            return (1 - p) + m / 60 * p

    f_highway = dict([(highway, []) for highway in highway_order_importance])

    with open("penalization_values.txt", "w+") as f:
        for h in range(24):
            for m in range(0, 60, 10):
                for highway in highway_order_importance:
                    value = f_penalization(h, m, 1 - highway_penalizer[highway])
                    f.write(f"{h} {m} {highway} {value}\n")
                    f_highway[highway].append(value)

    for highway in highway_order_importance:
        plt.plot(f_highway[highway], label = highway)
        
    plt.title("Penalization of maxspeed")
    xticks = list(range(0, 140, 20))
    xticks_labels = [f"{x//6}:{x%6 * 10}" for x in xticks]
    plt.xticks(xticks, labels = xticks_labels)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()