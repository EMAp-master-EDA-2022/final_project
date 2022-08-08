# Final Project - EDA Master 2022

This repository contain the code and files to help in the implementation of the final project of the discipline of data structures and algorithms at EMAp Masters. The task is to implement a graph data structure that supports queries of shortest path between edges using the algorithms Dijkstra and A*.

The network is obtained from OSMnx and was updated with the script `graph_processing.py` that generates an output `RJ.pkl` , each edge will present attributes `maxspeed`, `length` and `highway`(and some others not used). It will also create a text file `penalization_values.txt` for the penalization of the max speed of each highway type, the format of the file is `hour minute highway penalization`.

The deadline is 07/09.