import json
import numpy as np
import networkx as nx
from itertools import combinations
import csv
#################################################################################
with open('starwars-full-interactions-allCharacters-1.json') as f:
    d = json.load(f)
    nodes=d['nodes']
    links=d['links']

numnodes=len(nodes)
arr=np.zeros((numnodes,numnodes),dtype=int)
G = nx.Graph()

for j in range(len(links)):
    d2=links[j]
    ss=d2['source']
    tt=d2['target']

    G.add_edge(ss,tt)

    arr[ss][tt]=1
    arr[tt][ss]=1
#############################################
with open('matrix.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    for r in range(arr.shape[0]):
        writer.writerow(arr[r])
##############################################
com=[]
for k in range(numnodes):
    com.append(k)

comb = combinations(com,2)

myd = dict()
for t in range(numnodes):
    myd[t]=0

for m in list(comb):
    try:
        res=[p for p in nx.all_shortest_paths(G, source=m[0], target=m[1])]
        numshortStoT=len(res)
        for p in res:
            for n in range(len(p)):
                if (n!=0) and (n!=(len(p)-1)):
                    nn=p[n]
                    myd[nn]+=(1/numshortStoT) 
    except:
        pass

find=sorted(myd.items(), key=lambda item: item[1],reverse=True)

with open('rank.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for top in range(10):
        see=find[top]
        #print(see)
        writer.writerow(list(see))
#######################################3
#[[100, 95, 1, 7, 4], [100, 96, 1, 7, 4], [100, 95, 73, 7, 4], [100, 96, 73, 7, 4]]
