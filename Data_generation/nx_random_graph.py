import networkx as nx
import sys, getopt

def main(argv):
	N = int(argv[0])
	method = str(argv[1])
	m = int(argv[2])
	p = float(argv[3])
	
	G = -1
	if method == 'er':
		G = nx.erdos_renyi_graph(N, p, seed=None, directed=False)
	elif method == 'ba':
		G = nx.barabasi_albert_graph(N, m, seed=None)
	elif method == 'ws':
		G = nx.watts_strogatz_graph(N, m, p, seed=None)
	elif method == 'pl':
		G = nx.powerlaw_cluster_graph(N, m, p, seed=None)
	
	A = [[0 for x in range(N)] for y in range(N)]
	for (k,v) in G.edges():
		A[k][v] = 1
		A[v][k] = 1
	
	f = open('Data_generation/Adj.csv', 'w')
	for i in range(0,N):
		for j in range(0,N):
			f.write(str(A[i][j]))
			if j != N - 1:
				f.write(',')
		f.write('\n')
	f.close()
	
	print('success')
	return;

if __name__ == "__main__":
   main(sys.argv[1:])
