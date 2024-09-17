# Graph
A graph is a collection of nodes, also called vertices, and the connections between them, called edges.

## Important Terminologies
- **Vertex**: A vertex (or node) is a fundamental part of a graph. It can represent any entity, such as a location, a person, or an object. In diagrams, vertices are often represented as circles or dots.
- **Edge**: An edge (or arc) is a connection between two vertices in a graph. It can represent a relationship or a path between the entities represented by the vertices.
- **Directed Graph**: A directed graph is a type of graph where the edges have a direction. Each edge points from one vertex to another.
```mermaid
graph TD  
    A-->B  
    B-->C  
    C-->D  
    D-->A  
    A-->C  
    B-->D
```
- **Undirected Graph**: An undirected graph is a type of graph where the edges have no direction. The connection between two vertices is bidirectional.

```mermaid
graph TD  
    A<-->B  
    B<-->C  
    C<-->D  
    D<-->A  
    A<-->C  
    B<-->D
```
**As mermaid doesn't support undirected graphs it would be represented as bidirectional edges.**

- **Weighted Graph**: A weighted graph is a graph in which each edge has an associated numerical value, called weight. This can represent cost, distance, or any other metric.
```mermaid
graph TD  
    A-->|5|B  
    B-->|3|C  
    C-->|7|D  
    D-->|2|A  
    A-->|4|C  
    B-->|6|D
```

## Graph Representation

### Non-Weighted Graph
Let the graph be this:
```mermaid
graph TD  
    0[0] --> 1[1]  
    0[0] --> 2[2]  
    1[1] --> 3[3]  
    2[2] --> 3[3]  
    3[3] --> 4[4]  

```
#### 1. Adjacency List
The adjacency list is a dictionary where each key represents a node, and the value is a list of nodes to which the key node has directed edges.

```python
adj_list = {  
    0: [1, 2],  
    1: [3],  
    2: [3],  
    3: [4],  
    4: []  
}
```

#### 2. Adjacency Matrix
The adjacency matrix is a 2D list (list of lists) where the element at row i and column j is 1 if there is a directed edge from node i to node j, and 0 otherwise.

```python
adj_matrix = [  
    [0, 1, 1, 0, 0],  # 0  
    [0, 0, 0, 1, 0],  # 1  
    [0, 0, 0, 1, 0],  # 2  
    [0, 0, 0, 0, 1],  # 3  
    [0, 0, 0, 0, 0]   # 4  
]
```

If the graph is weighted, the adjacency list and adjacency matrix representations will include the weights of the edges. Here is how you can represent the given graph with weights.

### Weighted Graph
```mermaid
graph TD  
    0[0] -->|5| 1[1]  
    0[0] -->|3| 2[2]  
    1[1] -->|2| 3[3]  
    2[2] -->|4| 3[3]  
    3[3] -->|1| 4[4]
```

#### 1. Adjacency List
The adjacency list is a dictionary where each key represents a node, and the value is a list of tuples. Each tuple contains a node to which the key node has a directed edge and the weight of that edge.

```python
adj_list = {  
    0: [(1, 5), (2, 3)],  
    1: [(3, 2)],  
    2: [(3, 4)],  
    3: [(4, 1)],  
    4: []  
}
```

#### 2. Adjacency Matrix
The adjacency matrix is a 2D list (list of lists) where the element at row i and column j is the weight of the edge from node i to node j, and 0 if there is no edge.
```python
adj_matrix = [  
    [0, 5, 3, 0, 0],  # 0  
    [0, 0, 0, 2, 0],  # 1  
    [0, 0, 0, 4, 0],  # 2  
    [0, 0, 0, 0, 1],  # 3  
    [0, 0, 0, 0, 0]   # 4  
]
```

## Connected Components
A connected component in an undirected graph is a group of nodes such that:
- Every node in the group is connected to every other node in the group by some path.
- There are no connections between nodes in this group and any nodes outside of this group.


```mermaid
graph TD  
    A[A] --- B[B]  
    B[B] --- C[C]  
    D[D] --- E[E]  
    F[F]
```

**From now on we would be only using the adjacency list mechanism for representation**.

## Traversal Techniques
### Breadth First Search
The BFS traversal explores all nodes at the present depth before moving on to nodes at the next depth level.
```mermaid
graph TD  
    1[1] <--> 2[2]  
    1[1] <--> 6[6]  
    2[2] <--> 3[3]  
    2[2] <--> 4[4]  
    4[4] <--> 5[5]  
    6[6] <--> 7[7]  
    6[6] <--> 8[8]  
    7[7] <--> 5[5]
``` 
Possible BFS with starting node as `1`: [1], [2, 6], [3, 4, 7, 8], [5] - the sub-divisions are for the different levels.
Now if we change the starting node to `6`: [6], [1, 7, 8], [2, 5], [3, 4].

#### Implementation
- For the implementation of BFS we use a queue and a visited set.
- For each level we empty the queue and traverse the nodes not traversed already and add them to the result.
- The visited set is important as it helps us from revisiting the same node again and again. As soon as a node is added to the queue, it is marked as visited.

Code
```python
def bfs(n, edges):
    graph = {i: [] for i in range(n)}
    for edge in edges:
        src, dest = edge
        graph[src].append(dest)
        graph[dest].append(src)
    
    # initialize queue and visited set
    visited = set()
    queue = deque()

    # add the starting node
    queue.append(0)
    visited.add(0)

    result = []
    while queue:
        temp = deque()
        # get the nodes for the current level
        while queue:
            # follow FIFO
            node = queue.popleft()
            # add to result as they are being traversed
            result.append(node)
            for adj_node in graph[node]:
                if adj_node not in visited:
                    # visit adjacent nodes for the next level traversal
                    temp.add(adj_node)
                    # add the adjacent node to the visited set
                    visited.add(adj_node)
        queue = temp
    
    return result
```

### Depth First Search
The DFS traversal visits nodes by exploring as far as possible along each branch before backtracking.
```mermaid
graph TD  
    1[1] <--> 2[2]  
    1[1] <--> 6[6]  
    2[2] <--> 3[3]  
    2[2] <--> 4[4]  
    4[4] <--> 5[5]  
    6[6] <--> 7[7]  
    6[6] <--> 8[8]  
    7[7] <--> 5[5]
``` 
Possible DFS with starting node as `1`: [1, 2, 3, 4, 5, 7, 6, 8].

#### Implementation
- We would be using recursion to implement DFS.
- Like in BFS we would also have a `visited` set to mark the nodes that are already visited.

Code:
```python
def traverse(n, edges):
    graph = {i: [] for i in range(n)}
    for edge in edges:
        src, dest = edge
        graph[src].append(dest)
        graph[dest].append(src)
    
    # initialize visited set
    visited = set()
    
    result = []

    def dfs(node):
        visited.add(node)
        result.add(node)
        for adj_node in graph[node]:
            if adj_node not in visited:
                solve(node)
    
    solve(0)
    return result
```

#### Extension of DFS - Connected Components
For connected components, this is how the dfs logic would change.
Since we know that in one traversal all the nodes wouldn't be traversed, we would perform dfs for all the nodes if they are not visited yet.

```python
def traverse(graph)
    # initialize visited set
    visited = set()
    result = []

    def dfs(node):
        visited.add(node)
        result.append(node)

        # this ensures that all the nodes in this group are traversed
        for adj_node in graph[node]:
            if adj_node not in visited:
                dfs(adj_node)

    for node in graph:
        # check if the node exists
        if node not in visited:
            dfs(node)
```

Related problems:
- Number of Provinces
- Number of Islands

Now coming to the various algorithms of graph:

### Cycle Detection
Given a directed graph, check whether the graph contains a cycle or not.
This can be done using the DFS method.
We need to modify the existing DFS implementation to check for a backedge - that can cause cycles. For this we maintain a separate `rec_stack` set along with the existing `visited` set.

#### Why do we need a separate `rec_stack` set, wouldn't `visited` set be enough?
No, it would be enough for an undirected graph, but for a directed graph it can give false-positives for cycles.
```mermaid
graph LR  
    A((A)) --> B((B))  
    A((A)) --> C((C))  
    B((B)) --> D((D))  
    C((C)) --> D((D))
```
When visiting `C` and goind to it's adjacent nodes `D` already exists in the `visited` set. If we had only it, we would have wrongly judged the graph that it has a cycle.

```python
def cycle_detection(graph):
    # to check for cycles
    rec_stack = set()
    # to check if already visited
    visited = set()

    def dfs(node):        
        rec_stack.add(node)
        visited.add(node)

        for adj_node in graph[node]:
            # additonal check if node in recursion stack
            if adj_node in rec_stack:
                return True
            
            if adj_node not in visited:
                dfs(adj_node)
        
        # no longer, a part of the recursion stack
        rec_stack.remove(node)
        return False
    
    # Check all nodes in the graph to handle disconnected components  
    for node in graph:  
        if node not in visited:  
            if dfs(node):  
                return True 
    return False
```

### Topological Sort
Topological sort is a way of arranging the nodes in a directed acyclic graph (DAG) in a linear order such that for every directed edge from node `u` to node `v`:
- Node `u` appears before node `v` in the ordering.

Think of it like scheduling tasks where some tasks must be completed before others. Topological sort gives you an order in which to complete the tasks so that all the dependencies are respected.
Example:
Consider a graph representing tasks with dependencies:
```mermaid
graph TD  
    A[A] --> B[B]  
    A[A] --> C[C]  
    B[B] --> D[D]  
    C[C] --> D[D]
```
- Task A must be completed before tasks B and C.
- Tasks B and C must both be completed before task D.

A possible topological order for these tasks could be: A, B, C, D or A, C, B, D. Both orders respect the dependencies.

#### 1. Kahn's Algorithm
#### 2. DFS based
We add a node to the result only when all the adjacent nodes are already traversed.
This would be an extesnion of the existing `DFS` where along with the traversal we also empty the outgoing edges adjacency list.

Code
```python
def topological_sort(graph):
    visited = set()
    # for detecting cycles
    rec_stack = set()
    result = []

    def dfs(node):
        visited.add(node)
        rec_stack.add(node)

        for adj_node in graph[node]:
            if adj_node in rec_stack:
                return False
            if adj_node not in visited:
                if not dfs(node):
                    return False
        
        rec_stack.remove(node)
        result.append(node)
        return True
    
    for node in graph:
        if node not in visited:
            if not dfs(node):
                return []
    
    result.reverse()
    return result
```

### Flood Fill Algorithm
The Flood Fill algorithm is used to fill a contiguous region of pixels with a particular color, starting from a given seed pixel.

Code
```python
def flood_fill(grid, row, col, new_color):  
    rows, cols = len(grid), len(grid[0])  
    original_color = grid[row][col]
    # possible directions
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  
    # Base case: If the new color is the same as the original color, no change is needed  
    if original_color == new_color:  
        return grid  
  
    def is_valid(row, col):
        return 0<=row<rows and 0<=col<cols

    def fill(row, col):  
        # Set the new color  
        grid[row][col] = new_color

        # Recursively fill the neighboring cells
        for direction in directions:
            n_row, n_col = row + direction[0], col + direction[1]
            if is_valid(n_row, n_col) and grid[n_row][n_col] == original_color:        
                fill(n_row, n_col)
  
    # Start the flood fill from the given starting cell  
    fill(row, col)  
    return grid
```
The recursive approach handles the issue of visiting the same cell again by marking cells with the new color as they are visited. Once a cell is changed to the new color, it will no longer match the original_color, which prevents it from being revisited.

### Shortest Path Algorithms
### 1. Single Source Shortest Path
##### Dijkstra
Dijkstra's algorithm is a graph search algorithm that solves the single-source shortest path problem for a graph with **non-negative edge weights**, producing a shortest path tree. This means it finds the shortest paths from a starting node (source) to all other nodes in the graph.

###### How it Works:
- **Initialization**: Start with a distance array, dist, where dist[source] is 0 (distance to itself) and all other distances are set to infinity.
- **Priority Queue**: Use a priority queue (min-heap) to select the node with the smallest distance.
- **Relaxation**: For each selected node, update the distances of its neighbors if a shorter path is found via the current node.
- **Repeat**: Continue until all nodes have been processed.

Code
```python
def dijkstra(start, graph):
    # map to store shortest distance
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    
    # min-heap to store the minimum distance node at the root
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Nodes can be added to the priority queue multiple times. We only  
        # process a node the first time we remove it from the priority queue.
        if current_node in visited:
            continue
        
        visited.add(current_node)

        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight

            # Only consider this new path if it's better
            if dist[neighbor] > distance:
                dist[neighbor] = distance 
                heapq.heappush(priority_queue, (distance, neighbor))
    
    for node in graph:
        # for nodes that are not reachable from the start node
        if dist[node] == float('inf'):
            dist[node] = -1
    return dist
```
##### Bellman Ford
The Bellman-Ford algorithm is another algorithm for finding the shortest paths from a single source vertex to all other vertices in a weighted graph. Unlike Dijkstra's algorithm, Bellman-Ford can handle graphs with negative edge weights. It is slower but more versatile.

**Negative Weight Cycle**
A negative weight cycle in a graph is a loop where the sum of the edge weights is negative. These cycles are crucial in shortest path algorithms because they allow you to keep decreasing the path length endlessly by looping through the cycle. This makes it impossible to define a "shortest path" accurately.

###### How it Works
- **Initialization**: Start with a distance array, dist, where dist[source] i  0 (distance to itself) and all other distances are set to infinity.
- **Relaxation**: For each edge, update the distances of the two vertices if a shorter path is found.
- **Repeat**: Repeat the relaxation process for `|V| - 1` times (where `|V|` is the number of vertices).
- **Negative Cycle Check**: Perform an additional relaxation to detect negative weight cycles. If a shorter path is found, then a negative weight cycle exists.

Code
1. When the adjacency list is provided as an input
```python
def bellman_ford(start, graph):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    n = len(graph)

    for _ in range(n - 1):
        for node in graph:
            for neighbor, weight in graph[node]:
                if dist[node] != float('inf') and dist[node] + weight < dist[neighbor]:
                    dist[neighbor] = dist[node] + weight
    
    # Check for negative weight cycles
    for node in graph:
        for neighbor, weight in graph[node]:
            if dist[node] != float('inf') and dist[node] + weight < dist[neighbor]:
                # graph contains a negative weight cycle
                return -1

    return dist  
 ```
2. When the edges are provided as an input
```python
def bellman_ford_edges(start, edges, num_vertices):  
    dist = {i: float('inf') for i in range(num_vertices)}  
    dist[start] = 0  
  
    for _ in range(num_vertices - 1):  
        for u, v, w in edges:  
            if dist[u] != float('inf') and dist[u] + w < dist[v]:  
                dist[v] = dist[u] + w  
  
    # Check for negative weight cycles  
    for u, v, w in edges:  
        if dist[u] != float('inf') and dist[u] + w < dist[v]:  
            # Graph contains a negative weight cycle  
            return -1  
  
    return dist
```
**Why on |V|-1 we get the result?**
Since the longest possible path without repeating any vertex in a graph with |V| vertices has |V| - 1 edges, the algorithm needs up to |V| - 1 iterations to ensure that the shortest path to each vertex is found.

Example:
```mermaid
graph LR
    A[1] --> B[2]
    B --> C[3]
    C --> D[4]
    D --> E[5] 
```

Starting with node 1, for this graph to relax till E we would need `|V| - 1` or 4 iterations.

#### 2. Multi Source Shortest Path
##### Floyd Warshall Algorithm
The Floyd-Warshall algorithm is used to find the shortest paths between all pairs of vertices in a weighted graph. It can handle positive and negative edge weights but not negative weight cycles.

###### How it Works:
- **Initialization**: Create a distance matrix `dist` where `dist[i][j]` is the weight of the edge from vertex `i` to vertex `j`. If no such edge exists, initialize `dist[i][j]` to infinity. the diagonal elements `dist[i][i]` are set to 0.
- **Relaxation**: For each pair of vertices (i, j), update the `dist[i][j]` to be the minimum of `dist[i][k] + dist[k][j]` for each intermediate vertex k.
- **Repeat**: Repeat the relaxation for all pairs of vertices and all intermediate vertices.
- **Negative Cycle Check**: After computing the shortest paths, if the distance from a vertex to itself becomes negative (dist[i][i] < 0 for any vertex i), it indicates a negative weight cycle.

Code
```python
def floyd_warshall(graph):
    n = len(graph)
    dist = [[float('inf')] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0
        for j, weight in graph[i]:
            dist[i][j] = weight
    
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    
    # check for negative weight cycles
    for i in range(n):
        if dist[i][i] < 0:
            # graph contains a negative weight cycle
            return -1
    return dist
``` 

### Disjoint Union Set - DSU
Two sets are called disjoint if they don’t share any element; their intersection is an empty set. Also known as Union-Find as it supports the following operations:
- Merging disjoint sets into a single disjoint set using the **Union** operation.
- Finding the representative of a disjoint set using the **Find** operation.

**Union By Rank**:
The idea is to always attach the smaller tree under the root of the larger tree, thereby minimizing the maximum height of the trees. This technique is known as union by rank.

#### Find
The **Find** operation is used to determine which subset a particular element is in. This can be used to check if two elements are in the same subset.

##### Example:
```mermaid
graph BT  
    A1[1] --> B2[2]  
    B2 --> C3[3]  
    D4[4] --> E5[5]  
    F6[6] --> G7[7]  
    G7 --> H8[8]
```
- Find(1): To find the representative of the set containing element 1, we follow the pointer from 1 to 2, and from 2 to 3. Thus, the representative for element 1 is 3.
- Find(4): To find the representative of the set containing element 4, we follow the pointer from 4 to 5. Thus, the representative for element 4 is 5.

##### Code
```python
def find(node):
    if node != parent[node]:
        parent[node] = find(parent[node]) #path compression
    return parent[node]
```

#### Union
The Union operation is used to merge two subsets into a single subset. This is useful when you need to combine the sets containing two different elements.

##### Example
Consider the following sets represented as a forest:

```mermaid
graph BT  
    A1[1] --> B2[2]  
    B2 --> C3[3]  
    D4[4] --> E5[5]  
    F6[6] --> G7[7]  
    G7 --> H8[8]  
```

Let's say we want to merge the sets containing elements 1 and 4:
- First, we find the representatives of each set:
- Find(1) returns 3.
- Find(4) returns 5.
- Then, we merge the sets by making one representative the parent of the other.

##### Code
```python
def union(node1, node2):  
    root1 = find(node1)  
    root2 = find(node2)  
      
    if root1 != root2:  
        parent[root2] = root1  # Merge the sets
```

#### Union By Rank
The Union-Find algorithm can be optimized using the `Union by Rank` technique to keep the tree shallow, which improves the efficiency of both find and union operations.

##### Code
```python
n = 8
rank = [1] * n
def union(node1, node2):  
    parent1 = find(node1)  
    parent2 = find(node2)  
      
    if parent1 != parent2:
        if rank[parent1] >= rank[parent2]:
            parent[parent2] = parent1 # Merge the sets
            rank[parent1] = rank[parent1] + rank[parent2]
        else:  
            parent[parent1] = parent2
            rank[parent2] = rank[parent2] + rank[parent1]
```

#### Problems
##### Number of Connected Components in an Undirected Graph
Given an undirected graph with n nodes and edges, find the number of connected components in the graph.
```python
def count_components(n, edges):
    parent = {i: i for i in range(n)}
    rank = [1] * n

    def find(node):
        if node != parent[node]:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node 2):
        parent1 = find(node1)
        parent2 = find(node2)

        if parent1 != parent2:
            if rank[parent1] >= rank[parent2]:
                parent[node2] = parent1
                rank[parent1] = rank[parent1] + rank[parent2]
            else:
                parent[node1] = parent2
                rank[parent2] = rank[parent2] + rank[parent1]

    for src, dest in edges:
        union(src, dest)

    components = set()
    for node in range(n):
        components.add(find(node))

    return len(components) 
```

### Minimum Spanning Tree
A Minimum Spanning Tree (MST) of a weighted, connected, undirected graph is a spanning tree that has the minimum possible total edge weight compared to all other spanning trees of the graph.
**Easy words**: A Minimum Spanning Tree connects all the vertices in a graph with the minimum possible total edge weight, ensuring no cycles are formed.

#### Characteristics
- Spanning Tree: A spanning tree of a graph is a subgraph that includes all the vertices of the original graph and is a tree (i.e., it is connected and acyclic).
- Minimum Total Edge Weight: Among all possible spanning trees, the MST has the least sum of the weights of its edges.
- Uniqueness: If all the edge weights are distinct, the MST is unique. If there are edges with equal weights, there may be multiple MSTs with the same total weight.

#### Algorithms to find MST
##### 1. Prim's
Prim's Algorithm is a greedy algorithm that finds a Minimum Spanning Tree (MST) for a weighted, connected, undirected graph. The algorithm operates by growing the MST one vertex at a time, starting from an arbitrary vertex and repeatedly adding the smallest edge that connects a vertex in the MST to a vertex outside the MST.

**Step-by-Step Process:**
- Initialization:
  - Start with an arbitrary vertex, and mark it as part of the MST.
  - Initialize a priority queue (min-heap) to keep track of the smallest edges that connect vertices inside the MST to vertices outside the MST.
- Growing the MST:
  - While there are still vertices not included in the MST:
  - Extract the edge with the smallest weight from the priority queue.
  - Add the edge and the vertex it connects to the MST (if the vertex is not already in the MST).
  - For the newly added vertex, add all edges connecting it to vertices outside the MST to the priority queue.
- Completion:
  - The algorithm completes when all vertices are included in the MST.

Code
```python
def prim(start, graph):
    min_heap = [(0, start)]
    visited = set()
    min_cost = 0

    while min_heap:
        node_weight, node = heapq.heappop(min_heap)
        
        # If the node has already been visited, skip it
        if node in visited:  
            continue 

        visited.add(node)
        min_cost = min_cost + node_weight

        for neighbor, weight in graph[node]:
            if neighbor not in visited:
                heapq.heappush(min_heap, (weight, neighbor))
    
    return min_cost
```
##### 2. Kruskal's
### Hamiltonian Path - Travelling Salesman Problem
### Graph Coloring
### Strongly connected components - Kosaraju's Algorithm
#### Definition
A maximal subgraph of a directed graph such that for every pair of vertices (u) and (v) in the subgraph, there is a directed path from (u) to (v) and a directed path from (v) to (u).

Example:
```mermaid
graph LR 
    2 --> 0  
    0 --> 1  
    1 --> 2  
    2 --> 3  
    3 --> 4  
    4 --> 5  
    5 --> 6  
    4 --> 7  
    6 --> 7  
    6 --> 4
```

SCCs:
```mermaid
graph LR  
    subgraph SCC1  
        A0[0] --> A1[1]  
        A1[1] --> A2[2]  
        A2[2] --> A0[0]  
    end  
  
    subgraph SCC2  
        B3[3]  
    end  
  
    subgraph SCC3  
        C4[4] --> C5[5]  
        C5[5] --> C6[6]  
        C6[6] --> C4[4]  
    end  
  
    subgraph SCC4  
        C7[7]  
    end  
  
    A2 --> B3  
    B3 --> C4
    C4 --> C7  
    C6 --> C7 
```
### Network Flow
#### 1. Ford-Fulkerson
#### 2. Edmonds Karp
