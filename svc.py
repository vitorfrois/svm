import numpy as np
import cvxpy as cp
import networkx as nx
import scipy

class SVC:
    """
    Support Vector Classifier
    """
    n: int
    heat_kernel: np.array
    train_nodes: list[int]
    y: np.array
    X: np.array

    def __init__(self, t: int = 1):
        self.t = t

    @staticmethod
    def compute_heat_kernel(graph: nx.Graph, t: int = None) -> np.array:
        if t <= 0 or not type(t) is int:
            raise Exception(f't should be positive integer. Received {t}')
        laplacian = nx.normalized_laplacian_matrix(graph).toarray()
        heat_kernel = scipy.linalg.expm(-t*laplacian)
        return heat_kernel

    def fit(self, graph: nx.Graph, node_label_dict: dict[int, int]):
        self.n = len(node_label_dict)
        self.heat_kernel = SVC.compute_heat_kernel(graph, self.t)
        self.train_nodes = list(node_label_dict.keys())
        self.y = np.array(list(node_label_dict.values()))
        self.X = self.heat_kernel[self.train_nodes, :][:, self.train_nodes]
        
        D = self.X * np.outer(self.y, self.y)
        alpha = cp.Variable(self.n)
        constraints = [
            alpha.T @ self.y == 0, 
            alpha >= 0
        ]
        lagrangian = -(1/2)*cp.quad_form(alpha, D) + cp.sum(alpha)
        prob = cp.Problem(cp.Maximize(lagrangian), constraints)
        prob.solve()

        self.alpha = alpha.value
        print(alpha.value)
        self.W = np.array([self.y[i] * self.alpha[i] * self.X[i] for i in range(self.n)])
        self.b = (self.y - self.heat_kernel[self.train_nodes, :][:, self.train_nodes] @ (self.y * alpha.value))[0]

    def predict(self):
        self.y_pred = self.heat_kernel[:][:, self.train_nodes] @ (self.y * self.alpha) + self.b
        return self.y_pred

c = SVC(t=3)

labels = [
    1,1,1,1,1,1,1,1,1,-1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
]
train_samples = [0, 1, 33]
node_label_dict = {i: labels[i] for i in train_samples}

G = nx.karate_club_graph()
c.fit(G, node_label_dict)

y_pred = c.predict()

error = np.sum(np.sign(y_pred) == labels) / (len(labels))

print(y_pred)
print(f"Accuracy: {error:.2f}")