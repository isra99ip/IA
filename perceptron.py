# perceptron.py (from scratch, sin librerías ML)
# Activaciones estables + clipping de pesos
import math, random
from typing import List

# ---------- Activaciones ----------
def step(x: float) -> float:
    return 1.0 if x >= 0.0 else 0.0

def linear(x: float) -> float:
    return x

def sigmoid(x: float) -> float:
    """
    Sigmoide numéricamente estable:
    evita llamar exp(-x) con x muy grande.
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def tanh(x: float) -> float:
    # Implementación estable de la librería estándar
    return math.tanh(x)

def relu(x: float) -> float:
    return x if x > 0.0 else 0.0

def softmax(z: List[float]) -> List[float]:
    # Softmax estable (resta el máximo)
    m = max(z)
    exps = [math.exp(v - m) for v in z]
    s = sum(exps)
    return [v / s for v in exps]

ACTIVATIONS = {
    "step": step,
    "linear": linear,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "relu": relu,
}

# ---------- Perceptrón ----------
class Perceptron:
    """
    Perceptrón de una capa:
    - Una salida escalar con activaciones: step, linear, sigmoid, tanh, relu
    - 2 salidas con softmax (para clasificar 0/1)
    Entrenamiento online (por muestra).
    """
    def __init__(
        self,
        n_inputs: int,
        activation: str = "sigmoid",
        lr: float = 0.1,
        seed: int = 42,
        clip_limit: float = 50.0  # límite para clipping (None o <=0 para desactivar)
    ):
        assert n_inputs >= 1
        self.n = n_inputs
        self.activation = activation.lower()
        self.lr = lr
        self.clip_limit = clip_limit if (clip_limit is not None and clip_limit > 0) else None

        random.seed(seed)
        if self.activation == "softmax":
            # 2 neuronas de salida
            self.W = [[random.uniform(-0.5, 0.5) for _ in range(n_inputs)] for _ in range(2)]
            self.b = [0.0, 0.0]
        else:
            self.W = [random.uniform(-0.5, 0.5) for _ in range(n_inputs)]
            self.b = random.uniform(-0.5, 0.5)

    # ----- utilidades -----
    def _dot(self, w, x):
        return sum(wi * xi for wi, xi in zip(w, x))

    def _clip_params(self):
        """Evita explosiones numéricas recortando W y b al rango [-clip, clip]."""
        if self.clip_limit is None:
            return
        c = self.clip_limit

        if self.activation == "softmax":
            for k in range(2):
                for j in range(self.n):
                    if self.W[k][j] > c:   self.W[k][j] = c
                    if self.W[k][j] < -c:  self.W[k][j] = -c
            for k in range(2):
                if self.b[k] > c:   self.b[k] = c
                if self.b[k] < -c:  self.b[k] = -c
        else:
            for j in range(self.n):
                if self.W[j] > c:   self.W[j] = c
                if self.W[j] < -c:  self.W[j] = -c
            if self.b > c:   self.b = c
            if self.b < -c:  self.b = -c

    # ----- forward / predict -----
    def _forward(self, x: List[float]):
        if self.activation == "softmax":
            z = [self._dot(wk, x) + bk for wk, bk in zip(self.W, self.b)]
            y = softmax(z)
            return z, y
        else:
            z = self._dot(self.W, x) + self.b
            y = ACTIVATIONS[self.activation](z)
            return z, y

    def predict(self, x: List[float]):
        _, y = self._forward(x)
        if self.activation == "softmax":
            # clase 0 o 1
            return 0 if y[0] >= y[1] else 1
        # para activaciones escalares, umbral 0.5
        return 1 if y >= 0.5 else 0

    # ----- entrenamiento -----
    def fit(self, X: List[List[float]], y: List[int], epochs: int = 20):
        assert epochs >= 10, "La consigna exige > 10 iteraciones."
        for _ in range(epochs):
            for xi, ti in zip(X, y):
                if self.activation == "softmax":
                    # forward
                    z, p = self._forward(xi)  # p: probs [p0, p1]
                    target = [1.0, 0.0] if ti == 0 else [0.0, 1.0]
                    # gradiente softmax+cross-entropy: (p - target)
                    grad = [p[k] - target[k] for k in range(2)]
                    # update
                    for k in range(2):
                        for j in range(self.n):
                            self.W[k][j] -= self.lr * grad[k] * xi[j]
                        self.b[k] -= self.lr * grad[k]
                    self._clip_params()
                else:
                    z, yi = self._forward(xi)
                    err = float(ti) - yi

                    # derivada de la activación (respecto a z)
                    if   self.activation == "linear":
                        dphi = 1.0
                    elif self.activation == "sigmoid":
                        dphi = yi * (1.0 - yi)  # usando salida ya sigmoidal
                    elif self.activation == "tanh":
                        dphi = 1.0 - yi * yi     # usando salida tanh
                    elif self.activation == "relu":
                        dphi = 1.0 if z > 0.0 else 0.0
                    elif self.activation == "step":
                        dphi = None  # no derivable: usar regla clásica

                    # actualización
                    if self.activation == "step":
                        # Regla del perceptrón clásico
                        if err != 0.0:
                            for j in range(self.n):
                                self.W[j] += self.lr * err * xi[j]
                            self.b += self.lr * err
                    else:
                        grad = err * dphi
                        for j in range(self.n):
                            self.W[j] += self.lr * grad * xi[j]
                        self.b += self.lr * grad

                    self._clip_params()

# ---------- Datasets de prueba ----------
def dataset_AND():
    X = [[0,0],[0,1],[1,0],[1,1]]
    y = [0,0,0,1]
    return X, y

def dataset_OR():
    X = [[0,0],[0,1],[1,0],[1,1]]
    y = [0,1,1,1]
    return X, y

if __name__ == "__main__":
    # Prueba rápida con AND
    X, y = dataset_AND()
    clf = Perceptron(n_inputs=2, activation="sigmoid", lr=0.2, seed=1, clip_limit=50.0)
    clf.fit(X, y, epochs=20)
    for xi in X:
        print(xi, "->", clf.predict(xi))
