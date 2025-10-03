# run_experiments.py
import csv, sys
from perceptron import Perceptron

def load_csv(path):
    X=[]; y=[]
    with open(path, newline='', encoding='utf-8') as f:
        r=csv.reader(f); header=next(r)
        for row in r:
            *features, label = row
            X.append([float(v) for v in features])
            y.append(int(label))
    return X,y

def accuracy(model, X, y):
    return sum(1 for xi,ti in zip(X,y) if model.predict(xi)==ti)/len(y)

def main(path):
    X,y = load_csv(path)
    acts = ["step","linear","sigmoid","tanh","relu","softmax"]
    for a in acts:
        m = Perceptron(len(X[0]), activation=a, lr=0.1, seed=1)
        m.fit(X,y,epochs=20)        # (>10 exigido)
        print(f"{a:8s} -> acc={accuracy(m,X,y):.3f}")

if __name__=='__main__':
    if len(sys.argv)<2:
        print("Uso: python3 run_experiments.py <archivo.csv>"); exit(1)
    main(sys.argv[1])
