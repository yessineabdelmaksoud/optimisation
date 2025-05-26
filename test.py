import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Données du problème
X = np.array([
    [7, 5],    # x1, y=-1
    [10, 8],   # x2, y=-1
    [13, 9],   # x3, y=-1
    [8, 13],   # x4, y=1
    [6, 9],    # x5, y=1
    [2, 10]    # x6, y=1
])
y = np.array([-1, -1, -1, 1, 1, 1])

# Création du classifieur SVM avec noyau linéaire
clf = svm.SVC(kernel='linear', C=1000)  # Grand C pour être proche du SVM à marge dure
clf.fit(X, y)

# Extraction des paramètres du modèle
w = clf.coef_[0]  # Vecteur de poids
w0 = clf.intercept_[0]  # Biais

print(f"w = {w}")
print(f"w0 = {w0}")

# Calcul de la distance entre l'hyperplan et chaque point
def distance_point_hyperplan(x, w, b):
    return abs(np.dot(w, x) + b) / np.linalg.norm(w)

# Identifier les vecteurs de support
distances = [distance_point_hyperplan(x, w, w0) for x in X]
margin = min(distances)
print(f"Marge géométrique: {margin}")

# Indices des vecteurs de support (points les plus proches de l'hyperplan séparateur)
support_vectors_indices = [i for i, d in enumerate(distances) if abs(d - margin) < 1e-10]
print(f"Indices des vecteurs de support: {support_vectors_indices}")
print(f"Vecteurs de support: {X[support_vectors_indices]}")

# Visualisation
plt.figure(figsize=(10, 8))
# Tracer les points
colors = ['red' if label == -1 else 'blue' for label in y]
plt.scatter(X[:, 0], X[:, 1], c=colors, s=50)

# Nommer les points
for i, (x, y_val) in enumerate(zip(X, y)):
    plt.annotate(f'x{i+1}', (x[0], x[1]), xytext=(5, 5), textcoords='offset points')

# Tracer l'hyperplan séparateur
# Créer une grille pour tracer l'hyperplan
xx = np.linspace(0, 15, 100)
# w[0]*x + w[1]*y + w0 = 0 => y = (-w[0]*x - w0)/w[1]
yy = (-w[0] * xx - w0) / w[1]

plt.plot(xx, yy, 'k-')
# Tracer les marges
margin_plus = (-w[0] * xx - w0 + 1) / w[1]
margin_minus = (-w[0] * xx - w0 - 1) / w[1]
plt.plot(xx, margin_plus, 'k--')
plt.plot(xx, margin_minus, 'k--')

# Mettre en évidence les vecteurs de support
plt.scatter(X[support_vectors_indices, 0], X[support_vectors_indices, 1], 
           facecolors='none', edgecolors='green', s=100, linewidth=2)

plt.title('SVM avec noyau linéaire')
plt.xlabel('x1')
plt.ylabel('x2')
plt.grid(True)
plt.axis([0, 15, 0, 15])
plt.show()

# Calculer explicitement les équations
print("\nÉquation de l'hyperplan séparateur: {:.4f}*x1 + {:.4f}*x2 + {:.4f} = 0".format(w[0], w[1], w0))
print("Équation de la marge positive: {:.4f}*x1 + {:.4f}*x2 + {:.4f} = 0".format(w[0], w[1], w0-1))
print("Équation de la marge négative: {:.4f}*x1 + {:.4f}*x2 + {:.4f} = 0".format(w[0], w[1], w0+1))