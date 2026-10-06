"""
Clasificator binar cu SVM si reducere dimensionala cu PCA
Fruits-360: "Apple Red 1" vs "Apple Red 2"

"""
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------------------------------------------------------------------
# Configurare
# ----------------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # folderul scriptului
except NameError:                                           # Jupyter / Colab
    BASE_DIR = os.getcwd()
CLASSES = ["Apple Red 1", "Apple Red 2"]        # numele folderelor = etichetele
IMG_EXTS = (".jpg", ".jpeg", ".png")
TEST_SIZE = 0.25        # 75% / 25%
IMG_SIZE = (100, 100)
KERNEL = "rbf"          
C_SVM = 1.0
N_COMP_2D = 2           # componentele folosite la SVM cu PCA 
RECON_K = (50, 10, 5)   
RANDOM_STATE = 0


# ----------------------------------------------------------------------------
# 1. Incarcarea datelor si vectorii caracteristici (slide 3)
# ----------------------------------------------------------------------------
def find_images(root):
    """Imagini din folderele cu numele claselor (cautare recursiva)."""
    paths, labels = [], []
    for label, cls in enumerate(CLASSES):
        folder = os.path.join(root, cls)
        if not os.path.isdir(folder):
            raise SystemExit(f"Nu gasesc folderul '{cls}' in {root}. "
                             f"Pune fisier.py in acelasi loc cu folderele.")
        for dirpath, _, filenames in os.walk(folder):
            for fname in sorted(filenames):
                if fname.lower().endswith(IMG_EXTS) and not fname.startswith("."):
                    paths.append(os.path.join(dirpath, fname))
                    labels.append(label)
    return paths, np.array(labels)


def load_images(paths):
    """Returneaza X (n, 30000) in [0, 1]."""
    X = []
    for path in paths:
        img = Image.open(path).convert("RGB").resize(IMG_SIZE)
        X.append(np.asarray(img, dtype=np.float64).ravel() / 255.0)
    return np.array(X)


paths, labels = find_images(BASE_DIR)
counts = np.bincount(labels, minlength=2)
print(f"Folder: {BASE_DIR}\nGasite: {CLASSES[0]} = {counts[0]}, {CLASSES[1]} = {counts[1]}")
if min(counts) < 2:
    raise SystemExit("Nu am gasit imagini in ambele foldere.")

X_all = load_images(paths)
X_train, X_test, y_train, y_test = train_test_split(
    X_all, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE)
print(f"Train: {X_train.shape}  ({np.bincount(y_train)} per clasa)")
print(f"Test : {X_test.shape}  ({np.bincount(y_test)} per clasa)")

# Scaling: scadem media setului de antrenare (aceeasi medie si pe test)
mean = X_train.mean(axis=0)
Xc_train = X_train - mean
Xc_test = X_test - mean


def show_samples(X, y, label, title, n=8):
    idx = np.where(y == label)[0][:n]
    fig, axes = plt.subplots(1, len(idx), figsize=(2 * len(idx), 2.4))
    for ax, i in zip(np.atleast_1d(axes), idx):
        ax.imshow(X[i].reshape(*IMG_SIZE, 3))
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()


show_samples(X_train, y_train, 0, "Mere din primul soi (Apple Red 1)")  
show_samples(X_train, y_train, 1, "Mere din al doilea soi (Apple Red 2)") 

# ----------------------------------------------------------------------------
# 2. PCA 
# ----------------------------------------------------------------------------
pca = PCA(svd_solver="full")

# Antrenarea PCA pe datele de antrenare
pca.fit(Xc_train)

# Valorile proprii
eigvals = pca.explained_variance_

# Vectorii proprii
V = pca.components_.T

# Varianta explicata
var_ratio = pca.explained_variance_ratio_

# Varianta cumulativa
cum_var = np.cumsum(var_ratio)


def project(Xc, k):
    """Proiecteaza datele pe primele k componente PCA."""
    return pca.transform(Xc)[:, :k]


# ----------------------------------------------------------------------------
# 3. Datele in 2D 
# ----------------------------------------------------------------------------
Z_train2 = project(Xc_train, N_COMP_2D)
Z_test2 = project(Xc_test, N_COMP_2D)

plt.figure(figsize=(6, 5))
for label, cls in enumerate(CLASSES):
    m = y_train == label
    plt.scatter(Z_train2[m, 0], Z_train2[m, 1], s=8, alpha=0.6, label=cls)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Datele in 2D (primele 2 componente principale)")
plt.legend()
plt.tight_layout()


# ----------------------------------------------------------------------------
# 4. Imagine reconstruita cu 50, 10 si 5 componente 
# ----------------------------------------------------------------------------
i0 = 0
fig, axes = plt.subplots(1, len(RECON_K) + 1, figsize=(3 * (len(RECON_K) + 1), 3.2))
axes[0].imshow(X_train[i0].reshape(*IMG_SIZE, 3))
axes[0].set_title("Originala")
for ax, k in zip(axes[1:], RECON_K):
    Vk = V[:, :k]
    x_hat = mean + (Xc_train[i0] @ Vk) @ Vk.T      # x_hat = mu + V_k V_k^T (x - mu)
    ax.imshow(np.clip(x_hat, 0, 1).reshape(*IMG_SIZE, 3))
    ax.set_title(f"{k} componente")
for ax in axes:
    ax.axis("off")
plt.tight_layout()


# ----------------------------------------------------------------------------
# 6. Matrice de confuzie normalizata (stilul din slide 16)
# ----------------------------------------------------------------------------
def plot_confusion_matrix(cm, classes, title="Normalized confusion matrix", ax=None):
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set_title(title)
    ticks = np.arange(len(classes))
    ax.set_xticks(ticks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels(classes)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.figure.tight_layout()
    return ax


# ----------------------------------------------------------------------------
# 7. SVM fara PCA 
# ----------------------------------------------------------------------------
svm_full = SVC(kernel=KERNEL, C=C_SVM, random_state=RANDOM_STATE)
svm_full.fit(Xc_train, y_train)
y_pred_full = svm_full.predict(Xc_test)
acc_full = accuracy_score(y_test, y_pred_full)
cm_full = confusion_matrix(y_test, y_pred_full)

# ----------------------------------------------------------------------------
# 8. SVM dupa PCA, primele 2 componente 
# ----------------------------------------------------------------------------
svm_pca = SVC(kernel=KERNEL, C=C_SVM, random_state=RANDOM_STATE)
svm_pca.fit(Z_train2, y_train)
y_pred_pca = svm_pca.predict(Z_test2)
acc_pca = accuracy_score(y_test, y_pred_pca)
cm_pca = confusion_matrix(y_test, y_pred_pca)

# ----------------------------------------------------------------------------
# 9. Clasarea celor doua soiuri in 2D: frontiera de decizie (slide 18)
# ----------------------------------------------------------------------------
pad = 0.1 * (Z_train2.max(axis=0) - Z_train2.min(axis=0))
x_min, y_min = Z_train2.min(axis=0) - pad
x_max, y_max = Z_train2.max(axis=0) + pad
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
zz = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(6.5, 5))
plt.contourf(xx, yy, zz, alpha=0.25, cmap=plt.cm.coolwarm, levels=[-0.5, 0.5, 1.5])
for label, cls in enumerate(CLASSES):
    m = y_test == label
    plt.scatter(Z_test2[m, 0], Z_test2[m, 1], s=12, label=f"{cls} (test)")
plt.scatter(*svm_pca.support_vectors_.T, s=40, facecolors="none", edgecolors="k",
            linewidths=0.6, label="Vectori suport")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clasarea celor doua soiuri de mere (SVM pe 2 componente)")
plt.legend()
plt.tight_layout()


# ----------------------------------------------------------------------------
# 10. Comparatie SVM fara PCA vs cu PCA (slide 19)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
plot_confusion_matrix(cm_full, CLASSES, f"Fara PCA - acuratete {acc_full * 100:.2f}%", axes[0])
plot_confusion_matrix(cm_pca, CLASSES, f"Cu PCA ({N_COMP_2D} comp.) - acuratete {acc_pca * 100:.2f}%", axes[1])
plt.tight_layout()


# ----------------------------------------------------------------------------
# 11. Bonus: acuratetea in functie de numarul de componente
# ----------------------------------------------------------------------------
ks = [2, 5, 10, 20, 50, 100, 200]
ks = [k for k in ks if k <= V.shape[1]]
accs = []
for k in ks:
    clf = SVC(kernel=KERNEL, C=C_SVM, random_state=RANDOM_STATE)
    clf.fit(project(Xc_train, k), y_train)
    accs.append(accuracy_score(y_test, clf.predict(project(Xc_test, k))))

plt.figure(figsize=(6.5, 4))
plt.semilogx(ks, [a * 100 for a in accs], "o-", label="SVM dupa PCA")
plt.axhline(acc_full * 100, color="r", ls="--", label="SVM fara PCA")
plt.xlabel("Numar de componente principale")
plt.ylabel("Acuratete (%)")
plt.title("Acuratete vs. numar de componente")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()