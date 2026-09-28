
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import multivariate_normal

# ============================================================
# Output directory
# ============================================================
OUTDIR = Path(__file__).resolve().parent

# ============================================================
# 0. Random seed
# ============================================================
SEED = 7
rng = np.random.default_rng(SEED)

# ============================================================
# 1. Parameters from ESL
# ============================================================
n_centers = 10
n_train_each = 100
n_test = 10000

I = np.eye(2)
Sigma = I / 5

BLUE = 0
ORANGE = 1

# ============================================================
# 2. Generate Gaussian centers
# ============================================================
blue_centers = rng.multivariate_normal(
    mean=[1, 0],
    cov=I,
    size=n_centers
)

orange_centers = rng.multivariate_normal(
    mean=[0, 1],
    cov=I,
    size=n_centers
)

# ============================================================
# 3. Generate observations from Gaussian mixture
# ============================================================
def generate_from_mixture(centers, n, rng):
    center_id = rng.integers(0, len(centers), size=n)
    X = np.empty((n, 2))

    for i, idx in enumerate(center_id):
        X[i] = rng.multivariate_normal(
            mean=centers[idx],
            cov=Sigma
        )

    return X

# ============================================================
# 4. Training data
# ============================================================
X_blue = generate_from_mixture(
    blue_centers,
    n_train_each,
    rng
)

X_orange = generate_from_mixture(
    orange_centers,
    n_train_each,
    rng
)

X_train = np.vstack([X_blue, X_orange])

y_train = np.concatenate([
    np.zeros(n_train_each, dtype=int),
    np.ones(n_train_each, dtype=int)
])

print("Training set shape:", X_train.shape)
print("BLUE:", np.sum(y_train == BLUE))
print("ORANGE:", np.sum(y_train == ORANGE))

# ============================================================
# 5. Test data
# ============================================================
n_test_blue = n_test // 2
n_test_orange = n_test - n_test_blue

X_test_blue = generate_from_mixture(
    blue_centers,
    n_test_blue,
    rng
)

X_test_orange = generate_from_mixture(
    orange_centers,
    n_test_orange,
    rng
)

X_test = np.vstack([
    X_test_blue,
    X_test_orange
])

y_test = np.concatenate([
    np.zeros(n_test_blue, dtype=int),
    np.ones(n_test_orange, dtype=int)
])

# ============================================================
# 6. Plotting grid
# ============================================================
margin = 1.0

all_points = np.vstack([
    X_train,
    blue_centers,
    orange_centers
])

x1_min = all_points[:, 0].min() - margin
x1_max = all_points[:, 0].max() + margin
x2_min = all_points[:, 1].min() - margin
x2_max = all_points[:, 1].max() + margin

xx, yy = np.meshgrid(
    np.linspace(x1_min, x1_max, 500),
    np.linspace(x2_min, x2_max, 500)
)

grid = np.c_[xx.ravel(), yy.ravel()]

def draw_training_points(ax):
    ax.scatter(
        X_blue[:, 0],
        X_blue[:, 1],
        c="royalblue",
        s=28,
        edgecolors="black",
        linewidths=0.3,
        label="BLUE"
    )

    ax.scatter(
        X_orange[:, 0],
        X_orange[:, 1],
        c="orange",
        s=28,
        edgecolors="black",
        linewidths=0.3,
        label="ORANGE"
    )

    ax.set_xlabel(r"$X_1$")
    ax.set_ylabel(r"$X_2$")
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.legend()

# ============================================================
# Figure 2.1 — Least Squares Classification
# ============================================================
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

grid_linear = linear_model.predict(grid)
Z_linear = grid_linear.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 7))

ax.contourf(
    xx,
    yy,
    Z_linear,
    levels=[-100, 0.5, 100],
    colors=["lightskyblue", "moccasin"],
    alpha=0.35
)

ax.contour(
    xx,
    yy,
    Z_linear,
    levels=[0.5],
    colors="black",
    linewidths=2
)

draw_training_points(ax)
ax.set_title("Figure 2.1 — Least Squares Classification")

fig.tight_layout()
fig.savefig(OUTDIR / "figure2_1.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 2.2 — 15-NN
# ============================================================
knn15 = KNeighborsClassifier(n_neighbors=15)
knn15.fit(X_train, y_train)

prob15 = knn15.predict_proba(grid)[:, 1]
Z15 = prob15.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 7))

ax.contourf(
    xx,
    yy,
    Z15,
    levels=[0, 0.5, 1],
    colors=["lightskyblue", "moccasin"],
    alpha=0.35
)

ax.contour(
    xx,
    yy,
    Z15,
    levels=[0.5],
    colors="black",
    linewidths=2
)

draw_training_points(ax)
ax.set_title("Figure 2.2 — 15-Nearest Neighbors")

fig.tight_layout()
fig.savefig(OUTDIR / "figure2_2.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 2.3 — 1-NN
# ============================================================
knn1 = KNeighborsClassifier(n_neighbors=1)
knn1.fit(X_train, y_train)

pred1 = knn1.predict(grid)
Z1 = pred1.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(7, 7))

ax.contourf(
    xx,
    yy,
    Z1,
    levels=[-0.5, 0.5, 1.5],
    colors=["lightskyblue", "moccasin"],
    alpha=0.35
)

ax.contour(
    xx,
    yy,
    Z1,
    levels=[0.5],
    colors="black",
    linewidths=1.5
)

draw_training_points(ax)
ax.set_title("Figure 2.3 — 1-Nearest Neighbor")

fig.tight_layout()
fig.savefig(OUTDIR / "figure2_3.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Figure 2.4 — Train/Test Error
# ============================================================
k_values = np.array([
    151,
    101,
    69,
    45,
    31,
    21,
    15,
    11,
    7,
    5,
    3,
    1
])

train_errors = []
test_errors = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    train_prediction = knn.predict(X_train)
    test_prediction = knn.predict(X_test)

    train_errors.append(
        np.mean(train_prediction != y_train)
    )

    test_errors.append(
        np.mean(test_prediction != y_test)
    )

train_errors = np.array(train_errors)
test_errors = np.array(test_errors)

N = len(X_train)
df = N / k_values

# Linear regression error
train_linear_score = linear_model.predict(X_train)
test_linear_score = linear_model.predict(X_test)

train_linear_prediction = (
    train_linear_score > 0.5
).astype(int)

test_linear_prediction = (
    test_linear_score > 0.5
).astype(int)

train_linear_error = np.mean(
    train_linear_prediction != y_train
)

test_linear_error = np.mean(
    test_linear_prediction != y_test
)

# Bayes classifier
def mixture_density(X, centers):
    density = np.zeros(len(X))

    for center in centers:
        density += multivariate_normal.pdf(
            X,
            mean=center,
            cov=Sigma
        )

    density /= len(centers)
    return density

blue_density = mixture_density(X_test, blue_centers)
orange_density = mixture_density(X_test, orange_centers)

bayes_prediction = (
    orange_density > blue_density
).astype(int)

bayes_error = np.mean(
    bayes_prediction != y_test
)

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(
    df,
    train_errors,
    "o-",
    color="royalblue",
    label="Train"
)

ax.plot(
    df,
    test_errors,
    "o-",
    color="orange",
    label="Test"
)

ax.axhline(
    bayes_error,
    color="purple",
    linewidth=2,
    label="Bayes"
)

ax.scatter(
    3,
    train_linear_error,
    s=130,
    marker="s",
    color="royalblue"
)

ax.scatter(
    3,
    test_linear_error,
    s=130,
    marker="s",
    color="orange"
)

ax.set_xlabel(r"Degrees of Freedom — $N/k$")
ax.set_ylabel("Misclassification Error")
ax.set_title("Figure 2.4 — Train and Test Misclassification Error")
ax.legend()
ax.grid(alpha=0.2)

ax_top = ax.twiny()
ax_top.set_xlim(ax.get_xlim())
ax_top.set_xticks(df)
ax_top.set_xticklabels(k_values)
ax_top.set_xlabel("k — Number of Nearest Neighbors")

fig.tight_layout()
fig.savefig(OUTDIR / "figure2_4.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ============================================================
# Save numerical summary
# ============================================================
summary_path = OUTDIR / "results.txt"

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(f"Training set shape: {X_train.shape}\n")
    f.write(f"BLUE: {np.sum(y_train == BLUE)}\n")
    f.write(f"ORANGE: {np.sum(y_train == ORANGE)}\n\n")
    f.write(f"Linear regression training error: {train_linear_error:.4f}\n")
    f.write(f"Linear regression test error: {test_linear_error:.4f}\n")
    f.write(f"Estimated Bayes error: {bayes_error:.4f}\n\n")
    f.write("kNN results\n")
    f.write("-" * 50 + "\n")
    f.write(f"{'k':>5}{'N/k':>10}{'Train Error':>15}{'Test Error':>15}\n")

    for k, d, tr, te in zip(
        k_values,
        df,
        train_errors,
        test_errors
    ):
        f.write(
            f"{k:5d}{d:10.2f}{tr:15.4f}{te:15.4f}\n"
        )

print()
print("Linear regression:")
print("Training error:", train_linear_error)
print("Test error:", test_linear_error)

print()
print("Estimated Bayes error:", bayes_error)

print()
print("Saved files:")
for p in sorted(OUTDIR.iterdir()):
    print(" -", p.name)
