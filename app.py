import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris

st.set_page_config(page_title="Decision Tree Playground", layout="wide")
st.title("Decision Tree Playground")
st.write("Play with Decision Tree hyperparameters and see how the model changes!")

iris = load_iris()
X = iris.data[:, :2]
y = iris.target
feature_names = iris.feature_names[:2]
class_names = iris.target_names

st.sidebar.header("Decision Tree Parameters")

criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy", "log_loss"])
splitter = st.sidebar.selectbox("Splitter", ["best", "random"])
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 10, 1)
max_features = st.sidebar.selectbox("Max Features", [None, "sqrt", "log2"])
max_leaf_nodes = st.sidebar.slider("Max Leaf Nodes (None = unlimited)", 2, 20, 10)
if max_leaf_nodes == 10:
    max_leaf_nodes = None

clf = DecisionTreeClassifier(
    criterion=criterion,
    splitter=splitter,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
    max_leaf_nodes=max_leaf_nodes,
    random_state=42
)
clf.fit(X, y)

st.subheader("1. Decision Boundary (2D)")
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.3)
ax.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", s=40)
ax.set_xlabel(feature_names[0])
ax.set_ylabel(feature_names[1])
ax.set_title("Decision Regions")
st.pyplot(fig)

st.subheader("2. Decision Tree Structure")
depth = clf.get_depth()
fig_height = max(6, depth * 2.5)
font_size = max(6, 12 - depth)
fig, ax = plt.subplots(figsize=(18, fig_height))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,
    rounded=True,
    fontsize=font_size
)
st.pyplot(fig)

score = clf.score(X, y)
st.success(f"Training Accuracy: **{score:.3f}**")
