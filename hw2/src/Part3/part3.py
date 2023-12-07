import pickle
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

dataset = pickle.load(open("../data/part3_dataset.data", "rb"))

distance_functions = ["cosine", "euclidean"]
linkage_functions = ["single", "complete"]
k_values = [2, 3, 4, 5]


# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)


results = []
avg_silhouette_scores = []
for f in distance_functions:
    for l in linkage_functions:
        silhoutte_scores = []
        best_silhoutte_score = 0
        best_k = -1
        best_clustering = None
        k_values_for_plot = []
        silhouette_scores_for_plot = []

        for k in k_values:
            print(
                "\nRunning for distance function: "
                + f
                + ", linkage function: "
                + l
                + ", k: "
                + str(k)
                + "\n"
            )
            clustering = AgglomerativeClustering(
                n_clusters=k, metric=f, linkage=l, compute_distances=True
            ).fit(dataset)
            score = silhouette_score(dataset, clustering.labels_, metric=f)
            silhoutte_scores.append(score)

            k_values_for_plot.append(k)
            silhouette_scores_for_plot.append(score)

            if score > best_silhoutte_score:
                best_silhoutte_score = score
                best_clustering = clustering
                best_k = k

            results.append(
                {
                    "distance_function": f,
                    "linkage_function": l,
                    "k": k,
                    "score": score,
                }
            )
            print("Silhouette score: " + str(score))

        # Plot and save k vs silhouette score
        plt.plot(k_values_for_plot, silhouette_scores_for_plot, marker="o")
        plt.title("K vs Silhouette Score (Distance=" + f + ", Linkage=" + l + ")")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.savefig("part3_k_vs_silhouette_" + f + "_" + l + ".png")
        plt.close()

        print("Best silhouette score: " + str(best_silhoutte_score))
        avg_silhouette_scores.append(best_silhoutte_score)

        # Plot and save dendrogram
        plt.title(
            "Hierarchical Clustering Dendrogram (Best k="
            + str(best_k)
            + ", linkage="
            + l
            + ")"
        )
        plot_dendrogram(best_clustering)
        plt.savefig("part3_dendrogram_" + f + "_" + l + "_" + str(best_k) + ".png")
        plt.close()

print("Highest average silhouette score: " + str(max(avg_silhouette_scores)))
