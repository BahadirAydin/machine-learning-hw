import pickle
import matplotlib

matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import PCA
import seaborn as sns

dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))


def tsne_visualization(data, n_components, perplexity, title):
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    reduced_data = tsne.fit_transform(data)
    cmap = plt.colormaps["viridis"]
    plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=cmap(np.arange(len(data)) / len(data)),
        cmap=cmap,
    )

    plt.xlabel("t-SNE1")
    plt.ylabel("t-SNE2")
    plt.title(
        "t-SNE Visualization ({} components, perplexity {})".format(
            n_components, perplexity
        )
    )
    plt.colorbar()
    plt.savefig(title)
    plt.close()


def umap_visualization(
    data,
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    title="",
    filename="umap.png",
):
    fit = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric,
    )
    u = fit.fit_transform(data)
    cmap = plt.colormaps["viridis"]
    plt.scatter(
        u[:, 0],
        u[:, 1],
        c=cmap(np.arange(len(data)) / len(data)),
        cmap=cmap,
    )

    plt.title(
        "UMAP Visualization ({} neighbors , components {})".format(
            n_neighbors, n_components
        )
    )
    plt.colorbar()
    plt.savefig(filename)
    plt.close()


def pca_visualization(data, n_components, title):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    cmap = plt.colormaps["viridis"]
    plt.scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        c=cmap(np.arange(len(data)) / len(data)),
        cmap=cmap,
    )

    plt.xlabel("PCA1")
    plt.ylabel("PCA2")
    plt.title("PCA Visualization ({} components)".format(n_components))
    plt.colorbar()
    plt.savefig(title)
    plt.close()


tsne_visualization(dataset1, 2, 30, "tsne-2comps-dataset1.png")
tsne_visualization(dataset2, 2, 30, "tsne-2comps-dataset2.png")

tsne_visualization(dataset1, 3, 30, "tsne-3comps-dataset1.png")
tsne_visualization(dataset2, 3, 30, "tsne-3comps-dataset2.png")


pca_visualization(dataset1, 2, "pca-2comps-dataset1.png")
pca_visualization(dataset2, 2, "pca-2comps-dataset2.png")


# Apply UMAP with 2 components
umap_visualization(
    data=dataset1,
    n_components=2,
    title="UMAP 2 components Dataset 1",
    filename="umap-2comps-dataset1.png",
)
umap_visualization(
    data=dataset2,
    n_components=2,
    title="UMAP 2 components Dataset 2",
    filename="umap-2comps-dataset2.png",
)

umap_visualization(
    data=dataset1,
    n_components=3,
    title="UMAP 2 components Dataset 1",
    filename="umap-3comps-dataset1.png",
)
umap_visualization(
    data=dataset2,
    n_components=3,
    title="UMAP 2 components Dataset 2",
    filename="umap-3comps-dataset2.png",
)


print("Done!")
