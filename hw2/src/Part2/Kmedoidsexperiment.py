from Kmedoids import KMemoids
import pickle
import numpy as np
import matplotlib.pyplot as plt


dataset1 = pickle.load(open("../data/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../data/part2_dataset_2.data", "rb"))


def run_kmedoids(dataset, k):
    min_loss = np.inf
    for _ in range(10):
        kmedoids = KMemoids(dataset, k)
        _, _, loss = kmedoids.run()
        min_loss = min(min_loss, loss)
    return min_loss


def calculate_average_loss(dataset, k):
    losses = []
    for _ in range(10):
        losses.append(run_kmedoids(dataset, k))
    return np.mean(losses), calculate_confidence_interval(losses)


def calculate_confidence_interval(data):
    mean = np.mean(data)
    std = np.std(data)
    confidence_interval = 1.96 * std / np.sqrt(len(data))
    return (round(mean - confidence_interval, 5), round(mean + confidence_interval, 5))


def run(dataset, title):
    k_range = range(2, 11)
    average_losses = []
    for k in k_range:
        print(f"Running K-Medoids for k = {k}")
        avg_loss, conf_interval = calculate_average_loss(dataset, k)
        print(f"Average loss: {avg_loss}, Confidence interval: {conf_interval}")
        average_losses.append(avg_loss)

    # Plot average loss vs k
    plt.plot(k_range, average_losses)
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Average loss")
    plt.title("k vs Average Loss for K-Medoids")
    plt.grid()
    plt.tight_layout()
    plt.savefig(title, dpi=300)
    plt.close()


# Run elbow method for each dataset
print("### Running for dataset 1...###\n")
run(dataset1, "medoid_k_vs_loss1.png")
print("\n### Running for dataset 2... ###\n")
run(dataset2, "medoid_k_vs_loss2.png")
