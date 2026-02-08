import numpy as np

X = np.random.randn(300, 10)
y = np.array([0]*200 + [1]*80 + [2]*20)


np.random.seed(0)
labeled_idx = np.random.choice(len(y), 30, replace=False)
unlabeled_idx = np.array(list(set(range(len(y))) - set(labeled_idx)))


def uncertainty_sampling(unlabeled_indices):
    return np.random.rand(len(unlabeled_indices))


rounds = 3
query_size = 10

for r in range(rounds):
    scores = uncertainty_sampling(unlabeled_idx)
    query_indices = unlabeled_idx[np.argsort(scores)[-query_size:]]

    labeled_idx = np.concatenate([labeled_idx, query_indices])
    unlabeled_idx = np.array(list(set(unlabeled_idx) - set(query_indices)))

    print(f"Round {r+1}: Labeled samples = {len(labeled_idx)}")
