from collections import Counter
import numpy as np
import torch
# from scipy.stats import mode


class TrainingDataWrapper(torch.utils.data.Dataset):
    def __init__(self, my_dataset, num_classes):
        self.my_dataset = my_dataset
        self.num_classes = num_classes

    def __len__(self):
        return len(self.my_dataset)

    def __getitem__(self, index):
        # (i, annotations, x, x_random, i_random), y = self.my_dataset[index]
        x, annotations, i, y = self.my_dataset[index]
        annotation = mode(annotations, keepdims=False)[0]
        return x, annotation


class TrainingDataFlattenWrapper(torch.utils.data.Dataset):
    def __init__(self, my_dataset, num_classes):
        self.my_dataset = my_dataset
        self.num_classes = num_classes
        self.N = len(self.my_dataset)
        _, annotations, _, _ = self.my_dataset[0]
        self.M = len(annotations)

    def __len__(self):
        return self.M*self.N

    def __getitem__(self, index):
        x_ind = index // self.M
        y_ind = index % self.M
        x, annotations, _, _ = self.my_dataset[x_ind]
        annotation = annotations[y_ind]
        return x, annotation


def convert_train_batch_majority(batch):
    x, annotations, _ = batch
    rng = np.random.default_rng()
    if annotations.ndim > 1:
        # annotations = rng.permuted(annotations.numpy(), axis=1)
        # annotation = torch.mode(torch.from_numpy(annotations), dim=1, keepdims=False)[0]


        annotations_majority_vote = []
        for row in annotations:
            row = row.cpu().numpy()
            np.random.shuffle(row)
            counter = Counter(row)
            if -1 in counter:
                counter[-1] = 0
            annotations_majority_vote.append(counter.most_common(1)[0][0])
        annotation = torch.from_numpy(np.array(annotations_majority_vote))
    else:
        annotation = annotations
    return x, annotation

