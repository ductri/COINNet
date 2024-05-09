import pickle as pkl

def inspect(filename):
    with open(filename, 'rb') as i_f:
        annotations = pkl.load(i_f)
    print(f'Annotations {filename} by machine annotators on CIFAR10 train set, {annotations["machine_labels"].shape[0]} images')
    for i in range(annotations['machine_labels'].shape[1]):
        acc = (1.*(annotations['machine_labels'][:, i] == annotations['true_labels'])).mean()
        print(f'Annotator {i+1} acc: {acc}')

    agreement = (1.*(annotations['machine_labels'][:, 0] == annotations['machine_labels'][:, 1])).mean()
    print(f'Agreement between 1&2: {agreement}')

    agreement = (1.*(annotations['machine_labels'][:, 0] == annotations['machine_labels'][:, 2])).mean()
    print(f'Agreement between 1&3: {agreement}')

    agreement = (1.*(annotations['machine_labels'][:, 1] == annotations['machine_labels'][:, 2])).mean()
    print(f'Agreement between 2&3: {agreement}')

    print()


if __name__ == "__main__":
    filename = '/scratch/tri/shahana_outlier/data/cifar10_machine_annotations_case_5.pkl'
    inspect(filename)

