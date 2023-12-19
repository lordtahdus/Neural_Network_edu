import numpy as np

a = np.array([[1, 2, 3]]).transpose()
b = np.array([[4, 5, 6]]).transpose()
c = np.array([[7, 8, 9]]).transpose()
d = np.array([[10, 11, 12]]).transpose()

print(np.tile(a, 2))

training = [(a,b),(c,d),(a,b),(c,d)]

# for k in range(2):
#     mini_batches = [np.column_stack(pair) for pair in zip(*training[k:k+2])]

mini_batches = [
    [np.column_stack(pair) for pair in zip(*training[k:k+2])]
    for k in range(2)
]
print(np.sum(np.tile(a, 2), axis=1, keepdims=True))
print(len(mini_batches[0][0][0]))
