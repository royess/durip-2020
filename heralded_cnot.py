import numpy as np
import matplotlib.pyplot as plt

# Why didn't I do this sooner?

# Input states for (c, t) = (0,0) through (1,1)
psi_input = np.array([[1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 1, 0, 1, 0], [0, 1, 1, 1, 0, 1, 0],
                      [0, 1, 1, 0, 1, 1, 0]])

# Ideal output states for input above
psi_ideal = np.array([[1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 1, 0],
                      [0, 1, 1, 1, 0, 1, 0]])

r = np.sqrt([0, 0.227, 1 / 2, 1 / 2, 0.227, 1 / 2, 0.243, 1 / 2])
r = [np.random.normal(loc=r[i], scale=0.00 * r[i]) for i in range(len(r))]
t = np.sqrt([1 - r[i] ** 2 for i in range(len(r))])

def compute_overlap(p1, p2):
    product = p1 * p2
    return product if product < 1 else 1

gate = np.array([[1, 0, 0, 0, 0, 0, 0],
                 # c0
                 [0, r[1], -r[2] * t[1], -r[3] * t[2] * t[1], t[3] * t[2] * t[1], 0, 0],
                 # |1>
                 [0, -t[1] * r[5], -r[2] * r[1] * r[5] - t[2] * r[4] * t[5],
                  -r[3] * t[2] * r[1] * r[5] + r[3] * r[2] * r[4] * t[5],
                  +t[3] * t[2] * r[1] * r[5] - t[3] * r[2] * r[4] * t[5], t[4] * t[5], 0],
                 # c1
                 [0, -t[1] * t[5] * t[6] * t[7], r[2] * r[1] * t[5] * t[6] * t[7] - t[2] * r[4] * r[5] * t[6] * t[7],
                  -r[3] * r[2] * r[4] * r[5] * t[6] * t[7] - r[3] * t[2] * r[1] * t[5] * t[6] * t[7] + t[3] * r[7],
                  t[3] * t[2] * r[1] * t[5] * t[6] * t[7] + r[3] * r[7] + t[3] * r[2] * r[4] * r[5] * t[6] * t[7],
                  -t[4] * r[5] * t[6] * t[7], -r[6] * t[7]],
                 # t1
                 [0, t[1] * t[5] * t[6] * r[7], r[2] * r[1] * t[5] * t[6] * r[7] - t[2] * r[4] * r[5] * t[6] * r[7],
                  +r[3] * t[2] * r[1] * t[5] * t[6] * r[7] + r[3] * r[2] * r[4] * r[5] * t[6] * r[7] + t[3] * t[7],
                  -t[3] * t[2] * r[1] * t[5] * t[6] * r[7] - t[3] * r[2] * r[4] * r[5] * t[6] * r[7] + r[3] * t[7],
                  t[4] * r[5] * t[6] * r[7], r[6] * r[7]],
                 # t0
                 [0, 0, t[2] * t[4], -r[3] * r[2] * t[4], t[3] * r[2] * t[4], r[4], 0],
                 # |1>
                 [0, t[1] * t[5] * r[6], r[2] * r[1] * t[5] * r[6] - t[2] * r[4] * r[5] * r[6],
                  r[3] * r[2] * r[4] * r[5] * r[6] + r[3] * t[2] * r[1] * t[5] * r[6],
                  -t[3] * t[2] * r[1] * t[5] * r[6] - t[3] * r[2] * r[4] * r[5] * r[6],
                  t[4] * r[5] * r[6], -t[6]]])
# |0>

psi_out = np.array([np.matmul(gate, psi_input[j]) for j in range(len(psi_input))])
heralding_prob = [np.abs(psi_out[i, 1] * psi_out[i, 5]) for i in range(len(psi_out))]
truth_table = np.square([[psi_out[i, 0] * psi_out[i, 4], psi_out[i, 0] * psi_out[i, 3], psi_out[i, 1] * psi_out[i, 4],
                         psi_out[i, 1] * psi_out[i, 3]] for i in range(len(psi_out))])

