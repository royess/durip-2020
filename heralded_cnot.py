import numpy as np


def get_imperfect_cnot_gate(rel_err=0.01):
    r_ideal = np.sqrt([0, 0.227, 1 / 2, 1 / 2, 0.227, 1 / 2, 0.243, 1 / 2])
    r = [np.random.normal(loc=r_ideal[i], scale=rel_err * np.sqrt(r_ideal[i])) for i in range(len(r_ideal))]
    #TODO: Figure out why still getting some imaginary sqrt()'s in Jupyter NB (here, seems to be OK)
    for k in range(len(r)):
        if r[k] < 0 or r[k] >= 1:
            while r[k] < 0 or r[k] >= 1:
                r[k] = np.random.normal(loc=r_ideal[k], scale=rel_err * np.sqrt(r_ideal[k]))
            # print(r[k])
    t = np.sqrt([1 - r[i] ** 2 for i in range(len(r))])

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
    return gate


if __name__ == '__main__':

    psi_input = np.array([[1, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 1, 0],
                          [0, 1, 1, 0, 1, 1, 0]])

    # Ideal output states for input above
    psi_ideal = np.array([[1, 1, 0, 0, 1, 1, 0], [1, 1, 0, 1, 0, 1, 0], [0, 1, 1, 0, 1, 1, 0],
                          [0, 1, 1, 1, 0, 1, 0]])
    nsteps = 20
    nsamples = 500 # Average over many samples of random device errors to get smooth curves
    heralding_prob_samples = np.zeros((4, nsteps, nsamples))
    psi_out = np.zeros((4, 7, nsteps))
    unitary_fidelity_samples = np.zeros((nsteps, nsamples))

    for i in range(nsamples):
        for j in range(nsteps):
            imperfect_cnot = get_imperfect_cnot_gate(j * 0.006)
            ideal_cnot = get_imperfect_cnot_gate(0)
            psi_out[:, :, j] = np.array([np.matmul(imperfect_cnot, psi_input[i]) for i in range(len(psi_input))])
            heralding_prob_samples[:, j, i] = [np.abs(psi_out[i, 1, j] * psi_out[i, 5, j]) for i in
                                               range(len(psi_input))]
            unitary_fidelity_samples[j, i] = np.abs(
                np.trace(np.dot(np.matrix.getH(imperfect_cnot), ideal_cnot) / 7)) ** 2

    heralding_prob = np.mean(heralding_prob_samples, axis=2)
    unitary_fidelity = np.mean(unitary_fidelity_samples, axis=1)