import bosonic
import autograd.numpy as np
import matplotlib.pyplot as plt
import functools

'''
    The matrix of a mzi component 
'''
def mzi(alpha, phi):
    return (1/2) * np.array([
        [np.exp(1j*phi) * (np.exp(1j*alpha)+1), np.exp(1j*alpha)-1],
        [np.exp(1j*phi) * (np.exp(1j*alpha)-1), np.exp(1j*alpha)+1]
    ]) # TODO: whether this is the right expression!

def embeddedMzi(alpha, phi, mode, numModes):
    U = np.concatenate([
        np.concatenate([np.eye(mode), np.zeros((mode, 2)), np.zeros((mode, numModes-mode-2))], axis=1),
        np.concatenate([np.zeros((2, mode)), mzi(alpha, phi), np.zeros((2, numModes-mode-2))], axis=1),
        np.concatenate([np.zeros((numModes-mode-2, mode)), np.zeros((numModes-mode-2, 2)), np.eye(numModes-mode-2)], axis=1)], axis=0)
    return U

'''
    Compute multi-photon unitary matrix from phase array.

    phaseArr: total lenth is m**2.
    The composition:
        - phaseArr[0:m*(m-1)/2] is alphas;
        - phaseArr[m*(m-1)/2:m*(m-1)] is phis;
        - the remaining part is phase shifts on each m modes.
'''
def getMultiPhotonU(numModes, numPhotons, phaseArr):

    '''
        Multiply matrices by order:
        (m-1,m-1),
        (m-2,m-2), (m-2,m-1),
        ...,
        (1,1), (1,2), ..., (1,m-1)
    ''' 
    orderedPairs = [
        (i,j) for i in np.arange(numModes-1,1-1,-1)
        for j in np.arange(i,numModes,1) ]

    def getIdx(i, j, numModes):
        return ((1/2*(2*numModes-i)*(i-1) + (j-i)))
    
    singleU = np.diag(np.exp(1j*phaseArr[numModes*(numModes-1):]))
    
    for pair in orderedPairs:
        singleU = np.dot(singleU,
                        embeddedMzi(
                            phaseArr[int(getIdx(pair[0],pair[1],numModes))],
                            phaseArr[int(getIdx(pair[0],pair[1],numModes) + numModes*(numModes-1)/2)],
                            pair[1]-1, numModes))
    
    multiU = bosonic.aa_phi(singleU, numPhotons)

    return multiU

'''
    Compute output probabilities wrt input states
'''
def getOutputProb(numModes, numPhotons, phaseArr, inputStates, isNormalized=False):
    fockU = getMultiPhotonU(numModes, numPhotons, phaseArr)
    inputIndices = np.ndarray(shape=(numPhotons, 1), dtype=int)
    fockBasis = np.array(bosonic.fock.basis(numPhotons, numModes))
    
    inputIndices = np.array([
        np.where((fockBasis == inputStates[i]).all(axis=1))
        for i in range(np.size(inputStates, 0))
    ], dtype=int)
    
    inputComp = np.zeros((fockBasis.shape[0], numPhotons), dtype='complex')

    for i in range(numPhotons):
        inputComp[inputIndices[i,0], i] = 1
        
    outputComp = np.dot(fockU, inputComp)
    outputProb = np.abs(outputComp)[inputIndices].T ** 2
    
    if isNormalized:
        outputProb = outputProb / np.sum(outputProb, axis=0) 
    
    return outputProb

'''
    Compute ideal CNOT gate truth table as an example.

    NOTE: THE PARAMETERS HERE DON'T WORK!
'''
def getIdealCNOTTruthTable():
    numModes = 7
    numPhotons = 4

    alphaArr = np.array([
        0, 0, 0, 0, 0, 0,
        0.992, 1.571, 4.957, 1.792, 0.,
        1.571, 0.992, 0., 0,
        1.571, 0., 2.226,
        3.142, 0.,
        0.])
    
    phiArr = np.array([
        0., 0., 0., 0., 0., 0.,
        0., 4.712, 4.544, 5.375, 3.816,
        0., 1.571, 2.188, 4.712,
        0., 5.498, 1.571,
        0., 3.142,
        0.]
    )
    
    phaseshiftArr = np.zeros(7)

    phaseArr = np.concatenate([alphaArr, phiArr, phaseshiftArr])

    inputStates = np.array([[1, 1, 0, 1, 0, 1, 0],
                            [1, 1, 0, 0, 1, 1, 0],
                            [0, 1, 1, 1, 0, 1, 0],
                            [0, 1, 1, 0, 1, 1, 0]])

    return getOutputProb(numModes, numPhotons, phaseArr, inputStates, isNormalized=True)
        
if __name__=='__main__':
    cnot_truth_table_plot = getIdealCNOTTruthTable().ravel()

    fig_cnot = plt.figure(figsize=(6,6))
    ax1 = fig_cnot.add_subplot(111, projection='3d')

    _x = np.arange(4)
    _y = np.arange(4)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.ravel(), _yy.ravel()
    labels = ['C0T0', 'C0T1', 'C1T0', 'C1T1']

    top = 1
    bottom = np.zeros_like(top)
    width = depth = 0.7

    ax1.bar3d(x, y, bottom, width, depth, cnot_truth_table_plot, shade=True)
    ax1.set_xticks(np.arange(4)+1/2)
    ax1.set_yticks(np.arange(4)+1/2)
    ax1.set_xticklabels(labels,fontsize=13)
    ax1.set_yticklabels(labels,fontsize=13)
    # ax1.set_xlabel('Input',fontsize=15,rotation=0)
    # ax1.set_ylabel('Output',fontsize=15,rotation=0)
    # ax1.yaxis._axinfo['label']['space_factor'] = 1.0
    ax1.xaxis.set_rotate_label(False) 
    ax1.yaxis.set_rotate_label(False) 
    ax1.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_title('Heralded CNOT Gate Truth Table',fontsize=16,y=1.05)
    
    plt.savefig('figures/cnot_truth_table_Yuxuan.png')