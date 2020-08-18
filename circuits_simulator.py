import bosonic
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import functools

'''
    The matrix of a mzi component 
'''
def mzi(alpha, phi):
    return np.array([
        [np.exp(1j*phi) * np.sin(alpha/2), np.exp(1j*phi) * np.cos(alpha/2)],
        [np.cos(alpha/2), -np.sin(alpha/2)]
    ]) # TODO: whether this is a good expression!

'''
    Generalized quantum circuit
'''
class Circuit:
    def __init__(self, numModes, alphaArr, phiArr):
        self.numModes = numModes
        self.alphaArr = alphaArr
        self.phiArr = phiArr
        
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

        self.singleU = np.linalg.multi_dot(
            [block_diag(
                np.eye(pair[1]-1),
                mzi(alphaArr[pair[0]-1,pair[1]-1], phiArr[pair[0]-1,pair[1]-1]),
                np.eye(numModes-1-pair[1]))
             for pair in orderedPairs])
    
    '''
        Parameters:
            numPhotons: number of photons
            inputStates: an array of numModes x "num of input states"
            isNormalized: whether normalize the probability to
        Return:
            outputProb: probabilities to find output in each input states,
                        an array of "num of input states" x "num of input states"
            
    '''
    def transform(self, numPhotons, inputStates, isNormalized=False):
        fockU = bosonic.aa_phi(self.singleU.astype('complex'), numPhotons)
        inputIndices = np.ndarray(shape=(numPhotons, 1), dtype=int)
        fockBasis = np.array(bosonic.fock.basis(numPhotons, self.numModes))
        
        inputIndices = np.array([
            np.where((fockBasis == inputStates[i]).all(axis=1))
            for i in range(np.size(inputStates, 0))
        ], dtype=int)
        
        inputComp = np.zeros((fockBasis.shape[0], numPhotons), dtype='complex')

        for i in range(numPhotons):
            inputComp[inputIndices[i,0], i] = 1
            
        outputComp = fockU.dot(inputComp)
        outputProb = np.abs(outputComp)[inputIndices].T ** 2
        
        if isNormalized:
            outputProb /= outputProb.sum(axis=0)
        
        return outputProb
        
class IdealCNOTCircuit(Circuit):
    '''
        alpha_arr, phi_arr is taken from Supplementary Material S3.4
    '''
    def __init__(self):
        alphaArr = np.array([
            [3.142, 3.142, 3.142, 3.142, 3.142, 3.142],
            [0., 0.992, 1.571, 4.957, 1.792, 0.],
            [0., 0., 1.571, 0.992, 0., 0],
            [0., 0., 0., 1.571, 0., 2.226],
            [0., 0., 0., 0., 3.142, 0.],
            [0., 0., 0., 0., 0., 0.]
        ])
        
        phiArr = np.array([
            [0., 0., 0., 0., 0., 0.],
            [0., 0., 4.712, 4.544, 5.375, 3.816],
            [0., 0., 0., 1.571, 2.188, 4.712],
            [0., 0., 0., 0., 5.498, 1.571],
            [0., 0., 0., 0., 0., 3.142],
            [0., 0., 0., 0., 0., 0.]
        ])
        
        
        super(IdealCNOTCircuit, self).__init__(7, alphaArr, phiArr)
    
    def getTruthTable(self):
        inputStates = np.array([[1, 1, 0, 1, 0, 1, 0],
                                [1, 1, 0, 0, 1, 1, 0],
                                [0, 1, 1, 1, 0, 1, 0],
                                [0, 1, 1, 0, 1, 1, 0]])
        
        return self.transform(numPhotons=4, inputStates=inputStates, isNormalized=True)
        
if __name__=='__main__':
    idealCnot = IdealCNOTCircuit()
    cnot_truth_table_plot = idealCnot.getTruthTable().ravel()

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