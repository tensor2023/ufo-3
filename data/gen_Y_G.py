import os
import nibabel as nib
import numpy as np
import sys
from scipy.ndimage import binary_erosion, binary_dilation
from tqdm import tqdm
import scipy
from torch import erf, pi,sqrt
import torch
import random
def load_txt(filename):
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    with open(filename, 'r') as file:
        lines = file.readlines()
    vol = np.array([line.strip().split() for line in lines])
    return vol

def legendre_function(l,x):
    m_range = range(l+1)
    y=[]
    for i, m in enumerate(m_range):
        if m > l:
            continue
        y.append(scipy.special.lpmv(m, l, x))
        a=1
    out=np.stack(y)
    return out

def matrixB(A, maxOrder):
    """
    Compute the real spherical harmonic basis matrix B given input directions A.

    Args:
        A (np.ndarray): Direction matrix of shape (N, 3), where each row is a unit vector.
        maxOrder (int): Maximum even spherical harmonic order.

    Returns:
        B (np.ndarray): Real spherical harmonics basis matrix of shape (N, num_coeffs),
                        where num_coeffs depends on maxOrder.
    """

    rows, _ = A.shape  # e.g., 95 directions, each with 3 coordinates

    # Compute theta (elevation angle) for each vector
    thetavec = np.arctan2(np.sqrt(A[:, 0]**2 + A[:, 1]**2), A[:, 2])
    # arctan2 returns the polar angle in radians, range [0, π]

    # Compute phi (azimuthal angle) for each vector
    phivec = np.arctan2(A[:, 1], A[:, 0])
    # arctan2 returns the azimuth angle in radians, range [-π, π]

    order = np.arange(0, maxOrder + 1, 2)  # e.g., [0, 2, 4, 6, 8]
    R = (order + 1) * (order + 2) // 2     # Total number of SH coefficients up to each order

    P = np.zeros((rows, int(R[-1])), dtype=np.float32)  # Legendre polynomial components
    T = np.zeros((rows, int(R[-1])), dtype=np.float32)  # Trigonometric components
    scale = np.zeros(int(R[-1]), dtype=np.float32)      # Normalization factors

    # First-order SH term (l = 0)
    P[:, 0] = 1
    T[:, 0] = np.ones(rows, dtype=np.float32)
    scale[0] = np.sqrt(1 / (4.0 * np.pi))

    indx = 0
    for i in range(1, len(order)):
        l = order[i]
        temp = legendre_function(l, np.cos(thetavec)).T

        # Concatenate negative m, m = 0, and positive m components
        P[:, int(R[i - 1]):int(R[i])] = np.concatenate((temp[:, l+1:0:-1], temp), axis=1)

        # Construct trigonometric part using sin(mϕ) and cos(mϕ)
        m_range = np.arange(1, l + 1)
        sin_part = np.sin(np.outer(phivec, m_range[::-1]))
        cos_part = np.cos(np.outer(phivec, m_range))
        T[:, int(R[i - 1]):int(R[i])] = np.concatenate((sin_part, np.ones((rows, 1)), cos_part), axis=1)

        # Compute normalization scale for SH basis
        tempv = np.zeros(l, dtype=np.float32)
        for m in range(l):
            tempv[m] = np.sqrt(((2 * l + 1) / (2 * np.pi)) / np.prod(np.arange(l - m, l + m + 2)))

        scale[int(R[i - 1]):int(R[i])] = np.concatenate((
            tempv[::-1],  # negative m
            np.array([np.sqrt((2 * l + 1) / (4.0 * np.pi))]),  # m = 0
            tempv         # positive m
        ), axis=0)

    # Final SH basis matrix
    B = np.tile(scale, (rows, 1)) * P * T
    return B

def SHCoefficient(lambda1, lambda2, b):
    # Ensure inputs are in float32
    lambda1 = lambda1.double()
    lambda2 = lambda2.double()
    b = b.double()
    # maxl=8, lambda1 and lambda2 are eigenvalues at a voxel
    b1 = b * (lambda1 - lambda2)
    b2 = erf(torch.sqrt(b1))
    
    # Temp int e^{-b1*t^2}t^{l}, l= 0,2,4,6,8,10,12,14,16,18
    # using Wolframalpha to calculate a0, a2, a4, a6, a8 and G(:)
    a0 = sqrt(pi / b1) * b2
    a2 = np.sqrt(pi) * b2 / (2 * b1 ** 1.5) - 1.0 / (b1 * torch.exp(b1))
    a4 = -(3 + 2 * b1) / (2 * b1 ** 2 * torch.exp(b1)) + (3 * np.sqrt(pi) * b2) / (4 * b1 ** 2.5)
    a6 = 15 * np.sqrt(pi) * b2 / (8 * b1 ** 3.5) - (4 * b1 ** 2 + 10 * b1 + 15) / (4 * b1 ** 3 * torch.exp(b1))
    a8 = 105 * np.sqrt(pi) * b2 / (16 * b1 ** 4.5) - (2 * b1 * (2 * b1 * (2 * b1 + 7) + 35) + 105) / (8 * b1 ** 4 * torch.exp(b1))
    a10 = 945 * np.sqrt(pi) * b2 / (32 * b1 ** 5.5) - (2 * b1 * (2 * b1 * (2 * b1 * (2 * b1 + 9) + 63) + 315) + 945) / (16 * b1 ** 5 * torch.exp(b1))
    a12 = 10395 * np.sqrt(pi) * b2 / (64 * b1 ** 6.5) - (2 * b1 * (2 * b1 * (2 * b1 * (4 * b1 ** 2 + 22 * b1 + 99) + 693) + 3465) + 10395) / (32 * b1 ** 6 * torch.exp(b1))
    a14 = 135135 * np.sqrt(pi) * b2 / (128 * b1 ** 7.5) - (64 * b1 ** 6 + 416 * b1 ** 5 + 2288 * b1 ** 4 + 10296 * b1 ** 3 + 36036 * b1 ** 2 + 90090 * b1 + 135135) / (64 * b1 ** 7 * torch.exp(b1))
    a16 = 2027025 * np.sqrt(pi) * b2 / (256 * b1 ** 8.5) - (128 * b1 ** 7 + 960 * b1 ** 6 + 6240 * b1 ** 5 + 34320 * b1 ** 4 + 154440 * b1 ** 3 + 540540 * b1 ** 2 + 1351350 * b1 + 2027025) / (128 * b1 ** 8 * torch.exp(b1))

    # Initialize G with zeros
    G = [0,0,0,0,0,0,0,0,0,0]
    G[0] = a0  # l = 0
    G[1] = 1.5 * a2 - 0.5 * a0  # l = 2
    G[2] = 35 / 8 * a4 - 15 / 4 * a2 + 3 / 8 * a0  # l = 4
    G[3] = (231 * a6 - 315 * a4 + 105 * a2 - 5 * a0) / 16.0  # l = 6
    G[4] = (6435 * a8 - 12012 * a6 + 6930 * a4 - 1260 * a2 + 35 * a0) / 128.0  # l = 8
    G[5] = (46189 * a10 - 109395 * a8 + 90090 * a6 - 30030 * a4 + 3465 * a2 - 63 * a0) / 256.0  # l = 10
    G[6] = (676039 * a12 - 1939938 * a10 + 2078505 * a8 - 1021020 * a6 + 225225 * a4 - 18018 * a2 + 231 * a0) / 1024.0  # l = 12
    G[7] = (5014575 * a14 - 16900975 * a12 + 22309287 * a10 - 14549535 * a8 + 4849845 * a6 - 765765 * a4 + 45045 * a2 - 429 * a0) / 2048.0  # l = 14
    G[8] = (300540195 * a16 - 1163381400 * a14 + 1825305300 * a12 - 1487285800 * a10 + 669278610 * a8 - 162954792 * a6 + 19399380 * a4 - 875160 * a2 + 6435 * a0) / 32768  # l = 16
    
    # Apply exponential scaling to G
    G = [g * 2 * torch.pi * torch.exp(-b * (lambda2)) for g in G]#G * 2 * pi * torch.exp(-b * lambda2)
    # H1 = torch.zeros_like(G)
    # H1[0] = a2  # l = 0
    # H1[1] = (3 * a4 - a2) / 2  # l = 2
    # H1[2] = (35 * a6 - 30 * a4 + 3 * a2) / 8  # l = 4
    # H1[3] = (231 * a8 - 315 * a6 + 105 * a4 - 5 * a2) / 16  # l = 6
    # H1[4] = (6435 * a10 - 12012 * a8 + 6930 * a6 - 1260 * a4 + 35 * a2) / 128  # l = 8
    # H1[5] = (46189 * a12 - 109395 * a10 + 90090 * a8 - 30030 * a6 + 3465 * a4 - 63 * a2) / 256  # l = 10
    # H1[6] = (676039 * a14 - 1939938 * a12 + 2078505 * a10 - 1021020 * a8 + 225225 * a6 - 18018 * a4 + 231 * a2) / 1024  # l = 12

    # H1 = -2 * b * pi * torch.exp(-b * lambda2) * H1

    # # H2 = \dfrac{\partial G}{\partial \lambda_2}
    # H2 = -b * G - H1
    
    return G, 0,0


def matrixG_MultiShell(lambda1,lambda2,B,bval,device,maxOrder):
    Gmtx = torch.zeros(B.shape[0],B.shape[1]).to(device) 
    bval_set = bval
    bval_set =bval_set.view(bval.shape[0],1)
    g, _, _ = SHCoefficient(lambda1, lambda2, bval_set)
   
    g = g[:(maxOrder//2+1)]
    Gmtx[...,0:1] = g[0]
    if maxOrder>0:
        Gmtx[...,1:6] = g[1] # l=2 
    if maxOrder>2:
        Gmtx[...,6:15] = g[2] # l=4
    if maxOrder>4:
        Gmtx[...,15:28] = g[3] # l=6 
    if maxOrder>6:
        Gmtx[...,28:45] = g[4] # l=8
    if maxOrder>8:
        Gmtx[...,45:66] = g[5] # l=10
    if maxOrder>10:
        Gmtx[...,66:91] = g[6] # l=12
    if maxOrder>12:
        Gmtx[...,91:120] = g[7] # l=14
    if maxOrder>14:
        Gmtx[...,120:153] = g[8] # l=16
    
    return Gmtx

def ero(a,brain_image):
    structure_element = np.ones((a,a,a)).astype(np.float32)
    eroded_image = binary_erosion(brain_image, structure=structure_element)
    output2 = binary_dilation(eroded_image, structure=structure_element).astype(np.float32)
    return output2

def DTI_B_matrix(bvecs):
    '''Design matrix for DTI computation'''

    if bvecs.shape[0] == 3:
        bvecs = bvecs.T

    B = np.zeros((bvecs.shape[0], 6))

    B[:, 0] = bvecs[:, 0] * bvecs[:, 0]
    B[:, 1] = bvecs[:, 1] * bvecs[:, 1]
    B[:, 2] = bvecs[:, 2] * bvecs[:, 2]
    B[:, 3] = 2 * bvecs[:, 0] * bvecs[:, 1]
    B[:, 4] = 2 * bvecs[:, 0] * bvecs[:, 2]
    B[:, 5] = 2 * bvecs[:, 1] * bvecs[:, 2]

    return B

def B0_norm (GradTable, dti, mask):
    B0_Ind = np.where(GradTable[3]<100)[0] 
    B0 = np.mean(dti[...,B0_Ind], axis=-1) 
    
    inds_bvals = np.where(GradTable[-1] > 1250)[0]
    dti=dti[...,inds_bvals]
    
    GradTable=GradTable[...,inds_bvals]
  
    S_Max = np.max(dti, axis=-1)
    mask_index=np.where((mask > 0) & (S_Max < 2 * B0) & (B0>1))
    
    mask_zero=np.zeros_like(mask)
    mask_zero[mask_index]=mask[mask_index]
    dti_mask = np.zeros_like(dti)
    dti_mask[mask_index] = dti[mask_index]

    B0[B0 == 0] = 1
    B0=np.expand_dims(B0, axis=-1)
    dti_B0 = dti_mask/B0 
    dti=dti_B0.copy()
    dti_B0[dti_B0<0]=0
    dti_B0[dti_B0>1]=1
   
    return dti_B0,dti,GradTable.transpose()
def process_subject(table):
    device='cuda'
    Y=matrixB(table[:,:3],8)
    table=torch.from_numpy(table).double().to(device)
  
    Y=torch.from_numpy(Y).double().to(device)
    shape=[1]
    
    lambda1=torch.full(shape, 0.0017).to(device)
    lambda2=torch.full(shape, 0.0).to(device)
    G=matrixG_MultiShell(lambda1,lambda2,Y,table[:,3],device,10)
    Y=Y.detach().cpu().numpy()
    G=G.detach().cpu().numpy()

    return Y,G

def set_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  

if __name__ == '__main__':
    set_seed(42)
    subject_list = []
    train_list = './Tools/process/data/CHCP_FOD_cos.txt'
    subject_output_dir = './data/CHCP/3025/1737_46/' 
    train_vol_names = np.loadtxt(train_list, dtype=str, ndmin=1)

    subject_list = []
  
    for line in train_vol_names:
        parts = line[0].split()  
        first_path = parts[0]
        subject_id = first_path.split('/')[6]
        subject_list.append(subject_id)
        print('Adding:', subject_id)
    i=0
    for i, subject_id in tqdm(enumerate(subject_list), total=len(subject_list), desc="Processing subjects"):
        print(subject_id)
        table_path = './data/ISMRM2015/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/table.npy'#train_vol_names[i][2]
        
        table = np.load(table_path).astype(np.float32)#95,4
        # bval=np.full_like(table[:, 0:1], 1737).astype(np.float32)
        # table=np.concatenate((table,bval),axis=-1)
        if table.shape[1]!=4:
            table=table.transpose()
        B0_Ind = np.where(table[:,3]>100)[0] 
        table=table[B0_Ind]
       
        Y,G= process_subject(table)  
        
        np.save('./data/ISMRM2015/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/Y.npy', Y)
        print('Saved:',os.path.join(subject_output_dir, f'{subject_id}_Y.npy'))
        np.save('./data/ISMRM2015/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/G.npy', G)
        print('Saved:',os.path.join(subject_output_dir, f'{subject_id}_G.npy'))