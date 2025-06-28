"""
Generate simulated diffusion MRI signal
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import torch
from tqdm import tqdm
import scipy
import numpy as np
import nibabel as nib
import os
from tqdm import tqdm
class Reconstruction(torch.nn.Module):
    """Building Block for spherical harmonic convolution with a polar filter
    """

    def __init__(self,max_order,num_features):
        """Initialization.
        Args:
            polar_filter (:obj:`torch.Tensor`): [in_channel x S x L] Polar filter spherical harmonic coefficients
            polar_filter_inva (:obj:`torch.Tensor`): [in_channel x S x 1] Polar filter spherical harmonic coefficients
            shellSampling (:obj:`sampling.ShellSampling`): Itorchut sampling scheme
        """
        super(Reconstruction, self).__init__()
        self.max_order=max_order
        self.num_features=num_features

    def forward(self, x_equi_shc,iso,table,A):
        G_shape_1=A.shape[3]
        beta= torch.exp(-iso[:,:,:,None]*table[None,None,None])
       
        x_reconstructed_intra=torch.einsum('ijkgb,ijkb->ijkg', A,x_equi_shc[...,:-2])
        x_reconstructed_extra=torch.einsum('ijkg,ijkb->ijkg', beta,x_equi_shc[...,45:46])
        x_reconstructed_trap=torch.einsum('gb,ijkb->ijkg', torch.ones(G_shape_1,1).to(x_equi_shc.device),x_equi_shc[...,46:47])
        x_reconstructed=x_reconstructed_intra+x_reconstructed_extra+x_reconstructed_trap
        return x_reconstructed

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
    b1 = torch.abs(b[None,None,None] * (lambda1[:,:,:,None,None] - lambda2))#60,1
    b2 = torch.erf(torch.sqrt(b1))
    a0 = torch.sqrt(torch.pi / b1) * b2
    a2 = np.sqrt(torch.pi) * b2 / (2 * b1 ** 1.5) - 1.0 / (b1 * torch.exp(b1))
    a4 = -(3 + 2 * b1) / (torch.exp(b1) * 2 * b1 ** 2) + (3 * np.sqrt(torch.pi) * b2) / (4 * b1 ** 2.5)
    a6 = 15 * np.sqrt(torch.pi) * b2 / (8 * b1 ** 3.5) - (4 * b1 ** 2 + 10 * b1 + 15) / (4 * b1 ** 3 * torch.exp(b1))
    a8 = 105 * np.sqrt(torch.pi) * b2 / (16 * b1 ** 4.5) - (2 * b1 * (2 * b1 * (2 * b1 + 7) + 35) + 105) / (8 * b1 ** 4 * torch.exp(b1))
    a10 = 945 * np.sqrt(torch.pi) * b2 / (32 * b1 ** 5.5) - (2 * b1 * (2 * b1 * (2 * b1 * (2 * b1 + 9) + 63) + 315) + 945) / (16 * b1 ** 5 * torch.exp(b1))
    a12 = 10395 * np.sqrt(torch.pi) * b2 / (64 * b1 ** 6.5) - (2 * b1 * (2 * b1 * (2 * b1 * (4 * b1 ** 2 + 22 * b1 + 99) + 693) + 3465) + 10395) / (32 * b1 ** 6 * torch.exp(b1))
    a14 = 135135 * np.sqrt(torch.pi) * b2 / (128 * b1 ** 7.5) - (64 * b1 ** 6 + 416 * b1 ** 5 + 2288 * b1 ** 4 + 10296 * b1 ** 3 + 36036 * b1 ** 2 + 90090 * b1 + 135135) / (64 * b1 ** 7 * torch.exp(b1))
    a16 = 2027025 * np.sqrt(torch.pi) * b2 / (256 * b1 ** 8.5) - (128 * b1 ** 7 + 960 * b1 ** 6 + 6240 * b1 ** 5 + 34320 * b1 ** 4 + 154440 * b1 ** 3 + 540540 * b1 ** 2 + 1351350 * b1 + 2027025) / (128 * b1 ** 8 * torch.exp(b1))

    G = [0,0,0,0,0]
    G[0] = a0 # l = 0
    G[1] = 1.5 * a2 - 0.5 * a0 # l= 2
    G[2] = 35 / 8 * a4 - 15 / 4 * a2 + 3 / 8 * a0 # l = 4
    G[3] = (231*a6 - 315*a4 + 105*a2 - 5*a0)/16.0 # l = 6
    G[4] = (6435 * a8 - 12012 * a6 + 6930 * a4 - 1260 * a2 + 35 * a0) / 128.0 # l = 8
    # G[5] = (46189 * a10 - 109395 * a8 + 90090 * a6 - 30030 * a4 + 3465 * a2 - 63 * a0) / 256.0 # l = 10
    # G[6] = (676039 * a12 - 1939938 * a10 + 2078505 * a8 - 1021020 * a6 + 225225 * a4 - 18018 * a2 + 471 * a0) / 1024.0 # l = 12
    # G[7] = (5014575 * a14 - 16900975 * a12 + 24709287 * a10 - 14549535 * a8 + 4849845 * a6 - 765765 * a4 + 45045 * a2 - 429 * a0) / 2048.0 # l = 14
    # G[8] = (300540195 * a16 - 1163381400 * a14 + 1825305300 * a12 - 1487285800 * a10 + 669278610 * a8 - 162954792 * a6 + 19399380 * a4 - 875160 * a2 + 6435 * a0) / 32768.0 # l = 16
    G = [g * 2 * torch.pi * torch.exp(-b * lambda2) for g in G]

    # H1 = \dfrac{\partial G}{\partial \lambda_1}
    H1 = [0,0,0,0,0]
    H1[0] = a2 # l = 0
    H1[1] = (3 * a4 - a2) / 2 # l = 2
    H1[2] = (35 * a6 - 30 * a4 + 3 * a2) / 8 # l = 4
    H1[3] = (471 * a8 - 315 * a6 + 105 * a4 - 5 * a2) / 16 # l = 6
    H1[4] = (6435 * a10 - 12012 * a8 + 6930 * a6 - 1260 * a4 + 35 * a2) / 128 # l = 8
    # H1[5] = (461prev_nf * a12 - 109395 * a10 + 90090 * a8 - 30030 * a6 + 3465 * a4 - 63 * a2) / 256.0 # l = 10
    # H1[6] = (676039 * a14 - 1939938 * a12 + 2078505 * a10 - 1021020 * a8 + 225225 * a6 - 18018 * a4 + 471 * a2) / 1024 # l = 12

    H1 = [-2 * b * torch.pi * torch.exp(-b * lambda2) * h1 for h1 in H1]
    #-2 * b * torch.pi * torch.exp(-b * lambda2) * H1
    # H2 = \dfrac{\partial G}{\partial \lambda_2}
    H2 = [0,0,0,0,0]
    H2 = [-b * g -h1 for g, h1 in zip(G, H1)]
    G=torch.stack(G,dim=0)
    H1=torch.stack(H1,dim=0)
    H2=torch.stack(H2,dim=0)
    return G, H1, H2

def matrixG_MultiShell(lambda1,lambda2,B,bval,device,maxOrder):
    Gmtx = torch.zeros(lambda1.shape[0],lambda1.shape[1],lambda1.shape[2],B.shape[0],B.shape[1]).to(device) # FOD channel的总数
    bval_set = bval#torch.unique(bval) # 删去重复的元素.符合要求的只有B值为3000
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

def sort_signal(signal,signal1, indices):
    sorted_indices1 = np.argsort(signal1)[::-1] 
    sorted_indices = np.argsort(signal)[::-1] 
    return signal[sorted_indices],signal1[sorted_indices]

def get_valid_voxel(data,real_dmri,fod_coeffs,slice_len):
    while True:
        x, y,z = np.random.randint(data.shape[0]), np.random.randint(data.shape[1]), np.random.randint(slice_len)
        
        signal = data[x, y, z, :]
        signal1 = real_dmri[x, y, z, :]
        fod=fod_coeffs[x, y, z, :]
        if len(np.where(fod>0.1)[0])>3:  
            return x, y, z, signal,signal1

if __name__ == '__main__':
    newsimu=1
    slice_len=20
    start_slice=50
    end_slice=start_slice+slice_len
    device='cpu'
    
    iso1,iso2=0.0099,0.0099 
    land1,land2=0.0015,0.0019 
    lambda2 = torch.tensor(0.0).to(device)  
   
    #random.seed(42)
    real_dmri_path = "./data/HCP/100206/100206_dti_B0.nii.gz"
    tis_data_path= './data/HCP/100206/TissueMap.nii.gz'
    
    real_dmri0 = nib.load(real_dmri_path)
    real_dmri=real_dmri0.get_fdata().astype(np.float32)  
    real_dmri=real_dmri[:,:,start_slice:end_slice]
    tis_dmri = torch.from_numpy(nib.load(tis_data_path).get_fdata().astype(np.float32)[:,:,start_slice:end_slice]).to('cuda')

    lambda1 = torch.tensor(np.random.uniform(land1, land2,size=real_dmri.shape[:3])).to(device)
    simulated_dmri_path = os.path.join('./data/Simulation/100206',f'tis_l{int(land1*10000)}l{int(land2*10000)}i{int(iso1*10000)}i{int(iso2*10000)}.nii.gz')
    train_list = './Tools/Gen/gen_simu/HCP_simu.txt'
    save_dir = './data/Simulation/'
    os.makedirs(save_dir, exist_ok=True)
    train_vol_names = np.loadtxt(train_list, dtype=str, ndmin=1)
    config_options = [
    'two_fibers_90',
    'two_fibers_60',
    'two_fibers_45',
    'three_fibers_60',
    'single_fiber'
]
    subject_list = []
    for line in train_vol_names:
        parts = line[0].split()
        first_path = parts[0]
        subject_id = first_path.split('/')[5]
        subject_list.append(subject_id)
        fod_name='./data/HCP/100206/100206_FOD_WholeVolume.nii.gz'#os.path.join(subject_save_dir, f"{subject_id}_simu_fod_
        fod_coeffs=nib.load(fod_name).get_fdata(dtype=np.float32)[:,:,start_slice:end_slice]
        fod=fod_coeffs.copy()
        if subject_id=='100206':
            break
    if newsimu==False:
        
        x_reconstructed_np0 = nib.load(simulated_dmri_path)
        x_reconstructed_np=x_reconstructed_np0.get_fdata().astype(np.float32).squeeze()
    else:
        
    
        volume_shape = (16,16,16)
        max_order, num_features=8,45
       
        for i, subject_id in tqdm(enumerate(subject_list), total=len(subject_list), desc="Processing subjects"):
            print(f"Processing Subject {subject_id}...")
            
            subject_save_dir = os.path.join(save_dir, subject_id)
            os.makedirs(subject_save_dir, exist_ok=True)
            
            save_dti_recon_path = os.path.join(subject_save_dir, f'{subject_id}_simu_dmri.nii.gz')
            table_path = train_vol_names[i][2]
            table = np.loadtxt(table_path).astype(np.float32)
            bpo=table[...,-1]>100
            table=table[bpo]
            table[:,-1]=np.array([float(round(float(val) / 500) * 500) for val in table[:,-1]])
            Y=matrixB(table,8)
            table=torch.from_numpy(table).float().to(device)
            G=matrixG_MultiShell(lambda1,lambda2,Y,table[:,3],device,8)
            G_array = G.detach().cpu().numpy()

            np.save(os.path.join(subject_save_dir, f'{subject_id}_Y.npy'), Y)
            print('Saved:',os.path.join(subject_save_dir, f'{subject_id}_Y.npy'))
            np.save(os.path.join(subject_save_dir, f'{subject_id}_G.npy'), G_array)
            print('Saved:',os.path.join(subject_save_dir, f'{subject_id}_G.npy'))
            
            reconstruction_model = Reconstruction(max_order, num_features)
            reconstruction_model = reconstruction_model.to(device)
            
            fod_coeffs=torch.from_numpy(fod_coeffs).float().to(device)
            
            ex=tis_dmri[...,1:3]
            all=torch.cat((fod_coeffs,ex),dim=-1)
            Y1=torch.from_numpy(Y).float().to(device)
            iso = tis_dmri[...,3]

            
            x_reconstructed = reconstruction_model(all, iso, table[:,-1], Y1*G)
            x_reconstructed_np = x_reconstructed.squeeze().cpu().numpy()
            
            nib.save(nib.Nifti1Image(x_reconstructed_np, real_dmri0.affine), simulated_dmri_path)
            break
            
        print("All subject volumes have been generated and saved.")

    gradient_table_path = "./data/HCP/100206/GradientTable.txt"
    simulated_dmri = x_reconstructed_np

    
    gradient_table = np.loadtxt(gradient_table_path).transpose()
    B0_ind = np.where(gradient_table[-1] > 100)[0]  
    gradient_table = gradient_table[..., B0_ind]

    b1000_indices = np.where((gradient_table[-1] >= 900) & (gradient_table[-1] <= 1100))[0]
    b2000_indices = np.where((gradient_table[-1] >= 1900) & (gradient_table[-1] <= 2100))[0]
    b3000_indices = np.where((gradient_table[-1] >= 2900) & (gradient_table[-1] <= 3100))[0]

    for k in range(20):
        simu_x, simu_y, simu_z, simulated_signal, real_signal = get_valid_voxel(simulated_dmri,real_dmri,fod,slice_len)
        print(f"Selected voxel for simulated signal: ({simu_x}, {simu_y}, {simu_z})")
        print(f"Selected voxel for real signal: ({simu_x}, {simu_y}, {simu_z})")

        simulated_signal_b1000 = simulated_signal[b1000_indices]
        simulated_signal_b2000 = simulated_signal[b2000_indices]
        simulated_signal_b3000 = simulated_signal[b3000_indices]

        real_signal_b1000 = real_signal[b1000_indices]
        real_signal_b2000 = real_signal[b2000_indices]
        real_signal_b3000 = real_signal[b3000_indices]

        simulated_signal_b1000, real_signal_b1000 = sort_signal(simulated_signal_b1000,real_signal_b1000 ,b1000_indices)
        simulated_signal_b2000, real_signal_b2000 = sort_signal(simulated_signal_b2000,real_signal_b2000 ,b2000_indices)
        simulated_signal_b3000, real_signal_b3000 = sort_signal(simulated_signal_b3000,real_signal_b3000 ,b3000_indices)

        fig, ax = plt.subplots(figsize=(3, 6))

        ax.plot(range(len(simulated_signal_b1000)), simulated_signal_b1000, 'k-', label='Simulated Signal', marker='o')
        ax.plot(range(len(simulated_signal_b2000)), simulated_signal_b2000, 'k-', marker='o')#v
        ax.plot(range(len(simulated_signal_b3000)), simulated_signal_b3000, 'k-', marker='o')#x

        ax.plot(range(len(real_signal_b1000)), real_signal_b1000, 'r-', label='Real Signal: b=1000', marker='o')
        ax.plot(range(len(real_signal_b2000)), real_signal_b2000, 'g-', label='Real Signal: b=2000', marker='v')
        ax.plot(range(len(real_signal_b3000)), real_signal_b3000, 'b-', label='Real Signal: b=3000', marker='x')

        ax.set_xlabel('Gradient Directions (Sorted)', fontsize=12)
        ax.set_ylabel('Signal Intensity', fontsize=12)
        ax.set_title('Simulated vs Real Signal (Different b-values)', fontsize=14)
        ax.legend(fontsize=10)

        output_path=os.path.join(save_dir,f'v_{k}_tis_l{int(land1*10000)}l{int(land2*10000)}i{int(iso1*10000)}i{int(iso2*10000)}.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
