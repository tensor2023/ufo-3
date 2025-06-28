"""
Load CHCP diffusion MRI data
"""
import scipy.io as scio
import os.path
from scipy.ndimage import binary_erosion, binary_dilation
import torchio as tio
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))))
from data.base_dataset import BaseDataset
import numpy as np
import nibabel as nib
import scipy
import torch
import random
from dipy.reconst.shm import sf_to_sh
from dipy.core.sphere import Sphere
def nullable_string(val):
    if not val:
        return None
    return val

def nullable_int(val):
    if not val:
        return None
    return int(val)

def flip_axis_to_match_HCP_space(data, affine):
    """
    Checks if affine of the image has the same signs on the diagonal as HCP space. If this is not the case it will
    invert the sign of the affine (not returned here) and invert the axis accordingly.
    If displayed in an medical image viewer the image will look the same, but the order by the data in the image
    array will be changed.

    This function is inspired by the function with the same name in TractSeg
    """
    newAffine = affine.copy()  # could be returned if needed
    flipped_axis = []

    if affine[0, 0] > 0:
        flipped_axis.append("x")
        data = data[::-1, :, :]
        newAffine[0, 0] = newAffine[0, 0] * -1
        newAffine[0, 3] = newAffine[0, 3] * -1

    if affine[1, 1] < 0:
        flipped_axis.append("y")
        data = data[:, ::-1, :]
        newAffine[1, 1] = newAffine[1, 1] * -1
        newAffine[1, 3] = newAffine[1, 3] * -1

    if affine[2, 2] < 0:
        flipped_axis.append("z")
        data = data[:, :, ::-1]
        newAffine[2, 2] = newAffine[2, 2] * -1
        newAffine[2, 3] = newAffine[2, 3] * -1

    return data, newAffine, flipped_axis

def load_txt(filename):
    """
    Loads a file in nii, nii.gz, mgz, npz, or npy format. If input file is not a string,
    returns it directly (allows files preloaded in memory to be passed to a generator)

    Parameters:
        filename: Filename to load, or preloaded volume to be returned.
        np_var: If the file is a npz (compressed numpy) with multiple variables,
            the desired variable can be specified with np_var. Default is 'vol'.
        add_batch_axis: Adds an axis to the beginning of the array. Default is False.
        add_feat_axis: Adds an axis to the end of the array. Default is False.
        pad_shape: Zero-pad the array to a target shape. Default is None.
        resize: Volume resize factor. Default is 1
        ret_affine: Additionally returns the affine transform (or None if it doesn't exist).
    """
    if isinstance(filename, str) and not os.path.isfile(filename):
        raise ValueError("'%s' is not a file." % filename)

    with open(filename, 'r') as file:
        lines = file.readlines()
    vol = np.array([line.strip().split() for line in lines])

    return vol

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

def ero(a,brain_image):
    structure_element = np.ones((a,a,a)).astype(np.float32)
    eroded_image = binary_erosion(brain_image, structure=structure_element)
    output2 = binary_dilation(eroded_image, structure=structure_element).astype(np.float32)
    return output2
def randomSelectBvecs(b_vecs, nbvecs, b_vals, bval, cond_num):
    '''Randomly select b-vectors, subject to cond_num constraint'''

    inds_bvals = np.logical_and(b_vals > bval-100, b_vals < bval+100)
    b_vecs_to_chose_from = b_vecs[:, inds_bvals[0]]
    cond_num_curr = 10
    while cond_num_curr>cond_num:
        random_dirs = random.sample(range(0, np.shape(b_vecs_to_chose_from)[1]), nbvecs)
        b_vecs_selected_curr = b_vecs_to_chose_from[:, random_dirs]
        B_matrix = DTI_B_matrix(b_vecs_to_chose_from[:, random_dirs])
        cond_num_curr = np.linalg.cond(B_matrix, p=2)

    return b_vecs_selected_curr, random_dirs

def B0_norm (GradTable, dti, mask,prev):#GradTable[95,4]
    a=15
    mask=ero(a,mask)
    mask[mask<0.25]=0
    mask[mask>1]=0
    B0_Ind = np.where(GradTable[:,3]<100)[0] 
    S0 = np.mean(dti[...,B0_Ind], axis=-1) 

    nbvecs=10#prev
    cond_num=3.5
    bval=1500
    cond_num_curr = 8
    rad=100

    inds_bvals = np.logical_and(GradTable[:,-1] > bval-rad, GradTable[:,-1] < bval+rad)
    dti=dti[...,inds_bvals]
    GradTable=GradTable[inds_bvals]
    b_vecs_to_chose_from = GradTable[:,:3]
    if np.shape(b_vecs_to_chose_from)[0] < nbvecs:
        return np.zeros_like(dti),0,0
    while cond_num_curr>cond_num:
        #print(inds_bvals,b_vecs_to_chose_from.shape,b_vecs.shape)
        random_dirs = random.sample(range(0, np.shape(b_vecs_to_chose_from)[0]), nbvecs)
        b_vecs_selected_curr = b_vecs_to_chose_from[random_dirs]
        #b_vals_selected_curr = b_vals_to_chose_from[random_dirs]
        B_matrix = DTI_B_matrix(b_vecs_selected_curr)
        cond_num_curr = np.linalg.cond(B_matrix, p=2)
    dti=dti[...,random_dirs]
    GradTable=GradTable[random_dirs]
    
    S_Max = np.max(dti, axis=-1) 
    mask_index=np.where((mask > 0) & (S_Max < 2 * S0) & (S0>1))
    
    mask_zero=np.zeros_like(mask)
    mask_zero[mask_index]=1
    dti_mask = np.zeros_like(dti)
    dti_mask[mask_index] = dti[mask_index]
    del mask

    S0[S0 == 0] = 1
    S0=np.expand_dims(S0, axis=-1)
    dti_S0 = dti_mask/S0 
    dti_S0[dti_S0<0]=0
    dti_S0[dti_S0>1]=1
    return dti_S0,mask_zero,GradTable 


def ReadObjShape(fname):
    fid = open(fname,'r')
    info = fid.readline().strip().split()
    shape_type=info[0]
    if shape_type == 'P': 
        a = info[1:]
        num_pts = int(a[5])
        pt_coordinates = []
        pt_normals = []
        pt_colors = []
        triangles = []
        tmp=[]
        for i in range(num_pts):
            pt_coordinates.append(list(map(float, fid.readline().split())))
        for i in range(num_pts):
            pt_normals.append(list(map(float, fid.readline().split())))
        num_triangles = int(fid.readline().strip())
        color_type = float(fid.readline().strip().split()[0])
        if color_type == 2:
            for i in range(num_pts):
                pt_colors.append(list(map(float, fid.readline().split())))
        elif color_type == 0:
            pt_colors = list(map(float, fid.readline().split()))
        for i in range(640):
            tmp.append(list(map(int, fid.readline().split())))
        for i in range(1920):
            triangles.append(list(map(int, fid.readline().split())))
        pt_colors=np.stack(pt_colors, axis=0)
        pt_coordinates=np.stack(pt_coordinates, axis=0)
        pt_normals=np.stack(pt_normals, axis=0)
        triangles=np.hstack(triangles).reshape(num_triangles,3)
        triangles=np.hstack(triangles).reshape(-1,3)
        tmp=np.hstack(tmp)
    if shape_type == 'L': #line
        a = list(map(float, fid.readline().split()))
        num_pts = int(a[1])
        pt_coordinates = []
        for i in range(num_pts):
            pt_coordinates.append(list(map(float, fid.readline().split())))

    fid.close()
    return pt_coordinates, pt_normals, triangles, pt_colors

def MeshNeiboringVertices(coord, trg):
    """
    Given a triangular mesh, find the vertices in the one-ring neighborhood of
    each vertex. The output is a list of sets.
    
    Args:
        coord: numpy array of vertex coordinates, shape (N, 3)
        trg: numpy array of triangle indices, shape (M, 3)

    Returns:
        VertNbrList: list of sets, each set contains the indices of neighboring vertices
    """

    Vert2FaceMap, _ = BuildMeshNbrhood(coord, trg)  # Call BuildMeshNbrhood to get vertex-to-face map
    Vert2FaceList = [None] * len(coord)  # Initialize an empty list

    for i in range(len(coord)):
        Vert2FaceList[i] = Vert2FaceMap[i, 1:Vert2FaceMap[i, 0]]  # Exclude the count at position 0

    VertNbrList = [None] * len(coord)  # Initialize a list to store neighbors for each vertex

    for i in range(len(coord)):
        a = trg[Vert2FaceList[i] - 1, :]  # Get triangles connected to vertex i
        VertNbrList[i] = set(np.setdiff1d(a, i))  # Exclude vertex i itself and convert to set

    return VertNbrList  # Return a list of sets. Note: indices are not sorted, and indexing is zero-based.

def matload(filname,points):
    data = scio.loadmat(filname)
    cell = data['ConstraintSet']
    if points==200:
        return cell[0][128] 
    if points==216:
        return cell[0][139] 
    if points==257:
        return cell[0][-2] 
    if points==258:
        return cell[0][-1] 
    if points==69:
        return cell[0][36] 
    if points==100:
        return cell[0][59] 
    return cell[0][-1]
    
def pre_generator(dir_path,points):
    coord,_,trg,_ = ReadObjShape(os.path.join(dir_path,'data/sphere5120.obj'))
    ConstraintSet=matload(os.path.join(dir_path,'data/ConstraintSet.mat'),points).squeeze()
    VertNbrCellArr = MeshNeiboringVertices(coord,trg)
    return coord,ConstraintSet-1,VertNbrCellArr,trg

def BuildMeshNbrhood(coord, trg):
    """
    Build vertex-to-face and triangle-to-neighbor triangle maps for a triangular mesh.

    Args:
        coord: Tensor of shape (num_vertices, 3), the vertex coordinates.
        trg: Tensor of shape (num_faces, 3), each row contains indices of triangle vertices.

    Returns:
        Vert2FaceMap: Tensor of shape (num_vertices, max_faces_per_vertex), 
                      where the first column stores the count, and the rest store face indices.
        TrgNbrMap: Tensor of shape (num_faces, 3), where each row contains neighboring triangle indices.
                   For each triangle, the i-th column indicates the neighbor that shares the i-th edge.
    """

    Vert2FaceMap = torch.zeros((len(coord), 20), dtype=int)  # Initialize zero matrix
    Vert2FaceMap[:, 0] = 1  # First column stores the count of associated faces

    for i in range(len(trg)):  # Loop over all triangles
        for j in range(3):  # Loop over triangle's three vertices
            Vert2FaceMap[trg[i, j], 0] += 1  # Increment neighbor count
            Vert2FaceMap[trg[i, j], Vert2FaceMap[trg[i, j], 0] - 1] = i + 1  # Store face index (1-based)

    TrgNbrMap = torch.zeros((len(trg), 3), dtype=int)  # Initialize triangle-to-triangle neighbor map

    for i in range(len(trg)):  # Loop over all triangles
        # Neighbor sharing vertices 0 and 1
        for j in range(1, Vert2FaceMap[trg[i, 0], 0] + 1):
            for k in range(1, Vert2FaceMap[trg[i, 1], 0] + 1):
                if Vert2FaceMap[trg[i, 0], j] != i and Vert2FaceMap[trg[i, 0], j] == Vert2FaceMap[trg[i, 1], k]:
                    TrgNbrMap[i, 2] = Vert2FaceMap[trg[i, 0], j]
                    break
            if TrgNbrMap[i, 2] > 0:
                break

        # Neighbor sharing vertices 0 and 2
        for j in range(1, Vert2FaceMap[trg[i, 0], 0] + 1):
            for k in range(1, Vert2FaceMap[trg[i, 2], 0] + 1):
                if Vert2FaceMap[trg[i, 0], j] != i and Vert2FaceMap[trg[i, 0], j] == Vert2FaceMap[trg[i, 2], k]:
                    TrgNbrMap[i, 1] = Vert2FaceMap[trg[i, 0], j]
                    break
            if TrgNbrMap[i, 1] > 0:
                break

        # Neighbor sharing vertices 1 and 2
        for j in range(1, Vert2FaceMap[trg[i, 1], 0] + 1):
            for k in range(1, Vert2FaceMap[trg[i, 2], 0] + 1):
                if Vert2FaceMap[trg[i, 1], j] != i and Vert2FaceMap[trg[i, 1], j] == Vert2FaceMap[trg[i, 2], k]:
                    TrgNbrMap[i, 0] = Vert2FaceMap[trg[i, 1], j]
                    break
            if TrgNbrMap[i, 0] > 0:
                break

    return Vert2FaceMap, TrgNbrMap

class CHCPDataset(BaseDataset):
    """
    This dataset class can load the processed HCP dataset for angular super resolution in terms of fiber orientation
    distribution function computed by constrained spherical deconvolution.

    The processed HCP dataset is supposed to have FOD images generated by low angular resolution (typical single shell single
     tissue 32 gradient directions) DWI and original DWI (multi shell 1000, 2000, 3000, totally 288 gradient directions)

     This class is a heavy memory version, which loads every samples into the memory for further use.
    """

    def __init__(self, opt,dataset_list,prev_nf):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(CHCPDataset, self).__init__(opt)
        
        self.size_3d_patch = opt['size_3d_patch']
        self.margin = int(self.opt['size_3d_patch']/2)
        self.dataset_num_samples_per_data = 100
        self._fod_ids = []
        self.fod_info = []
        self.load_hcp(dataset_list=dataset_list,prev=prev_nf)
        self.prepare()
        self.transform = tio.Compose([
    tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),           
    tio.RandomNoise(mean=0, std=0.1),         
    tio.RandomGamma(log_gamma=(-0.3, 0.3)),                     
    tio.RandomBiasField(coefficients=0.5),                     
    tio.RandomBlur(std=(0, 2)),                             
    tio.RandomGhosting(num_ghosts=2, intensity=0.5),         
    tio.RescaleIntensity(out_min_max=(0, 1)),                  
])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains training data and corresponding golden standard
            fodlar (tensor)  -- the spherical harmonics coefficients of the fODF obtained from LAR DWI
            #fodgt (tensor)   -- the spherical harmonics coefficients of the fODF obtained from original HCP DWI
        """
        #print('start')
        fod_sample = self.fod_info[index % self.num_fod]#torch

        #fodlr6 = fod_sample['fodlr6']
        fodlr = fod_sample['fodlr']
        
        table=fod_sample['table']
        mask=fod_sample['mask']
        final_index = fod_sample['index_mask']
        index_length = fod_sample['index_length']

        idx = np.random.randint(index_length)
        x = final_index[0, idx]
        y = final_index[1, idx]
        z = final_index[2, idx]

        x_start = x - self.margin
        x_end = x_start + self.opt['size_3d_patch']
        y_start = y - self.margin
        y_end = y_start + self.opt['size_3d_patch']
        z_start = z - self.margin
        z_end = z_start + self.opt['size_3d_patch']
       
        fodlr_3D_patches = fodlr[x_start:x_end, y_start:y_end, z_start:z_end, :].transpose(3, 0, 1, 2)
        mask_3D_patches = mask[x_start:x_end, y_start:y_end, z_start:z_end]
        
        Y=fod_sample['Y']
        n=fod_sample['n']
        G=fod_sample['G']
        fodgt = fod_sample['fodgt']
        if len(fodgt)>1:
            fodgt_3D_patches = fodgt[x_start:x_end, y_start:y_end, z_start:z_end, :].transpose(3, 0, 1, 2)
            data_dict = {'fodlr': fodlr_3D_patches,
                    'fodgt': fodgt_3D_patches,
                    'table': table,
                    'mask' : mask_3D_patches.transpose(2, 0, 1),
                    'Y' : Y,
                    'G' : G,
                    'nside16sh8':n,
                    }
        else: 
            fodlr_tensor = torch.from_numpy(fodlr_3D_patches) 
            fodlr_image = tio.ScalarImage(tensor=fodlr_tensor)  
            augmented_fodlr_image = self.transform(fodlr_image)  
            augmented_fodlr_3D_patches = augmented_fodlr_image.tensor.squeeze().numpy()  
            fodgt_3D_patches='0'
            data_dict = {'fodlr': augmented_fodlr_3D_patches,
                        'fodgt': fodgt_3D_patches,
                        'table': table,
                        'mask' : mask_3D_patches.transpose(2, 0, 1),
                        'Y' : Y,
                        'G' : G,
                        'nside16sh8':n,
                        }
        return data_dict
    
    def load_hcp(self, dataset_list,prev):
        """Load the processed hcp dataset
        dataset_dir: The root directory of the processed hcp dataset.
        """
        #subject_ids = os.listdir(dataset_path)

        # Add fod samples
        if len(dataset_list)<5:#val
            for i, subject_id in enumerate(dataset_list):
                print('Loading {0}'.format(subject_id))#/home/gaoxq/data/HCP/100206/Diffusion/data
                self.add_fod_sample(fod_id=i,
                                    prev=prev,
                                    fodlr_path=os.path.join(subject_id[0]),
                                    fodlr_path6=os.path.join(subject_id[0]),
                                    fsl_5ttgen_mask_path=os.path.join(subject_id[1]),
                                    table_path=os.path.join(subject_id[2]),
                                    fodgt_path=os.path.join(subject_id[3]),
                                    mean_path=os.path.join(subject_id[4]),
                                    std_path=os.path.join(subject_id[5]),
                                    Y_path=os.path.join(subject_id[6]),
                                    G_path=os.path.join(subject_id[7]),
                                    n_path=os.path.join(subject_id[8]),
                                    subject_id=subject_id)

        else:
            for i, subject_id in enumerate(dataset_list):
                print('Loading {0}'.format(subject_id))#/home/gaoxq/data/HCP/100206/Diffusion/data
                self.add_fod_sample(fod_id=i,
                                    prev=prev,
                                    fodlr_path=os.path.join(subject_id[0]),
                                    fodlr_path6=os.path.join(subject_id[8]),
                                    fsl_5ttgen_mask_path=os.path.join(subject_id[1]),
                                    table_path=os.path.join(subject_id[2]),
                                    fodgt_path='0',
                                    mean_path=os.path.join(subject_id[4]),
                                    std_path=os.path.join(subject_id[5]),
                                    Y_path=os.path.join(subject_id[6]),
                                    G_path=os.path.join(subject_id[7]),
                                    n_path=os.path.join(subject_id[8]),
                                    subject_id=subject_id)

    def add_fod_sample(self,prev,fod_id, fodlr_path,fodlr_path6, fsl_5ttgen_mask_path, table_path,fodgt_path,mean_path,
                                std_path,Y_path,G_path,n_path,subject_id=None):
        fodlr = nib.load(fodlr_path).get_fdata(dtype=np.float32)
        fsl5ttgen_mask = nib.load(fsl_5ttgen_mask_path).get_fdata(dtype=np.float32)
        
        table=np.load(table_path)#10,4
        if table.shape[-1]!=4:
            table=table.transpose()
        #noise
        fodlr=fodlr#[20:101,20:125,20:101]
        fodlr[fodlr<0]=0
        fodlr[fodlr>1]=1
        # fsl5ttgen_mask=fsl5ttgen_mask#[20:101,20:125,20:101]
        if len(fsl5ttgen_mask.shape)>3:
            fsl5ttgen_mask=fsl5ttgen_mask[...,0]
        FAmin=0.25
        fsl5ttgen_mask[fsl5ttgen_mask>FAmin]=1
        fsl5ttgen_mask[fsl5ttgen_mask<=FAmin]=0

        cutted_x, cutted_y, cutted_z = fsl5ttgen_mask.shape
        # Including white matter, cortical grey matter, and subcortical grey matter tissues can improve the performance
        index_mask = np.where(fsl5ttgen_mask) 
        print('len:',len(index_mask[0]))
        index_mask = np.asarray(index_mask)
        x = index_mask[0, :]
        y = index_mask[1, :]
        z = index_mask[2, :]
        x_mask = np.logical_and(x >= self.margin, x < (cutted_x - self.margin))
        y_mask = np.logical_and(y >= self.margin, y < (cutted_y - self.margin))
        z_mask = np.logical_and(z >= self.margin, z < (cutted_z - self.margin))
        coord_mask = np.logical_and.reduce([x_mask, y_mask, z_mask])
        final_index = index_mask[:, coord_mask]
        index_length = len(final_index[0])
    
        Y=np.load(Y_path).astype(np.float32)
        G=np.load(G_path).astype(np.float32)
        n=np.load(n_path).astype(np.float32)
        if fodgt_path=='0':
            fod_info = {
                "id": fod_id,
                'fodlr': fodlr,
                'fodgt': '0',
                'mask' : fsl5ttgen_mask,
                'Y':Y,
                'G':G,
                'n':n,
                'table': table,
                'subject_id': subject_id,
                'index_mask': final_index,
                'index_length': index_length,
            }
        else:
            fodgt=nib.load(fodgt_path).get_fdata(dtype=np.float32)
            fodgt=fodgt
            fod_info = {
                "id": fod_id,
                'fodlr': fodlr,
                'fodgt': fodgt,
                'mask' : fsl5ttgen_mask,
                'Y':Y,
                'G':G,
                'n':n,
                'table': table,
                'subject_id': subject_id,
                'index_mask': final_index,
                'index_length': index_length,
            }
        self.fod_info.append(fod_info)

    def prepare(self):
        """Prepares the Dataset class for use.
        1. Compute the number of the fod lr/gt pairs
        2. Build the index for each fod sample
        """
        self.num_fod = len(self.fod_info)
        self._fod_ids = np.arange(self.num_fod)
        self.epoch_step = self.dataset_num_samples_per_data * self.num_fod

    @property
    def fod_ids(self):
        return self._fod_ids

    def __len__(self):
        """Return the total number of fod samples in the dataset.
        """
        return self.epoch_step*5

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--size_3d_patch', type=int, default=9,
                            help='the size of the 3D patches')
        parser.add_argument('--dataset_samples_overlapping_rate', type=float, default=0.3,
                            help='How many samples for each subject we want to randomly sample in one epoch')
        if is_train:
            parser.add_argument('--shuffle', type=bool, default=True, help='Shuffle the dataset during training')
        else:
            parser.add_argument('--shuffle', type=bool, default=False,
                                help='Do not shuffle the dataset during eval/val')
        return parser
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
