import os
import numpy as np
import nibabel as nib
import yaml
from utils.sampling import HealpixSampling, BvecSampling
from model.model import Model
from tqdm import tqdm
import torch

DEVICE = 'cuda' #torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def sliding_window(input_data, window_size, stride):
    # This function generates windows with the specified size and stride
    for x in range(0, input_data.shape[3] - window_size + 1, stride):
        for y in range(0, input_data.shape[4] - window_size + 1, stride):
            for z in range(0, input_data.shape[5] - window_size + 1, stride):
                yield input_data[...,x:x+window_size, y:y+window_size, z:z+window_size], x, y, z


def main(data_path, batch_size, kernel_sizeSph, kernel_sizeSpa, 
         filter_start, sh_degree, depth, n_side,
         rf_name, wm, gm, csf,
         normalize, output_name, epoch, patch_size, middle_voxel, graph_sampling, conv_name, isoSpa, concatenate,model_path):
    """Test a model
    Args:
        data_path (str): Data path
        batch_size (int): Batch size
        n_epoch (int): Number of training epoch
        kernel_size (int): Kernel Size
        filter_start (int): Number of output features of the first convolution layer
        sh_degree (int): Spherical harmonic degree of the fODF
        depth (int): Graph subsample depth
        n_side (int): Resolution of the Healpix map
        rf_name (str): Response function algorithm name
        wm (float): Use white matter
        gm (float): Use gray matter
        csf (float): Use CSF
        normalize (bool): Normalize the fODFs
        load_state (str): Load pre trained network
        output_name (str): Name of the model folder
        epoch (int): Epoch to use for testing
    """
    # Load the shell and the graph samplings
    table=torch.from_numpy(np.load(cfg['input']['Gradtable_path'])).float().to(DEVICE)
    if table.shape[-1]!=4:
        table=table.transpose(1,0)
    Y=torch.from_numpy(np.load(cfg['input']['Y_path'])).float().to(DEVICE)
    nside16sh8=torch.from_numpy(np.load(cfg['input']['nside16sh8'])).float().to(DEVICE).unsqueeze(0)
    G=torch.from_numpy(np.load(cfg['input']['G_path'])).float().to(DEVICE)
    A=Y*G
    bs = patch_size//2
    feature_in = 1
    if concatenate:
        feature_in = patch_size*patch_size*patch_size
        patch_size = 1
    #bvec = dataset.vec
    if graph_sampling=='healpix':
        graphSampling = HealpixSampling(n_side, depth, patch_size, sh_degree=sh_degree, pooling_name=conv_name)
    elif graph_sampling=='bvec':
        graphSampling = BvecSampling(bvec, depth, image_size=patch_size, sh_degree=sh_degree, pooling_mode='average')
    else:
        raise NotImplementedError

    # Create the deconvolution model and load the trained model
    model = Model(filter_start, kernel_sizeSph, kernel_sizeSpa, normalize, conv_name, isoSpa, feature_in)
    model.load_state_dict(torch.load(model_path), strict=False)
    # Load model in GPU
    model = model.to(DEVICE)
    model.eval()

    # Output initialization
    if middle_voxel:
        b_selected = 1
        b_start = patch_size//2
        b_end = b_start + 1
    else:
        b_selected = patch_size
        b_start = 0
        b_end = b_selected

    nb_coef = int((sh_degree + 1) * (sh_degree / 2 + 1))
    window_size, stride=9,9
    inputs_gz1 = nib.load(cfg['input']['dwi_path'])
    input_data = torch.tensor(inputs_gz1.get_fdata().astype(np.float32)).permute(3,0,1,2).unsqueeze(0).unsqueeze(1).to(DEVICE)#
    #input_data=input_data[...,20:101,20:125,20:101]
    count = np.zeros((input_data.shape[3],
                    input_data.shape[4],
                    input_data.shape[5]))
    reconstruction_list = np.zeros((input_data.shape[3],
                                    input_data.shape[4],
                                    input_data.shape[5], table.shape[-2]-1))
    if wm:
        fodf_shc_wm_list = np.zeros((input_data.shape[3],
                                    input_data.shape[4],
                                    input_data.shape[5], nb_coef))
    if gm:
        fodf_shc_gm_list = np.zeros((input_data.shape[3],
                                    input_data.shape[4],
                                    input_data.shape[5], 1))
    if csf:
        fodf_shc_csf_list = np.zeros((input_data.shape[3],
                                    input_data.shape[4],
                                    input_data.shape[5], 1))

    total_iterations = (
    (input_data.shape[3] - window_size) // stride + 1
    ) * (
        (input_data.shape[4] - window_size) // stride + 1
    ) * (
        (input_data.shape[5] - window_size) // stride + 1
    )
    with tqdm(total=total_iterations, desc="Overall Progress") as pbar:
        for input, x, y, z in sliding_window(input_data, window_size, stride):
            x_reconstructed, x_deconvolved_equi_shc, x_deconvolved_inva_shc = model(input,nside16sh8,table.unsqueeze(0),A.unsqueeze(0))
            
            # Update reconstructed data
            reconstruction_list[x:x+window_size, y:y+window_size, z:z+window_size] += x_reconstructed[0].permute(1,2,3,0).cpu().detach().numpy()

            # If wm (white matter) data is available, update fodf_shc_wm_list
            if wm:
                fodf_shc_wm_list[x:x+window_size, y:y+window_size, z:z+window_size] += x_deconvolved_equi_shc[0, 0].permute(1,2,3,0).cpu().detach().numpy()

            index = 0
            # If gm (gray matter) data is available, update fodf_shc_gm_list
            if gm:
                fodf_shc_gm_list[x:x+window_size, y:y+window_size, z:z+window_size] += x_deconvolved_inva_shc[0, index].permute(1,2,3,0).cpu().detach().numpy()
                index += 1

            # If csf (cerebrospinal fluid) data is available, update fodf_shc_csf_list
            if csf:
                fodf_shc_csf_list[x:x+window_size, y:y+window_size, z:z+window_size] += x_deconvolved_inva_shc[0, index].permute(1,2,3,0).cpu().detach().numpy()

            # Update the count for averaging later
            count[x:x+window_size, y:y+window_size, z:z+window_size] += 1
            pbar.update(1)
            '''
            for j in range(len(input)):
                sample_id_j = sample_id[j]
                print(dataset.x[sample_id_j], dataset.y[sample_id_j], dataset.z[sample_id_j])
                reconstruction_list[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                    dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                    dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected] += x_reconstructed[j, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
                if wm:
                    fodf_shc_wm_list[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                    dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                    dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected] += x_deconvolved_equi_shc[j, 0, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
                index = 0
                if gm:
                    fodf_shc_gm_list[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                    dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                    dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected] += x_deconvolved_inva_shc[j, index, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
                    index += 1
                if csf:
                    fodf_shc_csf_list[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                    dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                    dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected] += x_deconvolved_inva_shc[j, index, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
                count[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                    dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                    dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected] += 1
                print(np.sum(count[dataset.x[sample_id_j]:dataset.x[sample_id_j] + b_selected,
                                    dataset.y[sample_id_j]:dataset.y[sample_id_j] + b_selected,
                                    dataset.z[sample_id_j] :dataset.z[sample_id_j] + b_selected]))
            '''
    # Average patch
    try:
        reconstruction_list[count!=0] = reconstruction_list[count!=0] / count[count!=0, None]
        if wm:
            fodf_shc_wm_list[count!=0] = fodf_shc_wm_list[count!=0] / count[count!=0, None]
        if gm:
            fodf_shc_gm_list[count!=0] = fodf_shc_gm_list[count!=0] / count[count!=0, None]
        if csf:
            fodf_shc_csf_list[count!=0] = fodf_shc_csf_list[count!=0] / count[count!=0, None]
    except:
        print('Count failed')
    
    # Save the results
    bs = 0
    if bs>0:
        count = count[bs:-bs,bs:-bs,bs:-bs]
    #else:
    #    count = count[:dataset.X,:dataset.Y,:dataset.Z]
    count = np.array(count).astype(np.float32)
    img = nib.Nifti1Image(count, inputs_gz1.affine, inputs_gz1.header)
    nib.save(img, os.path.join('/home/xqgao/D/xuedwi2fod_e3so3bigdata/results/',output_name+'_count.nii.gz'))
    if bs>0:
        reconstruction_list = reconstruction_list[bs:-bs,bs:-bs,bs:-bs]
    #else:
    #    reconstruction_list = reconstruction_list[:dataset.X,:dataset.Y,:dataset.Z]
    reconstruction_list = np.array(reconstruction_list).astype(np.float32)
    img = nib.Nifti1Image(reconstruction_list, inputs_gz1.affine, inputs_gz1.header)
    nib.save(img, os.path.join('/home/xqgao/D/xuedwi2fod_e3so3bigdata/results/',output_name).replace('.nii.gz', '_reconstruction.nii.gz'))
    if wm:
        if bs>0:
            fodf_shc_wm_list = fodf_shc_wm_list[bs:-bs,bs:-bs,bs:-bs]
        #else:
        #    fodf_shc_wm_list = fodf_shc_wm_list[:dataset.X,:dataset.Y,:dataset.Z]
        fodf_shc_wm_list = np.array(fodf_shc_wm_list).astype(np.float32)
        img = nib.Nifti1Image(fodf_shc_wm_list, inputs_gz1.affine, inputs_gz1.header)
        nib.save(img, os.path.join('/home/xqgao/D/xuedwi2fod_e3so3bigdata/results/',output_name).replace('.nii.gz', '_fodf.nii.gz'))
    if gm:
        if bs>0:
            fodf_shc_gm_list = fodf_shc_gm_list[bs:-bs,bs:-bs,bs:-bs]
        #else:
        #    fodf_shc_gm_list = fodf_shc_gm_list[:dataset.X,:dataset.Y,:dataset.Z]
        fodf_shc_gm_list = np.array(fodf_shc_gm_list).astype(np.float32)
        img = nib.Nifti1Image(fodf_shc_gm_list, inputs_gz1.affine, inputs_gz1.header)
        nib.save(img, os.path.join('/home/xqgao/D/xuedwi2fod_e3so3bigdata/results/',output_name).replace('.nii.gz', '_fodf_gm.nii.gz'))
    if csf:
        if bs>0:
            fodf_shc_csf_list = fodf_shc_csf_list[bs:-bs,bs:-bs,bs:-bs]
        #else:
        #    fodf_shc_csf_list = fodf_shc_csf_list[:dataset.X,:dataset.Y,:dataset.Z]
        fodf_shc_csf_list = np.array(fodf_shc_csf_list).astype(np.float32)
        img = nib.Nifti1Image(fodf_shc_csf_list, inputs_gz1.affine, inputs_gz1.header)
        nib.save(img, os.path.join('/home/xqgao/D/xuedwi2fod_e3so3bigdata/results/',output_name).replace('.nii.gz', '_fodf_csf.nii.gz'))
    if cfg['input']['position']:
        pre_shape=[1,cfg['pred_patch_size']['a'],cfg['pred_patch_size']['b'],cfg['pred_patch_size']['c'],1]
        x_start=cfg['input']['x_start']#np.min(np.nonzero(ROI1)[0])
        y_start=cfg['input']['y_start']#np.min(np.nonzero(ROI1)[1])
        z_start=cfg['input']['z_start']#np.min(np.nonzero(ROI1)[2])
        x_end=x_start+pre_shape[1]
        y_end=y_start+pre_shape[2]
        z_end=z_start+pre_shape[3]
    else:
        ROI1 = nib.load(cfg['input']['ROI_path'])
        ROI1 = ROI1.get_fdata().astype(np.float32)
        x_start=np.min(np.nonzero(ROI1)[0])
        y_start=np.min(np.nonzero(ROI1)[1])
        z_start=np.min(np.nonzero(ROI1)[2])
        x_end=np.max(np.nonzero(ROI1)[0])
        y_end=np.max(np.nonzero(ROI1)[1])
        z_end=np.max(np.nonzero(ROI1)[2])

    output_file_name_ROI = output_name.replace('.nii.gz', '_cstr.nii.gz')
    ROI_output=fodf_shc_wm_list[x_start:x_end,y_start:y_end,z_start:z_end,:]
    ROI_output_img = nib.Nifti1Image(ROI_output, inputs_gz1.affine)
    nib.save(ROI_output_img, os.path.join('/home/xqgao/D/xuedwi2fod_e3so3bigdata/results/',output_file_name_ROI))


if __name__ == '__main__':
# Load the config file
    with open('/home/xqgao/D/xuedwi2fod_e3so3bigdata/test.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Test properties
    batch_size = cfg['batch_size']
    middle_voxel = cfg['middle_voxel']

    # Data path
    data_path = cfg['data_path']
    assert os.path.exists(data_path)

    # Load trained model
    output_name = cfg['output_name']
    epoch = cfg['epoch']

    # Load parameters
    with open('/home/xqgao/D/xuedwi2fod_e3so3bigdata/train.yaml', 'r') as file:
        args_json = yaml.load(file, Loader=yaml.FullLoader)

    # Model architecture properties
    filter_start = int(args_json['filter_start'])
    sh_degree = int(args_json['sh_degree'])
    kernel_sizeSph = int(args_json['kernel_sizeSph'])
    kernel_sizeSpa = int(args_json['kernel_sizeSpa'])
    depth = int(args_json['depth'])
    n_side = int(args_json['n_side'])
    normalize = bool(args_json['normalize'])
    patch_size = int(args_json['size_3d_patch'])
    try:
        graph_sampling = int(args_json['graph_sampling'])
    except:
        graph_sampling = 'healpix'
    conv_name = str(args_json['conv_name'])
    isoSpa = not bool(args_json['anisoSpa'])
   
    try:
        concatenate = bool(args_json['concatenate'])
    except:
        concatenate = False

    # Load response functions
    rf_name = str(args_json['rf_name'])
    wm = bool(args_json['wm'])
    gm = bool(args_json['gm'])
    csf = bool(args_json['csf'])


    print(f'Filter start: {filter_start}')
    print(f'SH degree: {sh_degree}')
    print(f'Kernel size spherical: {kernel_sizeSph}')
    print(f'Kernel size spatial: {kernel_sizeSpa}')
    print(f'Unet depth: {depth}')
    print(f'N sidet: {n_side}')
    print(f'fODF normalization: {normalize}')
    print(f'Patch size: {patch_size}')
    print(f'RF name: {rf_name}')
    print(f'Use WM: {wm}')
    print(f'Use GM: {gm}')
    print(f'Use CSF: {csf}')
    print(f'Concatenate: {concatenate}')

    # Test directory
    test_path = cfg['model_path']
    if not os.path.exists(test_path):
        print('Create new directory: {0}'.format(test_path))
        os.makedirs(test_path)


    main(data_path, batch_size, kernel_sizeSph, kernel_sizeSpa, 
         filter_start, sh_degree, depth, n_side,
         rf_name, wm, gm, csf,
         normalize, output_name, epoch, patch_size, middle_voxel, graph_sampling, conv_name, isoSpa, concatenate,test_path)

