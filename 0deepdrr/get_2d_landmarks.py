#! python3
"""Minimal projection example with DeepDRR."""
import os
import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
import nibabel as nib
from nibabel import nifti1
import numpy as np
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

def get_drr(p1,p2,out, item):

    data_nii = nib.load(p1)
    data = np.asarray(data_nii.get_fdata())
    affine = data_nii.affine.copy()
    hdr = data_nii.header.copy()
    label_cod = np.load(p2)



    
    i = 0
    patient = deepdrr.Volume.from_nifti(p1, use_thresholding=False, segmentation=False, nii_bool=False)
    patient.faceup()
    carm = deepdrr.MobileCArm(patient.center_in_world + geo.v(0, 0, -600))
    with Projector(patient, carm=carm) as projector:
        carm.move_to(alpha=0, beta=0, degrees=True)
        image = projector()
    image_utils.save(out + item + str(i).zfill(4) +'.png', image)
    print(i + "+" + item)


    marg = 5
    while(i<23):
        x, y, z = np.trunc(label_cod[i,0]).astype(int), np.trunc(label_cod[i,1]).astype(int), np.trunc(label_cod[i,2]).astype(int)
        mask = np.zeros_like(data)
        mask[x-marg:x+marg, y-marg:y+marg, z-marg:z+marg] = 600

        pelvic_ban = nib.Nifti1Image(mask, affine, hdr)
        patient = deepdrr.Volume.from_nifti(p1, use_thresholding=True, segmentation=True, nii_bool=True, nii_data = [pelvic_ban])
        patient.faceup()
        carm = deepdrr.MobileCArm(patient.center_in_world + geo.v(0, 0, -600))
        with Projector(patient, carm=carm) as projector:
            carm.move_to(alpha=0, beta=0, degrees=True)
            image2 = projector()
        i = i + 1
        image_utils.save(out + item + str(i).zfill(4) +'.png', image)
        


if __name__ == "__main__":

    path = '/home/huangzhen/drr_test/deepdrr/segment/'
    out = '/home/huangzhen/drr_test/deepdrr/save/'
    os.makedirs(out, exist_ok=True)  

    file_all = os.listdir(path)

    for item in file_all:
        if item[-12:] == '_0000.nii.gz':
            get_drr(path + item, path + item[:-12] + ".nii.gz.npy", out, item[:-12])
            print(item)