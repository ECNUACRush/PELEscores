#! python3
"""Minimal projection example with DeepDRR."""

import deepdrr
from deepdrr import geo, Volume, MobileCArm
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
import pdb
import os, re, math
import nibabel as nib
import numpy as np
# regex = re.compile(r'\d\d\d\d')

def main():
    # path = '/data1/hz/CTpelvic1k/bone/colon_bone/colon_003.nii.gz'
    # path = '/data1/hz/CTpelvic1k/original/colon_raw/colon_003.nii.gz'
    # path = '/data1/hz/CTpelvic1k_all/CTpelvic/CTPelvic1K_dataset4_data/dataset4_case_00014_data.nii.gz'
    # path = '/data1/hz/CTpelvic1k/original/aliyun_data/save/dataset2_1.3.6.1.4.1.9328.50.4.0552.nii.gz'
    # path = '/data1/hz/CTpelvic1k/original/colon_003.nii.gz'
    raw = '/data1/hz/CTpelvic1k/original/aliyun_data/new/COLONOG/'
    # path = '/data1/hz/CTpelvic1k/original/aliyun_data/new/COLONOG/'
    bone = '/data1/hz/CTpelvic1k/original/aliyun_data/dataset2Bone_new/'
    
    # bone = '/data1/hz/CTpelvic1k/original/aliyun_data/dataset2Raw_new/'
    # output_dir = '/data1/hz/CTpelvic1k/original/aliyun_data/dataset2Bone_new/'
    output_dir = '/data1/hz/CTpelvic1k/original/aliyun_data/dataset2Raw_new/'
    
    files_name1 = os.listdir(raw)
    files_name2 = os.listdir(bone)
    
    for file1 in files_name1:
      #if file1 in files_name2:
      if 'dataset2_' + str(file1[31:35]) + '.png' in files_name2:
              #img1 = nib.load(raw + file1)
        #img2 = nib.load(bone + file1)
        #if img1.shape == img2.shape:
          
          # if 'dataset2_' + str(file1[31:35]) + '.png' in files_name2:
          
          # if nib.load(bone+file1).shape == nib.load(raw+file1).shape:
          
    # output_dir = test_utils.get_output_dir()
    
    
    # seg_array = nib.load(bone).get_fdata()
    
              # seg_array = (nib.load(bone + file1).get_fdata() > 0.05)
      
          # patient = deepdrr.Volume.from_nifti(input_dir + file1, materials = dict(bone = seg_array))
          # data_dir = test_utils.download_sampledata("CTPelvic1K_sample")
          
              # patient = deepdrr.Volume.from_nifti(raw+file1, use_thresholding=False, materials = dict(bone = seg_array))
              patient = deepdrr.Volume.from_nifti(raw+file1, use_thresholding=True)
              patient.faceup()
              carm = MobileCArm(patient.center_in_world + geo.v(100, 0, -200))
            
          # Initialize the Projector object (allocates GPU memory)
              with Projector(patient, carm=carm) as projector:
              # Orient and position the patient model in world space.
                  # patient.orient_patient(head_first=True, supine=True)
                  # patient.place_center(carm.isocenter_in_world )
                  
                  # Move the C-arm to the desired pose.
                  carm.move_to(alpha=0, beta=0, degrees=True)
                  
                  # Run projection
                  image = projector()
              
              # pdb.set_trace()
              # pdb.set_trace()
              path = output_dir + 'dataset2_' + str(file1[31:35]) + '.png'
              
              image_utils.save(path, image)
              print(f"saved {file1} projection image to ")
    
    
if __name__ == "__main__":
    main()