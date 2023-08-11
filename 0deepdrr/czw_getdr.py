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
    raw = '/data1/ubuntu/drr_spine/ct/'
    bone = '/data1/ubuntu/drr_spine/spine_LAT/'
    output_dir = '/data1/ubuntu/drr_spine/project/'
    
    files_name1 = sorted(os.listdir(raw))
    files_name2 = sorted(os.listdir(bone))
    print(files_name1)
    for file1 in files_name1:
      #if file1 in files_name2:
      if '.nii.gz'  in file1:
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
              carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) #liver 1000 no liver geo.v(0, 0, -800))
              
              """
              rotate phantom
              """
              #patient.rotate(geo.core.Rotation.from_euler("x", 90, degrees=True), center=patient.center_in_world)
              
              
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
              path = output_dir + 'ct_' + str(file1[:]) + '.png'
              
              image_utils.save(path, image)
              print(f"saved {file1} projection image to ")
    
def main_spine():

    raw = '/data1/ubuntu/drr_spine/ct/train_ct/'
    bone = '/data1/ubuntu/drr_spine/spine/train_gt/'
    output_dir = '/data1/ubuntu/drr_spine/project/'
    ref = '/data1/ubuntu/drr_spine/spine/train_gt/'
    i = 0
    files_name1 = os.listdir(raw)
    ref_name = os.listdir(ref)
    # files_name2 = os.listdir(bone)
    
    for file1 in files_name1:
        # if file1 in files_name2:
        # if "dataset2" in file1:
        if '.nii.gz'  in file1:
          i = i + 1
          print("number: ",i)
          if i<612:
            continue 
          # output_dir = test_utils.get_output_dir()
    
    
          # seg_array = nib.load(bone).get_fdata()
    
          #seg_array = (nib.load(bone + file1.replace(".nii","_seg.nii")).get_fdata() > 0.05)
          
          # only 19 -- 24 spine
          #seg_array = (nib.load(bone + file1).get_fdata() > 18.05)
          seg_array = (nib.load(bone + file1.replace(".nii","_seg.nii")).get_fdata() > 18.05)
  
          patient = deepdrr.Volume.from_nifti(raw + file1, materials = dict(bone = seg_array))
          # data_dir = test_utils.download_sampledata("CTPelvic1K_sample")
      
          # patient = deepdrr.Volume.from_nifti(raw+file1, use_thresholding=False)
          
          patient.faceup()
          
          
          # rib
          #if "-covid19-A" in file1:
          #  carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -700)) #liver 1000 no liver geo.v(0, 0, -800))
          #else:
          #  carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) #liver 1000 no liver geo.v(0, 0, -800))

          # spine1K
          if "liver" in file1:
            carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) 
          else:
            carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) 

          #carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -1000))
          
          
          """
          rotate phantom
          """
          patient.rotate(geo.core.Rotation.from_euler("x", 270, degrees=True), center=patient.center_in_world)

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
          
          #spine1K
          path = output_dir + 'spine_LAT_six_spines/' + str(file1[:]) + '_left_AT.png'
          
          #path = output_dir + 'spine_LAT_six_spines/' + str(file1[:-7]) + 'left.png'
          
          image_utils.save(path, image)
          print(f"saved {file1} projection image to ")

def spine_mask():

    raw = '/data1/ubuntu/drr_spine/ct/rib/'
    bone = '/data1/ubuntu/drr_spine/spine/rib/'
    output_dir = '/data1/ubuntu/drr_spine/project/'
    ref = '/data1/ubuntu/drr_spine/spine/rib/'

    files_name1 = os.listdir(raw)
    ref_name = os.listdir(ref)
    # files_name2 = os.listdir(bone)
    
    for file1 in files_name1:
        # if file1 in files_name2:
        # if "dataset2" in file1:
        if '.nii.gz'  in file1:
          #seg_array = nib.load(bone).get_fdata()
    
          #seg_array = (nib.load(bone + file1.replace(".nii","_seg.nii")).get_fdata() > 0.05)
          
          # only 19 -- 24 spine
          seg_array1 = (nib.load(bone + file1).get_fdata() > 18.05).astype(int)
          seg_array2 = (nib.load(bone + file1).get_fdata() > 19.05).astype(int)
          seg_array3 = (nib.load(bone + file1).get_fdata() > 20.05).astype(int)
          seg_array4 = (nib.load(bone + file1).get_fdata() > 21.05).astype(int)
          seg_array5 = (nib.load(bone + file1).get_fdata() > 22.05).astype(int)
          seg_array6 = (nib.load(bone + file1).get_fdata() > 23.05).astype(int)

          
          seg_array = seg_array1 + seg_array2 +seg_array3 +seg_array4 + seg_array5 +seg_array6
  
          patient = deepdrr.Volume.from_nifti(raw + file1, materials = dict(bone = seg_array))
      # data_dir = test_utils.download_sampledata("CTPelvic1K_sample")
      
          # patient = deepdrr.Volume.from_nifti(raw+file1, use_thresholding=False)
          
          patient.faceup()
          
          if "-covid19-A" in file1:
            carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -700)) #liver 1000 no liver geo.v(0, 0, -800))
          else:
            carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) #liver 1000 no liver geo.v(0, 0, -800))
          #carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -1000))
          
          
          """
          rotate phantom
          """
          patient.rotate(geo.core.Rotation.from_euler("x", 90, degrees=True), center=patient.center_in_world)

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
          # path = output_dir + 'spine_' + str(file1[:]) + '.png'
          path = output_dir + 'spine_LAT_six_spine_indep/' + str(file1[:-7]) + '.png'
          
          image_utils.save(path, image)
          print(f"saved {file1} projection image to ")

def main_all():

    raw = '/data1/ubuntu/drr_spine/ct/rib/'
    bone = '/data1/ubuntu/drr_spine/spine/rib/'
    output_dir = '/data1/ubuntu/drr_spine/project/'
    ref = '/data1/ubuntu/drr_spine/spine/rib/'
    i = 0
    files_name1 = sorted(os.listdir(raw))
    files_name2 = sorted(os.listdir(bone))
    print(files_name1)
    print(len(files_name1))
    for file1 in files_name1:
      i = i + 1
      #if file1 in files_name2:
      #if '.nii.gz'  in file1:
      if i>366: #366 (AP RIB:300-365δͶӰ)

              patient = deepdrr.Volume.from_nifti(raw+file1, use_thresholding=True)
              
              patient.faceup()
              
              #if "-covid19-A" in file1:
              #    carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -700)) #liver 1000 no liver geo.v(0, 0, -800))
              #else:
              #    carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) #liver 1000 no liver geo.v(0, 0, -800))
              
              
              # spine1K
              carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) 

              """
              rotate phantom
              """
              #patient.rotate(geo.core.Rotation.from_euler("x", 180, degrees=True), center=patient.center_in_world)
              
              with Projector(patient, carm=carm) as projector:
                  
                  # Move the C-arm to the desired pose.
                  carm.move_to(alpha=0, beta=0, degrees=True)
                  
                  # Run projection
                  image = projector()
              
              path = output_dir + 'drr_image_AP_PA/' + str(file1[:-7]) + '.png'
              
              image_utils.save(path, image)
              print(f"saved {path}")
              
              # spinr1K
              #seg_array = (nib.load(bone + file1.replace(".nii","_seg.nii")).get_fdata() > 0.05)
              
              #
              seg_array = (nib.load(bone + file1).get_fdata() > 0.05)
  
              
              """
              spine
              """
              patient = deepdrr.Volume.from_nifti(raw + file1, materials = dict(bone = seg_array))
              
              patient.faceup()
              
              if "-covid19-A" in file1:
                  carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -700)) #liver 1000 no liver geo.v(0, 0, -800))
              else:
                  carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) #liver 1000 no liver geo.v(0, 0, -800))
              
              
              """
              rotate phantom
              """
              #patient.rotate(geo.core.Rotation.from_euler("x", 180, degrees=True), center=patient.center_in_world)
    
              with Projector(patient, carm=carm) as projector:
                  carm.move_to(alpha=0, beta=0, degrees=True)
                  image = projector()
              path = output_dir+ 'drr_spine_AP_PA/' +  str(file1[:-7]) + '.png'
              
              image_utils.save(path, image)
              print(f"saved {path}")
def main_all_lat():

    raw = '/data1/ubuntu/drr_spine/ct/train_ct/'
    bone = '/data1/ubuntu/drr_spine/spine/train_gt/'
    output_dir = '/data1/ubuntu/drr_spine/project/'
    ref = '/data1/ubuntu/drr_spine/spine/train_gt/'
    i = 0
    files_name1 = sorted(os.listdir(raw))
    files_name2 = sorted(os.listdir(bone))
    print(files_name1)
    print(len(files_name1))
    for file1 in files_name1:
      i = i + 1
      #if file1 in files_name2:
      if '.nii.gz'  in file1:
          print("number: ",i)

          if i>817: 

              patient = deepdrr.Volume.from_nifti(raw+file1, use_thresholding=True)
              
              patient.faceup()
              #if "-covid19-A" in file1:
              #    carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -700)) #liver 1000 no liver geo.v(0, 0, -800))
              #else:
              #    carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) #liver 1000 no liver geo.v(0, 0, -800))
              # spine1K
              carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) 
              """
              rotate phantom
              """
              patient.rotate(geo.core.Rotation.from_euler("x", 270, degrees=True), center=patient.center_in_world)
              
              with Projector(patient, carm=carm) as projector:
                  
                  # Move the C-arm to the desired pose.
                  carm.move_to(alpha=0, beta=0, degrees=True)
                  
                  # Run projection
                  image = projector()
              
              #path = output_dir + 'drr_image_LAT/' + str(file1[:-7]) + '.png'
              #spine1K
              path = output_dir + 'drr_image_LAT/' + str(file1[:]) + '_left_AT.png'
              
              image_utils.save(path, image)
              print(f"saved {path}")
              
              # spinr1K
              seg_array = (nib.load(bone + file1.replace(".nii","_seg.nii")).get_fdata() > 0.05)
              
              #seg_array = (nib.load(bone + file1).get_fdata() > 0.05)
  
              
              """
              spine
              """
              patient = deepdrr.Volume.from_nifti(raw + file1, materials = dict(bone = seg_array))
              
              patient.faceup()
              
              #if "-covid19-A" in file1:
              #    carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -700)) #liver 1000 no liver geo.v(0, 0, -800))
              #else:
              #    carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) #liver 1000 no liver geo.v(0, 0, -800))
              # spine1K
              carm = MobileCArm(patient.center_in_world + geo.v(0, 0, -900)) 
              
              """
              rotate phantom
              """
              patient.rotate(geo.core.Rotation.from_euler("x", 270, degrees=True), center=patient.center_in_world)
    
              with Projector(patient, carm=carm) as projector:
                  carm.move_to(alpha=0, beta=0, degrees=True)
                  image = projector()
              #path = output_dir+ 'drr_spine_LAT/' +  str(file1[:-7]) + '.png'
              
              #spine1K
              path = output_dir + 'drr_spine_LAT/' + str(file1[:]) + '_left_AT.png'
              
              image_utils.save(path, image)
              print(f"saved {path}")
if __name__ == "__main__":
    #main()
    #main_spine()
    #main_all()
    main_all_lat()
    #spine_mask()