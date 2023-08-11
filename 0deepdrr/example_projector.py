#! python3
"""Minimal projection example with DeepDRR."""

import deepdrr
from deepdrr import geo
from deepdrr.utils import test_utils, image_utils
from deepdrr.projector import Projector
from deepdrr.device.mobile_carm import pose_vector_angles
import pdb
import os, re, math
import nibabel as nib
import numpy as np
def main():
    # input_dir = '../../datasets/DeepDRR_Data/raw/'
    # input_dir = '/home/huangzhen/drr_test/deepdrr/bone_try/'
    
    # input_dir = '/data1/hz/CTpelvic1k/original/CTPelvic1K_dataset6_data/'
    input_dir = '/data1/hz/CTpelvic1k/bone/'
    # input_dir = '/data1/hz/CTpelvic1k/original/pelvic1k_remove_bed/'
    
    filenames = os.listdir(input_dir)
    for f in filenames:
      if os.path.isdir(f):
        filenames.remove(f)
    
    regex = re.compile(r'\d\d\d\d')
    
    
    output_dir = "output3/"
    bins = np.linspace(0,1,1000) 
    # data_dir = test_utils.download_sampledata("CTPelvic1K_sample")
    # patient = deepdrr.Volume.from_nifti(
        # data_dir / "dataset6_CLINIC_0001_data.nii.gz", use_thresholding=True
    # )

    # for z in range(-450, -50, 50): 
    
    for i,file1 in enumerate(filenames):
      if not file1.startswith("dataset6_CLINIC"):
          continue
      num = re.findall(r"\d+",file1)
      # if num == []: # 源文件夹中有两个dataset，一次投一个
        # continue
      num = str(num)
      
      # 如果不传入meterial这个参数，那么他会自动生成一个，结果是air=all True, bone=all False, soft tissue= all False
      # seg_array = (nib.load(input_dir +"../mask/" + file1.replace("_data.nii.gz","_mask_4label.nii.gz").replace("colon_","dataset6_CLINIC_")).get_fdata() > 0.05)

      # patient = deepdrr.Volume.from_nifti(input_dir + file1, materials = dict(bone = seg_array))
      pos = np.linspace(0,0.2,10)
      patient = deepdrr.Volume.from_nifti(input_dir + file1)
      patient.faceup()
      epi_vox_center = (np.array(patient.data.shape) - 1) / 2.
      position = patient.anatomical_from_ijk.data @ (list(epi_vox_center)+[1])

      # define the simulated C-arm
      alpha,beta = 0,0
      alpha,beta=pose_vector_angles(- position[:3])
      pp = []
      for p in pos:
        carm = deepdrr.MobileCArm(patient.center_in_world + geo.v(position[0]*p, position[1]*p, position[2]*p),sensor_width=1536,sensor_height=1536)
        with Projector(patient, carm=carm) as projector:
            carm.move_to(alpha=alpha, beta=beta)
            image = projector()
        array = np.array(image)/255
        histogram , _ = np.histogram(array,bins = bins)
        histogram = histogram/histogram.max()
        std = histogram.std(0)
        mean = histogram.mean(0)
        pp.append([-math.log(1 - mean)*2 - math.log(std),image,p])
      pp = sorted(pp,key=lambda x:x[0],reverse=False)
      for value in pp:
          print(value[0],value[2])
      image = pp[0][1]
      path = output_dir+num+f"_{round(pp[0][2],3)}.png"
      image_utils.save(path, image)
      print(f"saved example projection image to {path}")
    # print(f'{z} : ok')
if __name__ == "__main__":
    main()

