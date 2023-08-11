import numpy as np
import os,sys,re
import deepdrr
from PIL import Image

if __name__=="__main__":
    input_dir = "output2/"
    filenames = os.listdir(input_dir)
    bins = np.linspace(0,1,100)
    for f in filenames:
        if os.path.isdir(f):
          filenames.remove(f)

    for i,file1 in enumerate(filenames):
      num = re.findall(r"\d+",file1)
      num = str(num) + "wo_material_0.1.png"
      patient = Image.open(input_dir + file1)
      patient = np.array(patient)/255
      histogram,_ = np.histogram(patient.data,bins = bins)
      std = histogram.std(0)
      print(file1,std)



