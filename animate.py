from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os

saveDirectory = 'Raymond' #can be whatever you want, your name probably
#create save directory
if not os.path.exists(saveDirectory): #make folder if doesn't exist
    os.makedirs(saveDirectory)
  
#csv source url
#direct link from https://sites.google.com/site/gdocs2direct/, csv file
url = 'https://drive.google.com/uc?export=download&id=1ith3awBE8vTcpVGo6o3SVQj2aZhW5f4o' #raymond's

try:
  reader = pd.read_csv(url)
  ar = reader.to_numpy()
except:
  print("Error")

xPoints = ar[:,0]
yPoints = ar[:,1]

if len(xPoints)==len(yPoints):
  samples = len(xPoints)
  t = np.linspace(1,samples,samples)
else:
  print("Error")

L = samples

#DFT built from scratch

def sinDFT(y,t,n):
  comparisonValues = np.sin(n*t*2*np.pi/L)
  dotprod = np.dot(y,comparisonValues)
  norm = np.dot(comparisonValues,comparisonValues)
  return dotprod/norm

def cosDFT(y,t,n):
  comparisonValues = np.cos(n*t*2*np.pi/L)
  dotprod = np.dot(y,comparisonValues)
  norm = np.dot(comparisonValues,comparisonValues)
  return dotprod/norm

def constant(y,samples):
  return np.sum(y)/samples

def DFT(y,t):
  sinls,cosls = [],[]
  for n in range(1,int((len(t)+1)/2)):
    sinls.append(sinDFT(y,t,n))
    cosls.append(cosDFT(y,t,n))
  return (sinls,cosls)

def fourier(t,sinterms,costerms,y,numterms):
  total = np.zeros(len(t))
  for i in range(numterms):
    total += sinterms[i]*np.sin((i+1)*t*(2*np.pi/L)) + costerms[i]*np.cos((i+1)*t*(2*np.pi/L))
  total += constant(y,samples)
  return total

#compute the values using DFT
ySins = DFT(yPoints,t)[0]
yCoses = DFT(yPoints,t)[1]
xSins = DFT(xPoints,t)[0]
xCoses = DFT(xPoints,t)[1]

tVals = np.linspace(1,samples,10000)

#axes
xmin = np.min(xPoints)-np.mean(xPoints)/10
xmax = np.max(xPoints)+np.mean(xPoints)/10
ymin = np.min(yPoints)-np.mean(yPoints)/10
ymax = np.max(yPoints)+np.mean(yPoints)/10

totalTerms = len(xSins)+1

def makeImg(numterms):
  plt.axis([xmin,xmax,ymin,ymax])
  plt.title(f'number of terms = {numterms}')
  plt.plot(fourier(tVals,xSins,xCoses,xPoints,numterms),fourier(tVals,ySins,yCoses,yPoints,numterms))
  plt.savefig(saveDirectory+'/'+f"{numterms:04d}") #need to change the f string to match the number of digits in the num of points there are
  plt.close()

#frames
for i in range(1,totalTerms):
  makeImg(i)
  print(f'saved image {i}')

print('done saving, making gif...')

#animate —————————————————————————————————————————————————————————————————————————————————————————————————————————————
import glob
import contextlib
from PIL import Image

# filepaths
fp_in = saveDirectory+"/*.png"
fp_out = saveDirectory+"/!"+saveDirectory+".gif"

# use exit stack to automatically close opened images
with contextlib.ExitStack() as stack:
    # lazily load images
    imgs = (stack.enter_context(Image.open(f))
            for f in sorted(glob.glob(fp_in)))
    # extract  first image from iterator
    img = next(imgs)
    # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)

print('done')
