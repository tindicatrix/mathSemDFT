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

try: #try to convert csv of ordered pairs into a numpy array
  reader = pd.read_csv(url)
  ar = reader.to_numpy()
except:
  print("Error")

xPoints = ar[:,0] #seperates the x values into its own array
yPoints = ar[:,1] #seperates the y values into its own array

if len(xPoints)==len(yPoints): #makes sure that each x has a y
  samples = len(xPoints)       
  t = np.linspace(1,samples,samples) #creates an array of parameter variable t with [sample] points from 1 to [sample]
else:
  print("Error")

L = samples #define 1 wavelength

#DFT built from scratch

def sinDFT(y,t,n): #computes the sin terms for the DFT
  comparisonValues = np.sin(n*t*2*np.pi/L) #creates an array of points that follow a perfect sin wave of frequency n
  dotprod = np.dot(y,comparisonValues) #take the dot product between our points and the actual waves
  norm = np.dot(comparisonValues,comparisonValues) #find the total sum under the points are, for normalization
  return dotprod/norm

def cosDFT(y,t,n): #computes the cos terms for the DFT
  comparisonValues = np.cos(n*t*2*np.pi/L) #creates an array of points that follow a perfect sin wave of frequency n
  dotprod = np.dot(y,comparisonValues) #take the dot product between our points and the actual waves
  norm = np.dot(comparisonValues,comparisonValues) #find the total sum under the points are, for normalization
  return dotprod/norm

def constant(y,samples): #finds the constant term
  return np.sum(y)/samples

def DFT(y,t): #computes the total DFT with sin, cos terms
  sinls,cosls = [],[]
  for n in range(1,int((len(t)+1)/2)):
    sinls.append(sinDFT(y,t,n))
    cosls.append(cosDFT(y,t,n))
  return (sinls,cosls)

def fourier(t,sinterms,costerms,y,numterms): #plots the fourier series using the values obtained by DFT
  total = np.zeros(len(t)) #creates a zero vector of length of t
  for i in range(numterms): #iterate through every term in the fourier series, adding it to the total (does the sum)
    total += sinterms[i]*np.sin((i+1)*t*(2*np.pi/L)) + costerms[i]*np.cos((i+1)*t*(2*np.pi/L))
  total += constant(y,samples) #adds the constant term to the sum
  return total

#compute the values using DFT
ySins = DFT(yPoints,t)[0] #finds every sin term for y(t)
yCoses = DFT(yPoints,t)[1] #finds every cos term for y(t)
xSins = DFT(xPoints,t)[0] #finds every sin term for x(t)
xCoses = DFT(xPoints,t)[1] #finds every cos term for x(t)

tVals = np.linspace(1,samples,10000) #parameteric t-values for plotting purposes

#axes
#sets axes for the graph constant between each frame and slightly larger than the image itself
xmin = np.min(xPoints)-np.mean(xPoints)/10 
xmax = np.max(xPoints)+np.mean(xPoints)/10
ymin = np.min(yPoints)-np.mean(yPoints)/10
ymax = np.max(yPoints)+np.mean(yPoints)/10

totalTerms = len(xSins)+1

def makeImg(numterms): #saves each frame 
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
             save_all=True, duration=75, loop=0)

print('done')
