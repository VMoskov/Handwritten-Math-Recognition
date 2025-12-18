# convert the isolate symbol to image 
# ground truth is from a csv file using the IUD info from the inkml file

import sys, os
from skimage.draw import line
from skimage.io import imread, imsave
from inkml import *
import numpy as np
from BBfromInk import *
from PIL import Image
from imageFromInk import *
from skimage.filters import gaussian


def main():
    if len(sys.argv) < 3:
        print("Usage: [[python3]] convertInkmlPng_iso.py <DirIn> <DirOut> <GT.txt> [size[:marge]]")
        print("   DirIn is the directory containing the inkml files")
        print("   DirOut is the directory where the png files will be saved")
        print("   GT.txt is the file containing the ground truth")
        print("   size is the size of the image (default 32)")
        print("   marge is the number of pixels to add to the ink bounding box (default 1)")
        sys.exit(0)
        
    dirIn = sys.argv[1]
    dirOut = sys.argv[2]
    fileGT = sys.argv[3]
        
    # check if all  exist
    if not os.path.exists(dirIn):
        print("Directory %s does not exist" % dirIn)
        sys.exit(0)
    if not os.path.exists(dirOut):
        print("Directory %s does not exist" % dirOut)
        sys.exit(0)
    if not os.path.exists(fileGT):
        print("File %s does not exist" % fileGT)
        sys.exit(0)    
    
    # read the ground truth in a dictionary
    gt = {}
    classSet = set()
    with open(fileGT, 'r') as f:
        for line in f:
            s = line.strip().split(',')
            if s[1] == '/':
                s[1] = 'div_op'
            if s[1] == '.':
                s[1] = 'dot'
            # if upper case, add _ 
            if s[1].isupper():
                s[1] = s[1] + '_'
            # ignore empty class
            gt[s[0]] = s[1]
            classSet.add(s[1])

    # remove junk class
    if 'junk' in classSet:
        classSet.remove('junk')

    # create the output directory for each class if not exist   
    for c in classSet:
        if not os.path.exists(os.path.join(dirOut, c)):
            os.makedirs(os.path.join(dirOut, c))

    size = 32
    marge = 1
    if len(sys.argv) > 4:
        sm = sys.argv[4].split(':')
        size = int(sm[0])
        if len(sm) > 1:
            marge = int(sm[1])

    # for each file in the IN directory

    for fileIn in os.listdir(dirIn):
        if fileIn.endswith('.inkml'):
            try:
                #print("Processing ", fileIn)
                ink = parse_inkml(os.path.join(dirIn, fileIn))
                if ink.UI in gt and gt[ink.UI] in classSet:
                    img, updated_ink = convert_to_imgs(ink,size,fit=False, marge=marge, center=True)   
                    # TODO: apply a Gaussian noise with kernel 3x3
                    im = Image.fromarray(img)
                    # save the image in the correct directory 
                    fileOut = os.path.join(dirOut, gt[ink.UI], fileIn.replace('.inkml', '.png'))
                    im.save(fileOut)
                else:
                    print("No ground truth for ", ink.UI)
            except Exception as e:
                print("Error processing ", fileIn, " : ", e)

if __name__ == '__main__':
    main()