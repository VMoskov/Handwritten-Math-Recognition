import xml.etree.ElementTree as ET
import sys, os
from skimage.draw import line
from skimage.io import imread, imsave
from inkml import *
import numpy as np

def parse_inkml(inkml_file_abs_path, return_label=False):
    """ parse the file and take into account the inch unit if necessary"""
    if inkml_file_abs_path.endswith('.inkml'):
        ink = Inkml(inkml_file_abs_path)
        ink.loadStrokesPoints()

        in_inches = False
        for t_i in ink.strokesPoints.keys():
            for coord_i in range(len(ink.strokesPoints[t_i].coords)):
                in_inches = all([not axis_coord.is_integer() for axis_coord in ink.strokesPoints[t_i].coords[coord_i]])
            if in_inches:
                break

        if in_inches:
            ## for each stroke, for each point, multiply by 1000
            for t_i in ink.strokesPoints.keys():
                for coord_i in range(len(ink.strokesPoints[t_i].coords)):
                    ink.strokesPoints[t_i].coords[coord_i] = [axis_coord*1000 for axis_coord in ink.strokesPoints[t_i].coords[coord_i]]
        return ink
        
    else:
        print('File ', inkml_file_abs_path, ' does not exist !')
        return Inkml()


def mergeBB(listBB):
    """ merge the bounding boxes of list """
    x_min = min([bb[0] for bb in listBB])
    y_min = min([bb[1] for bb in listBB])
    x_max = max([bb[2] for bb in listBB])
    y_max = max([bb[3] for bb in listBB])
    return [x_min, y_min, x_max, y_max]

def get_bounding_box_fromInk(ink):
    """ get the bounding box of the inkml symbols """
    bb_seg = {}
    for s_id in ink.segments.keys():
        k = s_id
        if ink.segments[s_id].href != "":
            k = ink.segments[s_id].href
        bb_seg[k] = mergeBB([s.get_min_max() for s in ink.getStrokesFromSeg(s_id)])
    return bb_seg;



def main():
    if len(sys.argv) < 2:
        print("Usage: [[python3]] BBfromInk.py <fileIn.ink> [<fileOut.lg>]")
        print("   Output the Bounding Box of the inkml file ")
        sys.exit(0)
    file = sys.argv[1]
    # x_min, y_min, x_max, y_max 
    ink = parse_inkml(file)
    coord_bb = get_bounding_box_fromInk(ink)

    fileOut = sys.stdout
    if len(sys.argv) == 3:
        fileOut = open(sys.argv[2], 'w')
    
    for s_id in coord_bb.keys():
        fileOut.write("BB, %s, %d, %d, %d, %d\n" % (s_id, coord_bb[s_id][0], coord_bb[s_id][1], coord_bb[s_id][2], coord_bb[s_id][3]))
    fileOut.close()


if __name__ == '__main__':
    main()