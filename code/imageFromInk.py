import sys, os
from skimage.draw import line
from skimage.io import imread, imsave
from inkml import *
import numpy as np
from BBfromInk import *
from PIL import Image
from math import floor

def convert_to_imgs(ink, box_axis_size,fit = True, marge = 5, center=False): 
    """
    Convert inkml file to image
    :param ink: Inkml object
    :param box_axis_size: size of the image
    :param fit: if True, the image will be crop to fit the ink (in smaller dimension)
    :param marge: if fit is True, the ink will be resized to fit the dim with a marge of marge pixels
    :param center: if True, the ink will be centered ; ignored if fit is True
    :return: the image and the inkml rescaled and translated
    """
    pattern_drawn = np.ones(shape=(box_axis_size, box_axis_size), dtype=np.float32)
    # Special case of inkml file with zero trace (empty)
    if len(ink) == 0:
        return np.matrix(pattern_drawn * 255, np.uint8)

    # get the general BB of the ink
    min_x, min_y, max_x, max_y = mergeBB([s.get_min_max() for s in ink.strokesPoints.values()])
    'trace dimensions'
    trace_height, trace_width = max_y - min_y , max_x - min_x 
    if trace_height == 0:
        trace_height += 1
    if trace_width == 0:
        trace_width += 1
    #print("trace height : ", trace_height, " trace width : ", trace_width)
    '' 'KEEP original size ratio' ''
    trace_ratio = (trace_width) / (trace_height)
    box_ratio = box_axis_size / box_axis_size #Wouldn't it always be 1
    scale_factor = 1.0
    '' 'Set \"rescale coefficient\" magnitude' ''
    if trace_ratio < box_ratio:
        scale_factor = ((box_axis_size-1 - 2 * marge) / trace_height)
    else:
        scale_factor = ((box_axis_size-1 - 2 * marge) / trace_width)
    #print("scale f : ", scale_factor)

    #shift pattern to its relative position
    #print(mergeBB([s.get_min_max() for s in ink.strokesPoints.values()]))
    ink.translate(-min_x, -min_y )
    #print(mergeBB([s.get_min_max() for s in ink.strokesPoints.values()]))
    #rescale
    ink.scale(scale_factor)
    #print(mergeBB([s.get_min_max() for s in ink.strokesPoints.values()]))
    if fit:
        if trace_width > trace_height:
            pattern_drawn = np.ones(shape=( round(trace_height * scale_factor + 2 * marge), box_axis_size), dtype=np.float32)
        else:
            pattern_drawn = np.ones(shape=(box_axis_size,round(trace_width * scale_factor + 2 * marge)), dtype=np.float32)
    else:
        # center pattern
        if center:
            ink.translate((box_axis_size -  2 * marge - trace_width * scale_factor) / 2, (box_axis_size -  2 * marge - trace_height * scale_factor) / 2)
    #print(mergeBB([s.get_min_max() for s in ink.strokesPoints.values()]))
    #take into account the marge
    ink.translate(marge, marge) 
    #print(mergeBB([s.get_min_max() for s in ink.strokesPoints.values()]))
    #print("pattern drawn shape : ", pattern_drawn.shape)
    for trace in ink.strokesPoints.values():
        strkInt = Stroke(([floor(x) for x in trace.coords[0]], [floor(y) for y in trace.coords[1]]))
        #print("stroke : ", strkInt.coords)
        pattern_drawn = draw_pattern(strkInt, pattern_drawn)
    return np.matrix(pattern_drawn * 255, np.uint8), ink

def draw_pattern(trace,pattern_drawn):
    H,W = pattern_drawn.shape
    ' SINGLE POINT TO DRAW '
    if len(trace) == 1:
            x_coord, y_coord = trace[0]
            if x_coord >= 0 and x_coord < W and y_coord >= 0 and y_coord < H:
                pattern_drawn[y_coord, x_coord] = 0.0 #0 means black
    else:
        ' TRACE HAS MORE THAN 1 POINT '
        'Iterate through list of traces endpoints'
        for pt_idx in range(len(trace) - 1):
                'Indices of pixels that belong to the line. May be used to directly index into an array'
                #print("draw line : ",trace[pt_idx], trace[pt_idx+1])
                linesX = linesY = []
                oneLineY, oneLineX = line(r0=trace[pt_idx][1], c0=trace[pt_idx][0],
                                   r1=trace[pt_idx + 1][1], c1=trace[pt_idx + 1][0])
                # size of the line with kernel of shape T (3 pixels)
                linesX = np.concatenate(
                    [ oneLineX, oneLineX, oneLineX+1 ])
                linesY = np.concatenate(
                    [ oneLineY+1, oneLineY, oneLineY])

                linesX[linesX<0] = 0
                linesX[linesX>=W] = W-1

                linesY[linesY<0] = 0
                linesY[linesY>=H] = H-1

                pattern_drawn[ linesY, linesX] = 0.0
                # pattern_drawn[ oneLineX, oneLineY ] = 0.0
    return pattern_drawn

def get_label(id, ink, labelSet):
    """
    get the label and return it as label_i
    :param id: id of the segment
    :param ink: Inkml object
    :return: the label of the segment and the index of the symbol
    """
    label = "none"
    #print("search id : ", id , " in ", labelSet)
    if id in ink.segments:
        label = ink.segments[id].label
    i = 1
    while (label + "_" + str(i)) in labelSet:
        i += 1
    label = label + "_" + str(i)
    labelSet.append(label)
    return label

def main():
    if len(sys.argv) < 3:
        print("Usage: [[python3]] imageFromInk.py <fileIn.ink> <fileOut.png> [size[:marge] [fit]]")
        print("   create a png image from the inkml file ")
        print("   fileOut.png is the name of the image file to create")
        print("   fileOut.lg will contain the bounding box of the symbols")
        print("   size is the size of the image (default 1000)")
        print("   fit is a boolean to fit the ink in the image (default False)")
        print("   marge is the number of pixels to add to the ink bounding box (default 5)")
        sys.exit(0)
    ink = parse_inkml(sys.argv[1])
    fileOut = sys.argv[2]
    size = 1000
    marge = 5
    if len(sys.argv) > 3:
        sm = sys.argv[3].split(':')
        size = int(sm[0])
        if len(sm) > 1:
            marge = int(sm[1])
    if len(sys.argv) > 4 and (sys.argv[4] == 'fit' or sys.argv[4] == 'True'):
        fit = True
    else:
        fit = False
    img, updated_ink = convert_to_imgs(ink,size,fit, marge, center=True)
    im = Image.fromarray(img)
    im.save(fileOut)

    coord_bb = get_bounding_box_fromInk(updated_ink)
    # LG file with bounding box
    lgname = fileOut.replace('.png', '.lg')
    labelSet = []
    with open(lgname, 'w') as f:
        for s_id in coord_bb.keys():
            l = get_label(s_id, updated_ink, labelSet)
            f.write("BB, %s, %d, %d, %d, %d\n" % (l, coord_bb[s_id][0], coord_bb[s_id][1], coord_bb[s_id][2], coord_bb[s_id][3]))
    f.close()


if __name__ == '__main__':
    main()