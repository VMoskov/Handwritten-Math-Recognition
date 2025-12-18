Handwritten Math Recognition
========

Author/Contact: Harold Mouchère (harold.mouchere@univ-nantes.fr)

This is a Master Project to apply machine (deep) learning tool on a challenging computer vision problem: recognition of handwritten Math Expression.


**The subject is available in the directory "subject"** and the provided code in "code". Most of the code deals with the pre-processing of the data (inkml) and the evaluation of the recognition (LG files).

## Data 

Some toy examples are available in the "data" directory but the complete dataset is available in the [CROHME Zenodo repository](https://zenodo.org/records/8428035).

Pre-processed data are available in [uncloud storage](https://uncloud.univ-nantes.fr/index.php/s/OdAtErZgxKGjsNy).

Several format of data are used in this project. Depending of your interest, everything will not be useful. 

- InkML files: vectorial representation of handwritten math expressions (strokes with (x,y) coordinates and time). These files are used in the provided code for segmentation and recognition of symbols.
- PNG files: raster images of handwritten math expressions. These files can be used for deep learning approaches (symbol detection and recognition).
- LG files: text files representing the ground-truth or the recognition results of math expressions.
- LG files with bounding boxes: text files representing the ground-truth or the recognition results of math expressions with bounding boxes of symbols (used for detection tasks).

## Usage of some provided tools

The provided code allows to convert inkml files to images, segment inkml files into symbol hypotheses, build LG files with symbol bounding boxes, for training and evaluation of symbols detection and recognition.

### imageFromInk.py
```
Usage: [[python3]] imageFromInk.py <fileIn.ink> <fileOut.png> [size[:marge] [fit]]
   create a png image from the inkml file 
   fileOut.png is the name of the image file to create
   fileOut.lg will contain the bounding box of the symbols
   size is the size of the image (default 1000)
   fit is a boolean to fit the ink in the image (default False)
   marge is the number of pixels to add to the ink bounding box (default 5)
```

### BBfromInk.py
```
Usage: [[python3]] BBfromInk.py <fileIn.ink> [<fileOut.lg>]
           Output the Bounding Box of the inkml file 
```

### processAll.sh
Apply all recognition step to all inkml files and produce the LG files

```
Usage: processAll <input_inkml_dir> <output_lg_dir>
```

###  python3 segmenter.py 
Generate from an inkml file hypotheses of symbols in a LG file.

```
usage: python3 segmenter.py [-i fname][-o fname][-s N]
     -i fname / --inkml fname  : input file name (inkml file)
     -o fname / --output fname : output file name (LG file)
     -s N / --str N            : if no inkmlfile is selected, run with N strokes
```

### python3 segmentSelect.py
Keep or not each segment hypotheses and generate a new LG file 

usage: python3 [-o fname] [-s] segmentSelect.py inkmlfile lgFile
     inkmlfile  : input inkml file name
     lgFile     : input LG file name
     -o fname / --output fname : output file name (LG file)
     -s         : save hyp images
	 
### python3 symbolReco.py

Recognize each hypothesis and save all acceptable recognition in a LG file

```
usage: python3 symbolReco.py [-s] [-o fname][-w weigthFile] inkmlfile lgFile
     inkmlfile  : input inkml file name
     lgFile     : input LG file name
     -o fname / --output fname : output file name (LG file)
     -w fname / --weight fname : weight file name (nn pytorch file)
     -s         : save hyp images
```
 
###  python3 selectBestSeg.py

From an LG file with several hypotheses, keep only one coherent global solution (greedy sub-optimal)

usage: python3 selectBestSeg.py  [-o fname] lgFile
     lgFile     : input LG file name
     -o fname / --output fname : output file name (LG file)
	 
### ./listExistingLG.sh
Preparation of the partial evaluation of recognition. Not needed if all expressions are recognized.
Generate the list of existing LG files with associated LG ground-thruth.

Usage: processAll <ground-truthdir> <output_lg_dir>

### evaluate (from LgEval lib)

Compare output LG files with ground-truth LG file and generate a detailed summary of metrics.

2 Usages: global evaluation or partial evaluation from a list of couple
       evaluate outputDir groundTruthDir 
       evaluate fileList 
