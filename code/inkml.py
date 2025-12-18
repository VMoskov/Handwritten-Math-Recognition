################################################################
# inkml.py - InkML parsing lib
#
# Author: H. Mouchere, Feb. 2014
# Copyright (c) 2014,2023 Harold Mouchere
################################################################

import xml.etree.ElementTree as ET
class Segment(object):
    """Class to reprsent a Segment compound of strokes (id) with an id and label."""
    __slots__ = ('id', 'label' ,'strId', 'href')
    
    def __init__(self, *args):
        if len(args) == 4:
            self.id = args[0]
            self.label = args[1]
            self.strId = args[2]
            self.href = args[3]
        else:
            self.id = "none"
            self.label = ""
            self.strId = set([])
            self.href = ""
    

class Stroke(object):
    """Class to represent a stroke as x and y coordinates"""
    __slots__ = ('coords')
    
    def __init__(self, *args):
        if len(args) == 1:
            # if it a string, use the init_from_list method
            if isinstance(args[0], str):
                self.init_from_list(args[0])
            else:
                self.coords = args[0]            
        else:
            self.coords = []
    
    def init_from_list(self,coords_str):
        self.coords = list(zip(*[[float(axis_coord) for axis_coord in coord.strip(' ').split(' ')] for coord in (coords_str).replace('\n', '').split(',')]))

    def get_min_max(self):
        """return the min and max coordinates of the stroke"""
        xmin = min(self.coords[0])
        xmax = max(self.coords[0])
        ymin = min(self.coords[1])
        ymax = max(self.coords[1])
        return (xmin,ymin,xmax,ymax)

    def translate(self, dx,dy):
        """ translate the points of the stroke"""
        self.coords = (list(map(lambda x: x+dx, self.coords[0])), list(map(lambda x: x+dy, self.coords[1])))

    def scale(self, sx,sy):
        """ scale the points of the stroke"""
        self.coords = (list(map(lambda x: x*sx, self.coords[0])), list(map(lambda x: x*sy, self.coords[1])))

    def __str__(self) -> str:
        """return the string representation of the stroke"""
        lpoints = list(zip(self.coords[0], self.coords[1]))
        return ','.join([' '.join([str(x) for x in point]) for point in lpoints])

    def __len__(self):
        """return the number of points in the stroke"""
        if len(self.coords) == 0:
            return 0
        return len(self.coords[0])    

    def __getitem__(self, key):
        """return the point at index key"""
        return (self.coords[0][key],self.coords[1][key])    


class Inkml(object):
    """Class to represent an INKML file with strokes, segmentation and labels"""
    __slots__ = ('fileName', 'strokes', 'strokesPoints', 'strkOrder','segments','truth', 'mathML', 'UI', 'writer', 'copyright');
    
    NS = {'ns': 'http://www.w3.org/2003/InkML', 'xml': 'http://www.w3.org/XML/1998/namespace'}
    
    ##################################
    # Constructors (in __init__)
    ##################################
    def __init__(self,*args):
        """can be read from an inkml file (first arg)"""
        self.fileName = None
        self.strokes = {}
        self.strokesPoints = {}
        self.strkOrder = []
        self.segments = {}
        self.truth = ""
        self.mathML = None
        self.UI = ""
        self.writer = ""
        self.copyright = ""
        if len(args) == 1:
            self.fileName = args[0]
            self.loadFromFile()
    
    def __len__(self):
        """return the number of strokes"""
        return len(self.strokes)

    def fixNS(self,ns,att):
        """Build the right tag or element name with namespace"""
        return '{'+Inkml.NS[ns]+'}'+att

    def loadFromFile(self):
        """load the ink from an inkml file (strokes, segments, labels)"""
        tree = ET.parse(self.fileName)
        # # ET.register_namespace();
        root = tree.getroot()
        for info in root.findall('ns:annotation',namespaces=Inkml.NS):
            if 'type' in info.attrib:
                if info.attrib['type'] == 'truth' and info.text is not None:
                    self.truth = info.text.strip()
                if info.attrib['type'] == 'UI' and info.text is not None:
                    self.UI = info.text.strip()
                if info.attrib['type'] == 'writer' and info.text is not None:
                    self.writer = info.text.strip()
                if info.attrib['type'] == 'copyright' and info.text is not None:
                    self.copyright = info.text.strip()
        for strk in root.findall('ns:trace',namespaces=Inkml.NS):
            self.strokes[strk.attrib['id']] = strk.text.strip()
            self.strkOrder.append(strk.attrib['id'])
        segments = root.find('ns:traceGroup',namespaces=Inkml.NS)
        if segments is None or len(segments) == 0:
            print ("No segmentation info")
            return
        for seg in (segments.iterfind('ns:traceGroup',namespaces=Inkml.NS)):
            id = seg.attrib[self.fixNS('xml','id')]
            label = seg.find('ns:annotation',namespaces=Inkml.NS).text
            strkList = set([])
            for t in seg.findall('ns:traceView',namespaces=Inkml.NS):
                strkList.add(t.attrib['traceDataRef'])
            href = ""
            if seg.find('ns:annotationXML',namespaces=Inkml.NS) is not None:
                href = seg.find('ns:annotationXML',namespaces=Inkml.NS).attrib['href']
            self.segments[id] = Segment(id,label, strkList, href)
        # load MathML  
        if root.find('ns:annotationXML',namespaces=Inkml.NS) is not None:
            self.mathML = root.find('ns:annotationXML',namespaces=Inkml.NS)

    def loadStrokesPoints(self):
        """ load the strokes as a list of points from the strokes strings """
        for (id,s) in self.strokes.items():
            self.strokesPoints[id] = Stroke(s)
            
    def getInkML(self,file):
        """write the ink to an inkml file (strokes, segments, labels)"""
        outputfile = open(file,'w')
        outputfile.write("<ink xmlns=\"http://www.w3.org/2003/InkML\">\n<traceFormat>\n<channel name=\"X\" type=\"decimal\"/>\n<channel name=\"Y\" type=\"decimal\"/>\n</traceFormat>")
        outputfile.write("<annotation type=\"truth\">"+self.truth+"</annotation>\n")
        outputfile.write("<annotation type=\"UI\">"+self.UI+"</annotation>\n")
        if self.writer != "":
            outputfile.write("<annotation type=\"writer\">"+self.writer+"</annotation>\n")
        if self.copyright != "":
            outputfile.write("<annotation type=\"copyright\">"+self.copyright+"</annotation>\n")
        for (id,s) in self.strokes.items():
            outputfile.write("<trace id=\""+id+"\">\n"+s+"\n</trace>\n")
        outputfile.write("<traceGroup>\n")
        for (id,s) in self.segments.items():
            outputfile.write("\t<traceGroup xml:id=\""+id+"\">\n")
            outputfile.write("\t\t<annotation type=\"truth\">"+s.label+"</annotation>\n")
            for t in s.strId:
                outputfile.write("\t\t<traceView traceDataRef=\""+t+"\"/>\n")
            outputfile.write("\t</traceGroup>\n")
        outputfile.write("</traceGroup>\n</ink>")
        outputfile.close()
    
    def isRightSeg(self, seg):
        """return true is the set seg is an existing segmentation"""
        for s in self.segments.values():
            if s.strId == seg:
                return True
        return False
    def getInkMLwithoutGT(self,withseg,file):
        """write the ink to an inkml file (strokes, segments, labels)"""
        outputfile = open(file,'w')
        outputfile.write("<ink xmlns=\"http://www.w3.org/2003/InkML\">\n<traceFormat>\n<channel name=\"X\" type=\"decimal\"/>\n<channel name=\"Y\" type=\"decimal\"/>\n</traceFormat>")
        outputfile.write("<annotation type=\"UI\">"+self.UI+"</annotation>\n")
        for id in sorted(self.strokes.keys(), key=lambda x: float(x)):
            outputfile.write("<trace id=\""+id+"\">\n"+self.strokes[id]+"\n</trace>\n")
        if withseg :
            outputfile.write("<traceGroup>\n")
            for id in sorted(self.segments.keys(), key=lambda x: float(x) if x.isdigit() else x):
                outputfile.write("\t<traceGroup xml:id=\""+id+"\">\n")
                outputfile.write("\t\t<annotation type=\"truth\">"+self.segments[id].label+"</annotation>\n")
                for t in sorted(self.segments[id].strId, key=lambda x: float(x) if x.isdigit() else x):
                    outputfile.write("\t\t<traceView traceDataRef=\""+t+"\"/>\n")
                outputfile.write("\t</traceGroup>\n")
            outputfile.write("</traceGroup>")
        outputfile.write("</ink>")
        outputfile.close()    
    
    def getStrokesFromSeg(self,seg):
        """return the strokes of a segment"""
        return [self.strokesPoints[s] for s in self.segments[seg].strId]
    
    def translate(self, dx, dy):
        """translate the ink by dx and dy"""
        for (id,s) in self.strokesPoints.items():
            s.translate(dx,dy)
    
    def scale(self, sx, sy=None):
        """scale the ink by sx and sy"""
        if sy is None:
            sy = sx
        for (id,s) in self.strokesPoints.items():
            s.scale(sx,sy)
