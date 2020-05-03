import os
from PIL import Image

size = 128, 128

def resize():
    infile='predict.jpg'
    outfile = os.path.splitext(infile)[0] + "_modified.jpg"
    if infile != outfile:
        try:
            im = Image.open(infile)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(outfile, "JPEG")
        except IOError:
            print("cannot create thumbnail for '%s'" % infile)
