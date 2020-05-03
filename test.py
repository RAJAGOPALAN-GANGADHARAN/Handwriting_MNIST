import os
import sys
from PIL import Image

size = 128, 128

for infile in sys.argv[1:]:
    outfile = os.path.splitext(infile)[0] +"_modified" ".jpg"
    if infile != outfile:
        try:
            im = Image.open(infile)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(outfile, "JPEG")
        except IOError:
            print("cannot create thumbnail for '%s'" % infile)

# plt.subplot(5, 5, i+1)
        # plt.xticks([])
        # plt.yticks([])
        # plt.grid(False)
        # plt.imshow(image)
        # plt.xlabel("done")
        # plt.show()
        # x=input()
        # i += 1
