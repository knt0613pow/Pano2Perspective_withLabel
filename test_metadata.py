from PIL import Image
from PIL.ExifTags import TAGS
from PIL import ExifTags

# path to the image or video
imagename = "./bundang/분당메이홈_7_2014_11_00_37.36078324_127.11329148.jpg"

# read the image data using PIL
image = Image.open(imagename)

# extract EXIF data
# exifdata = image.getexif()

# iterating over all EXIF data fields
# for tag_id in exifdata:
#     # get the tag name, instead of human unreadable tag id
#     tag = TAGS.get(tag_id, tag_id)
#     data = exifdata.get(tag_id)
#     # decode bytes 
#     if isinstance(data, bytes):
#         data = data.decode()
#     print(f"{tag:25}: {data}")

# breakpoint()

# exif = {
#     TAGS[k]: v
#     for k, v in image._getexif().items()
#     if k in TAGS
# }

exifdata = image._getexif()
gpsinfo = {}
# breakpoint()
# for key in exifdata[34853].keys():
#     decode = ExifTags.GPSTAGS.get(key,key)
#     gpsinfo[decode] = exifdata[34853][key]
print( exifdata[34853][17][0])
