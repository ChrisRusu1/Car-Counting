import pycocotools
from pycocotools import COCO
import requests

coco = COCO('cocoapi/annotations/instances_train2017.json')
cars = coco.loadCars(coco.getCarIds())
nms=[car['name'] for car in cars]
print('COCO caregories: \n{}\n'.format(' '.join(nms)))

carIds = coco.getCarIds(carNms=['vehicle'])
imgIds = coco.getImgIds(carIds=carIds )
images = coco.loadImgs(imgIds)
print("imgIds: ", imgIds)
print("images: ", images)

for im in images:
    print("im: ", im)
    img_data = requests.get(im['coco_url']).content
    #with open('downloaded_images/' + im['file_name'], 'wb') as handler:
    #handler.write(img_data)