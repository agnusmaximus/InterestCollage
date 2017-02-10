# Source: http://garethrees.org/2013/08/09/triangulation/

import copy
from itertools import product
from itertools import cycle
import random
from scipy.spatial import Delaunay
from PIL import Image, ImageDraw
import numpy as np

def show_overlay_triangulation(image, triangulation):
    image = copy.copy(image)
    draw = ImageDraw.Draw(image)
    for triangle_points in triangulation:
        draw.line(triangle_points, fill=(0, 0, 0, 255), width=1)
    #image.show()

def crop_region(polygon, image):
    base_image = np.asarray(image)

    maskIm = Image.new('L', (base_image.shape[1], base_image.shape[0]), 0)
    ImageDraw.Draw(maskIm).polygon(polygon, outline=1, fill=1)
    bbox = maskIm.getbbox()
    mask = np.array(maskIm)

    img = np.empty(base_image.shape,dtype='uint8')
    img[:,:,:3] = base_image[:,:,:3]
    img[:,:,3] = mask * 200
    result = Image.fromarray(img, "RGBA")

    cropped = result.crop(bbox)

    return cropped, (bbox[0], bbox[1])

def shatter(image, n_points=10000):
    w, h = image.size
    selected_triangulation = triangulate(rectangle(n_points, w, h))
    show_overlay_triangulation(image, selected_triangulation)
    shattered = []
    for i, triangle in enumerate(selected_triangulation):
        cropped, location = crop_region(triangle, image)
        if cropped:
            shattered.append((cropped, location))
    return shattered

def rectangle(n_points, n, m):
    return [(random.randint(0, n),
             random.randint(0, m)) for i in range(n_points)] + [(0,0),(0,m),(n,0),(n,m)]

def triangulate(points):
    tri = Delaunay(points)
    triangulation = [[points[x] for x in simplices] for simplices in tri.simplices]
    return triangulation
