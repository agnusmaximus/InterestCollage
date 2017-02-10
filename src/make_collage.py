import sys
import pickle
import scipy
import glob
import itertools
import numpy as np
import uuid
from triangulate import *
from PIL import Image, ImageDraw, ImageFilter
from PIL import *
import matplotlib.pyplot as plt
import random
import copy
import math
from math import ceil
from scipy import optimize

np.set_printoptions(threshold='nan')

class Particle():
    def __init__(self, shattered_piece, radius_pad=0):
        self.image = shattered_piece[0]
        self.y, self.x = shattered_piece[1]
        self.orig_x, self.orig_y = self.x, self.y
        self.radius_pad = radius_pad

    def compute_direction_vector(self, weights, array):
        x, y, w, h = self.y, self.x, self.image.size[0], self.image.size[1]
        
        assert(w > 0)
        assert(h > 0)

        x_min = max(0, x-self.radius_pad)
        x_max = min(weights.shape[1], x+w+self.radius_pad)
        y_min = max(0, y-self.radius_pad)
        y_max = min(weights.shape[0], y+h+self.radius_pad)
        subweights = weights[y_min:y_max, x_min:x_max]
        dir_vector = np.array([0.0, 0.0])

        # Don't move if all same color
        all_same_color = np.all(subweights.flatten() == subweights.flatten()[0])
        if all_same_color:
            self.direction = np.array([random.uniform(-1,1),random.uniform(-1,1)]).astype(float)
            
            # Somewhat want to go back to the center
            if random.uniform(0, 1) < .4:
                center_x, center_y = weights.shape[0]/float(2), weights.shape[1]/float(2)
                #self.direction = np.array([self.orig_x-(self.x+float(w)/2), 
                #                           self.orig_y-(self.y+float(h)/2)])
                self.direction = np.array([center_x-(self.x+float(w)/2), 
                                           center_y-(self.y+float(h)/2)])
                
            self.direction = self.direction / max(np.linalg.norm(self.direction), 1)

            return

        #print(array[y_min:y_max,x_min:x_max])
        #print(subweights)
        #sys.exit(0)

        n_rows = x_max-x_min
        n_cols = y_max-y_min

        r,g,b,a = self.image.split()
        a = np.array(a)

        #print(subweights.shape, n_rows, n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                if a[j][i] != 0:
                    dir_vector += subweights[j][i] * np.array([i-n_rows/float(2), j-n_cols/float(2)]).astype(float)
        self.direction = dir_vector

        self.direction = self.direction / max(np.linalg.norm(self.direction), 1)

    def apply_direction_vector(self, (w,h)):
        self.x += self.direction[0]*2
        self.y += self.direction[1]*2
        self.x = int(np.round(self.x))
        self.y = int(np.round(self.y))
        self.x = max(self.x, 0)
        self.y = max(self.y, 0)
        self.x = min(self.x, h - self.image.size[1])
        self.y = min(self.y, w - self.image.size[0])

def render_particles(particles, sz):
    new_image = Image.new("RGBA", sz, (255, 255, 255, 20))
    for particle in particles:
        new_image.paste(particle.image, (particle.y, particle.x), particle.image)
    return new_image

def random_location(size):
    return (random.randint(0, size[0]), random.randint(0, size[1]))

def organize_grid(fnames, dim=200, n_cols=3):
    imgs = [Image.open(x) for x in fnames]

    for img in imgs:
        assert(img.size == (dim, dim))

    n_rows = int(ceil(float(len(imgs) / n_cols)))
    grid = Image.new("RGBA", (dim * n_cols, dim * n_rows))
    for row in range(n_rows):
        for col in range(n_cols):
            cur_index = row * n_cols + col
            if cur_index >= len(imgs):
                continue
            grid.paste(imgs[cur_index], (col*dim, row*dim), imgs[cur_index])
    return grid

def resize_and_pad(image, new_size):
    old_size = image.size
    new_im = Image.new("RGBA", new_size, (255, 255, 255, 255))  
    new_im.paste(image, ((new_size[0]-old_size[0])/2,
                         (new_size[1]-old_size[1])/2))
    return new_im

def render_particles_on_top_of_target(particles, target, sz):
    img = render_particles(particles, sz)
    overall = resize_and_pad(target, sz)
    overall.paste(img, (0,0), img)
    return overall

def compute_direction_vectors(particles, image_to_approximate, size):
    print("Computing attractive forces...")
    image_to_approximate = resize_and_pad(image_to_approximate, size)
    rendered_array = np.array(render_particles(particles, size).convert('L'))
    to_approximate_array = np.array(image_to_approximate.convert('L'))
    weights = to_approximate_array

    print("Assigning attractive force vectors...")
    for i, particle in enumerate(particles):
        particle.compute_direction_vector(weights, to_approximate_array)
    print("Done computing direction vectors...")

def apply_direction_vectors(particles, (w,h)):
    for particle in particles:
        particle.apply_direction_vector((w,h))

def make_collage(collage_names, overlay_names):
    collage_initial = organize_grid(collage_names)
    shattered_pieces = shatter(collage_initial, n_points=15000)
    particles = [Particle(x) for x in shattered_pieces]
    #particles = [Particle(shattered_pieces[len(shattered_pieces)/2])]

    target = Image.open(overlay_names[0]).convert('L')
    #target = target.filter(ImageFilter.FIND_EDGES)
    target = target.point(lambda x: 0 if x<128 else 255, '1')

    render_particles_on_top_of_target(particles, target, collage_initial.size).show()

    for i in range(1000):
        compute_direction_vectors(particles, target, collage_initial.size)
        apply_direction_vectors(particles, collage_initial.size)
        intermediate_overlay = render_particles_on_top_of_target(particles, target, collage_initial.size)
        intermediate = render_particles(particles, collage_initial.size)
        intermediate.save("test/%d.png" % i)
        intermediate_overlay.save("test/%d_overlayed.png" % i)

if __name__=="__main__":

    if len(sys.argv) != 3:
        print("Usage: make_collage.py collage_name_1,collage_name_2,...,collage_name_n overlap_name_1,overlap_name_2,...,overlay_name_n")
        sys.exit(0)

    collage_file_names = sys.argv[1].split(",")
    overlay_file_names = sys.argv[2].split(",")

    make_collage(collage_file_names,
                 overlay_file_names)
