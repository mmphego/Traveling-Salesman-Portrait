#!/usr/bin/env python

'''
This is a script written by Randal S. Olson (randalolson.com) for the Traveling Salesman Portrait project.

More information on the project can be found on my blog:

http://www.randalolson.com/2018/04/11/traveling-salesman-portrait-in-python/

Please check my project repository for information on how this script can be used and shared:

https://github.com/rhiever/Data-Analysis-and-Machine-Learning-Projects
'''
import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import urllib

from PIL import Image
from scipy.spatial.distance import pdist, squareform
from tsp_solver.greedy_numpy import solve_tsp


def main(*args):
    if image_link.startswith('http'):
        try:
            image_path = image_url.split('/')[-1]
        except Exception:
            image_path = 'Image.jpg'

        if not os.path.exists(image_path):
            print ('Getting image from URL')
            urllib.urlretrieve(image_url, image_path)
    else:
        print ('Loading image from local path')
        image_path = image_link

    original_image = Image.open(image_path)
    # Convert Original image to black and white.
    original_image = original_image.convert('L')
    w, h = original_image.width, original_image.height
    bw_image = original_image.convert('1', dither=Image.NONE)

    bw_image_array = np.array(bw_image, dtype=np.int)
    black_indices = np.argwhere(bw_image_array == 0)
    # Changing "size" to a larger value makes this algorithm take longer,
    # but provides more granularity to the portrait
    chosen_black_indices = black_indices[np.random.choice(black_indices.shape[0], replace=False,
                                                             size=size)]

    fig = plt.figure(figsize=(w/100 +1, h/100), dpi=110)
    plt.scatter([x[1] for x in chosen_black_indices], [x[0] for x in chosen_black_indices],
                color='black', s=1)
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    print ('Scatter image created')
    fig.savefig(image_path.replace('.jpg', '_scatter.png'), bbox_inches='tight', dpi=fig.dpi)
    print ('Creating tsp image. Note: This will take time!!!')
    distances = pdist(chosen_black_indices)
    distance_matrix = squareform(distances)

    optimized_path = solve_tsp(distance_matrix)
    optimized_path_points = [chosen_black_indices[x] for x in optimized_path]

    plt.figure(figsize=((w/100) + 2, (h/100) + 2), dpi=100)
    plt.plot([x[1] for x in optimized_path_points], [x[0] for x in optimized_path_points],
            color='black', lw=1)
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.savefig('traveling-salesman-portrait.png', bbox_inches='tight')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-u', '--url', dest='url', action='store', default=False,
        help='Image URL')
    parser.add_argument('-l', '--local', dest='local', action='store', default=False,
        help='Load local image')
    parser.add_argument('-s', '--size', dest='size', action='store', default=10000, type=int,
        help=('Changing "size" to a larger value makes this algorithm take longer, '
              'but provides more granularity to the portrait'))

    args = vars(parser.parse_args())
    local_image = args.get('local')
    image_url = args.get('url')
    size = args.get('size')
    if local_image:
        image_link = local_image
    else:
        image_link = image_url
    try:
        assert image_link
        main(image_link, size)
    except:
        print 'Provide URL or Image location'
        sys.exit(1)
