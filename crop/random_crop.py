#!/usr/bin/env python
# coding: UTF-8

import numpy as np
from PIL import Image


__all__ = ['UniformDist', 'NormalDist', 'LogUniformDist', 'RandomRotateStretchCropResize', 'RandomContrastBrightnessNoise']


class UniformDist(object):

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __call__(self):
        return np.random.uniform(self.a, self.b)


class LogUniformDist(object):

    def __init__(self, a, b):
        assert a > 0 and b > 0, 'Parameters of LogUniformDist should be positive!'
        self.a, self.b = np.log(a), np.log(b)

    def __call__(self):
        return np.exp(np.random.uniform(self.a, self.b))


class NormalDist(object):

    def __init__(self, mean, sigma):
        self.mean, self.sigma = mean, sigma

    def __call__(self):
        return np.random.randn() * self.sigma + self.mean


def RandomRotateStretchCropResize(img, points, bbox,
                                  rotate=None,
                                  scale=None,
                                  translate=None,
                                  resizeTo=None):
    '''
        Randomly rotate an image first.
        Then randomly stretch and translate bounding box, and crop image to that box.

        @param img: PIL image
        @param points: numpy array of [x, y]
        @param bbox: bounding box, (x1, y1, x2, y2)
        @param scale: a (tuple of) random variable, in scalar.
        @param roate: a (tuple of) random variable, in degree.
        @param translate: a (tuple of) random variable, in pixel.
        @param resizeTo: a tuple of new image size.

        @return (new_img, new_points, mat)
            np.dot([x, y, 1], mat) = [new_points_x, new_points_y, 1]

    '''

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    mat = np.eye(3)

    # 1. rotate
    if rotate is not None:
        deg = rotate()
    else:
        deg = 0.0

    ct = np.array(img.size, dtype=np.float) / 2
    mat = np.dot(mat, [[1., 0, 0], [0, 1, 0], [-ct[0], -ct[1], 1]])
    rotrad = np.deg2rad(deg)
    c, s = np.cos(rotrad), np.sin(rotrad)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    mat = np.dot(mat, rot)

    nimg = img.rotate(deg, Image.BICUBIC, expand=True)
    nct = np.array(nimg.size, dtype=np.float) / 2
    mat = np.dot(mat, [[1., 0, 0], [0, 1, 0], [nct[0], nct[1], 1]])

    bbox = np.array(bbox, dtype=np.float).reshape(-1, 2)
    xmin, xmax = bbox.min(axis=0), bbox.max(axis=0)
    xct = 0.5 * (xmin + xmax)
    nxct = np.dot([[xct[0], xct[1], 1]], mat)[0, 0:2]

    bbox = bbox - xct + nxct
    xmin, xmax = bbox.min(axis=0), bbox.max(axis=0)
    xct = 0.5 * (xmin + xmax)

    # 2. stretch
    if scale is not None:
        if isinstance(scale, tuple) or isinstance(scale, list):
            xs, ys = scale[0](), scale[1]()
        else:
            xs = ys = scale()

        bbox = (bbox - xct) * [xs, ys] + xct
        #print (bbox)
        #print (mat)

    # 3. translate
    if translate is not None:
        if isinstance(translate, tuple) or isinstance(translate, list):
            xt, yt = translate[0](), translate[1]()
        else:
            xt = yt = translate()

        bbox += [xt, yt]

    bbox = np.round(bbox).astype(np.int).flatten()
    # print(bbox)
    mat = np.dot(mat, [[1, 0, 0], [0, 1, 0], [-bbox[0], -bbox[1], 1]])
    nimg = nimg.crop(bbox)

    # 4. resize
    if resizeTo is not None:
        oldSize = np.array(nimg.size, dtype=np.float)
        newSize = np.round(np.array(resizeTo, dtype=np.float))

        nimg = nimg.resize(tuple(newSize.astype(np.int32)), Image.BICUBIC)  # LANCZOS
        #print(newSize, oldSize)
        xs, ys = newSize / np.maximum(oldSize, 1.0)

        mat = np.dot(mat, [[xs, 0, 0], [0.0, ys, 0], [0, 0, 1]])

    return (nimg, np.dot(np.concatenate([points, [[1]] * len(points)], axis=1), mat)[:, 0:2], mat)


def RandomContrastBrightnessNoise(img, contrast=None, brightness=None, noise=None):
    im = np.array(img)
    assert im.dtype == np.uint8, "Only uint8 type is supported (for now)."
    im = im.astype(np.float)

    if contrast is not None:
        im -= 127
        im *= contrast()
        im += 127
    if brightness is not None:
        im += brightness()
    if noise is not None:
        im += np.random.randn(*im.shape) * noise()

    im[im > 255] = 255
    im[im < 0] = 0
    ret = Image.fromarray(im.astype(np.uint8))
    return ret


if __name__ == '__main__':
    from PIL import ImageDraw

    im = Image.open('figure_1.png').convert('RGB')

    pts = np.array([[448., 165], [428., 165], [448., 185]])
    bbox = np.array([[398., 115], [498, 215]])
    draw = ImageDraw.Draw(im)
    draw.line([tuple(bbox[0]), tuple(bbox[1])], fill=(0, 0, 255), width=3)
    im.save('debug.png')

    irotate = 0
    iscale = 1.1
    itranslate = 0

    for i in range(5):
        nim, npts, mat = RandomRotateStretchCropResize(im, pts, bbox,
                                                       rotate=UniformDist(-irotate, irotate),
                                                       scale=[UniformDist(iscale, iscale), UniformDist(iscale, iscale)],
                                                       translate=[UniformDist(itranslate, itranslate), UniformDist(itranslate, itranslate)],
                                                       resizeTo=(64, 64))

        nim = RandomContrastBrightnessNoise(nim, contrast=LogUniformDist(0.5, 2.0), brightness=UniformDist(-50, 50), noise=NormalDist(0.0, 3.0))
        draw = ImageDraw.Draw(nim)
        draw.line(tuple(npts[0]) + tuple(npts[1]), fill=(0, 255, 0), width=3)
        draw.line(tuple(npts[0]) + tuple(npts[2]), fill=(255, 0, 0), width=3)

        nim.save('local%03d.png' % i)
