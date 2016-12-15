from panda3d.core import *
from direct.showbase.ShowBase import ShowBase
import math
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import os
import random


class Renderer(ShowBase):

    def __init__(self, depth, background, attenuation=True):

        self.generate_depth = depth
        self.replace_background = background
        self.attenuation = attenuation

        loadPrcFileData("", "window-type offscreen")
        loadPrcFileData('', 'win-size 128 128')
        ShowBase.__init__(self)
        base.disableMouse()
        base.setBackgroundColor(0.5, 0.5, 0.5)

        # setup scene
        self.scene = NodePath("Scene")
        self.scene.reparentTo(self.render)
        self.scene.setScale(1, 1, 1)
        self.scene.setTwoSided(True)
        self.scene.setPos(0, 0, 0)
        self.scene.setHpr(0, 0, 0)

        self.near_plane = 0.1
        self.far_plane = 5.0

        self.resolution = 128
        self.max_16bit_val = 65535

        self.light_sources = []
        self.light_nodes = []

        self.createLightSources()

        self.alight = AmbientLight('alight')
        self.alight.setColor(VBase4(10, 10, 10, 1))
        self.alnp = self.render.attachNewNode(self.alight)
        self.render.setLight(self.alnp)

        base.camLens.setNear(self.near_plane)
        base.camLens.setFar(self.far_plane)

        # prepare texture and camera for depth rendering
        if self.generate_depth is True:
            self.depth_tex = Texture()
            self.depth_tex.setFormat(Texture.FDepthComponent)
            self.depth_buffer = base.win.makeTextureBuffer(
                'depthmap', self.resolution, self.resolution,
                self.depth_tex, to_ram=True)
            self.depth_cam = self.makeCamera(self.depth_buffer,
                                             lens=base.camLens)
            print(self.depth_cam.node().getLens().getFilmSize())
            self.depth_cam.reparentTo(base.render)

        # list of models in memory
        self.models = []
        self.backgrounds = []
        self.model = None

    def delete(self):
        self.alnp.removeNode()
        for n in self.light_nodes:
            n.removeNode()
        for m in self.models:
            self.loader.unloadModel(m)
        base.destroy()

    def createLightSources(self):
        for i in range(0, 7):
            plight = PointLight('plight')
            if self.attenuation is True:
                plight.setAttenuation((1, 0, 1))
            plight.setColor(VBase4(0, 0, 0, 0))
            self.light_sources.append(plight)
            plnp = self.render.attachNewNode(plight)
            plnp.setPos(3, 3, 3)
            render.setLight(plnp)
            self.light_nodes.append(plnp)

    def activateLightSources(self, light_sources, spher=True):
        i = 0
        for lght in light_sources:
            lp_rad = lght[0]
            lp_el = lght[1]
            lp_az = lght[2]
            lp_int = lght[3]
            if spher:
                self.light_nodes[i].setPos(
                    lp_rad*math.cos(lp_el)*math.cos(lp_az),
                    lp_rad*math.cos(lp_el)*math.sin(lp_az),
                    lp_rad*math.sin(lp_el))
            else:
                self.light_nodes[i].setPos(lp_rad, lp_el, lp_az)
            self.light_sources[i].setColor(VBase4(lp_int, lp_int, lp_int, 1))
            i += 1

    def deactivateLightSources(self):
        for i in range(0, 7):
            self.light_sources[i].setColor(VBase4(0, 0, 0, 0))

    def textureToImage(self, texture):
        im = texture.getRamImageAs("RGB")
        strim = im.getData()
        image = np.fromstring(strim, dtype='uint8')
        image = image.reshape(self.resolution, self.resolution, 3)
        image = np.flipud(image)
        return image

    def textureToString(self, texture):
        im = texture.getRamImageAs("RGB")
        return im.getData()

    def setCameraPosition(self, rad, el, az):
        xx = rad*math.cos(el)*math.cos(az)
        yy = rad*math.cos(el)*math.sin(az)
        zz = rad*math.sin(el)
        self.camera.setPos(xx, yy, zz)
        self.camera.lookAt(0, 0, 0)

        if self.generate_depth is True:
            self.depth_cam.setPos(xx, yy, zz)
            self.depth_cam.lookAt(0, 0, 0)

    def loadImagenetBackgrounds(self, path, start, count):
        for i in range(start, start + count):
            fname = "ILSVRC2012_preprocessed_val_" + str(i).zfill(8) + ".JPEG"
            im = scipy.misc.imread(os.path.join(path, fname))
            im = scipy.misc.imresize(im,
                                     (self.resolution,
                                      self.resolution, 3), interp='nearest')
            self.backgrounds.append(im)

    def selectModel(self, model_ind):
        self.model = self.models[model_ind]
        self.model.reparentTo(self.scene)

    def unselectModel(self, model_ind):
        self.model.detachNode()
        self.model = None

    def loadModels(self, model_names, models_path):
        mn = []
        for i in range(0, len(model_names)):
            mn.append(
                models_path + "/" +
                model_names[i].rstrip() + "/model.bam")

        self.models = self.loader.loadModel(mn)

    def renderView(self, camera_pos, light_sources,
                   blur, blending, spher=True, default_bg_setting=True):

        self.setCameraPosition(camera_pos[0],
                               math.radians(camera_pos[1]),
                               math.radians(camera_pos[2]))

        self.activateLightSources(light_sources, spher)

        base.graphicsEngine.renderFrame()
        tex = base.win.getScreenshot()
        im = self.textureToImage(tex)

        if self.generate_depth is True:
            depth_im = PNMImage()
            self.depth_tex.store(depth_im)

            depth_map = np.zeros([self.resolution,
                                  self.resolution], dtype='float')
            for i in range(0, self.resolution):
                for j in range(0, self.resolution):
                    depth_val = depth_im.getGray(j, i)
                    depth_map[i, j] = self.far_plane * self.near_plane /\
                        (self.far_plane - depth_val *
                            (self.far_plane - self.near_plane))
                    depth_map[i, j] = depth_map[i, j] / self.far_plane

            dm_uint = np.round(depth_map * self.max_16bit_val).astype('uint16')

        if self.replace_background is True and default_bg_setting is True:
            mask = (dm_uint == self.max_16bit_val)
            temp = np.multiply(
                mask.astype(dtype=np.float32).reshape(
                        self.resolution, self.resolution, 1), im)
            im = im - temp
            blurred_mask = scipy.ndimage.gaussian_filter(
                mask.astype(dtype=np.float32), blending)
            inv_mask = (blurred_mask - 1)*(-1)

            bg_ind = random.randint(0, len(self.backgrounds)-1)
            im = np.multiply(
                self.backgrounds[bg_ind],
                blurred_mask.reshape(self.resolution, self.resolution, 1)) + \
                np.multiply(im, inv_mask.reshape(self.resolution,
                                                 self.resolution, 1))

            im = scipy.ndimage.gaussian_filter(im, sigma=blur)

        im = im.astype(dtype=np.uint8)

        self.deactivateLightSources()

        return im, dm_uint
