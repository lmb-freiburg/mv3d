from renderer import *

import random


class RealTimeRenderer():

    def __init__(self, batch_size):
        self.total_bg_count = 40000
        self.render_bg = False
        self.num_iterations = 1000
        self.bg_batch_size = 100

        self.bg_path = "../data/bg_imagenet/val"
        self.bam_path = "../data/obj_cars"

        self.iteration = 0
        self.rend = None
        self.batch_size = batch_size

    def load_model_names(self, file_path):
        with open(file_path) as f:
            self.model_names = f.readlines()
        self.model_names = [x.strip() for x in self.model_names]

    def render(self, rad_factor=100.0):

        if self.iteration + self.batch_size >= self.num_iterations:
            self.iteration = 0
            self.rend.delete()

        print("iteration: " + str(self.iteration))

        if self.iteration == 0:
            model_num = random.randint(
                0, len(self.model_names) - self.bg_batch_size)
            self.rend = Renderer(True, self.render_bg)
            self.rend.loadImagenetBackgrounds(
                self.bg_path, random.randint(100, self.total_bg_count),
                self.bg_batch_size)
            self.rend.loadModels(
                self.model_names[model_num:model_num + self.bg_batch_size],
                self.bam_path)

        output_color = []
        output_color.append([])
        output_color.append([])
        output_depth = []
        output_depth.append([])
        output_depth.append([])
        output_labels = []

        for i in range(0, self.batch_size):
            model_id = random.randint(0, self.bg_batch_size-1)
            self.rend.selectModel(model_id)

            num_light = random.randint(2, 4)
            lights = []
            for nl in range(0, num_light):
                light_pos = [random.random()*2. + 2.5,
                             random.randint(-90, 90),
                             random.randint(0, 360),
                             random.randint(10, 15)]
                lights.append(light_pos)

            for im_num in range(0, 2):
                rad = np.around(random.random() * 0.6 + 1.7, decimals=1)
                el = int(random.random() * 50 - 10)
                az = int(random.random() * 360)
                if im_num == 0:
                    blur = random.random() * 0.4 + 0.2
                    blending = random.random() * 0.3 + 1
                else:
                    blur = 0
                    blending = 0
                    output_labels.append([rad / rad_factor,
                                         math.sin(math.radians(el)),
                                         math.cos(math.radians(el)),
                                         math.sin(math.radians(az)),
                                         math.cos(math.radians(az))])

                im, dm = self.rend.renderView(
                    [rad, el, az], lights, blur, blending,
                    default_bg_setting=bool(1-im_num))

                output_color[im_num].append((im / 255.0 - 0.5) * 1.5)
                output_depth[im_num].append((dm / 65535.0 - 0.5) * 1.5)

            self.rend.unselectModel(model_id)

            self.iteration += 1

        return np.asarray(output_color[0]), np.asarray(output_depth[0]), \
            np.asarray(output_color[1]), np.asarray(output_depth[1]), \
            np.asarray(output_labels)
