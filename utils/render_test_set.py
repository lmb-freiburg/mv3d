from renderer import *

import os
import sys

bg_path = "../data/bg_imagenet/val"
bam_path = "../data/obj_cars"

render_bg = True
normal_test = True
samples_per_model = 10


def main(argv=None):

    if normal_test is True:
        models_file = "../data/cars_test_normal.txt"
        output_path = "../data/test_normal_rendered"
    else:
        models_file = "../data/cars_test_difficult.txt"
        output_path = "../data/test_difficult_rendered"

    if render_bg is True:
        output_path = os.path.join(output_path, "bg")
    else:
        output_path = os.path.join(output_path, "nobg")

    with open(models_file) as f:
        model_names = f.readlines()
    print(model_names)
    model_count = len(model_names)

    rend = Renderer(True, render_bg)
    rend.loadImagenetBackgrounds(bg_path, 1, 100)
    rend.loadModels(model_names[0:model_count], bam_path)

    for model_num in range(0, model_count):
        model_name_str = model_names[model_num].rstrip()
        os.mkdir(os.path.join(output_path, model_name_str))

        rend.selectModel(model_num)

        for sample_num in range(0, samples_per_model):

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

                im, dm = rend.renderView([rad, el, az], lights,
                                         blur, blending,
                                         default_bg_setting=bool(1-im_num))

                output_name = os.path.join(output_path,
                                           model_name_str,
                                           str(sample_num).zfill(3) +
                                           "_cl_" + str(im_num) + "_" +
                                           str(int(rad*10)) + "_" + str(az) +
                                           "_" + str(el) + ".png")
                scipy.misc.toimage(im, cmin=0, cmax=255).save(output_name)

                output_name = os.path.join(output_path,
                                           model_name_str,
                                           str(sample_num).zfill(3) +
                                           "_dm_" + str(im_num) + "_" +
                                           str(int(rad*10)) + "_" + str(az) +
                                           "_" + str(el) + ".png")
                scipy.misc.toimage(dm, cmin=0, cmax=65535,
                                   low=0, high=65535, mode='I').save(
                                   output_name)

        rend.unselectModel(model_num)


if __name__ == "__main__":
    sys.exit(main())
