import yaml
import os
from PIL import Image

DIRNAME = os.path.dirname(os.path.abspath(__file__))
class Converter(object):
    def __init__(self, filename):
        self.path = DIRNAME + "/../data/yaml/" + filename
        self.name, _ = os.path.splitext(filename)
        self.loaded_file = yaml.load(open(self.path, 'rb').read())

    def get_absolute_paths(self):
        abs_path_to_raw = os.path.abspath(os.path.join(os.path.dirname(self.path), "../raw/", self.name))
        return [abs_path_to_raw + "/" + entity["filename"] for entity in self.loaded_file]

    def get_image_sizes(self):
        image_absolute_paths = self.get_absolute_paths()
        image_sizes = []

        for image_absolute_path in image_absolute_paths:
            im = Image.open(image_absolute_path)
            im_width, im_height = im.size
            image_sizes.append({
                "im_width": im_width,
                "im_height": im_height
            })

        return image_sizes


    def get_annotations(self):
        image_entities = [entity["annotations"] for entity in self.loaded_file]

        annotations = []
        for image_entity in image_entities:
            image_annotations = []
            for raw_annotation in image_entity:
                image_annotations.append({
                    "class": raw_annotation["class"].upper(),
                    "x_min": raw_annotation["xmin"],
                    "x_max": raw_annotation["x_width"] + raw_annotation["xmin"],
                    "y_min": raw_annotation["ymin"],
                    "y_max": raw_annotation["y_height"] + raw_annotation["ymin"],
                })
            annotations.append(image_annotations)

        return annotations
