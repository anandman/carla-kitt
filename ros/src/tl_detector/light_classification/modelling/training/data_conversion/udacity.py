import yaml
import os
from PIL import Image

class Converter(object):
    def __init__(self, path):
        self.path = path
        self.loaded_file = yaml.load(open(path, 'rb').read())

    def get_absolute_paths(self):
        abs_path_to_raw = os.path.abspath(os.path.join(os.path.dirname(self.path), "../raw"))
        return [abs_path_to_raw + entity["annotations"]["filename"] for entity in self.loaded_file]

    def get_annotations(self):
        image_absolute_paths = self.get_absolute_paths()
        image_entities = [entity["annotations"] for entity in self.loaded_file]

        image_entities = zip(image_absolute_paths, image_entities)

        annotations = []
        for image_absolute_path, image_entity in image_entities:
            im = Image.open(image_absolute_path)
            im_width, im_height = im.size
            for raw_annotation in image_entity:
                annotations.append({
                    "class": raw_annotation["class"].upper(),
                    "x_min": raw_annotation["xmin"],
                    "x_max": raw_annotation["x_width"] + raw_annotation["xmin"],
                    "y_min": raw_annotation["ymin"],
                    "y_max": raw_annotation["y_height"] + raw_annotation["ymin"],
                    "im_width": im_width,
                    "im_height": im_height
                })

        return annotations
