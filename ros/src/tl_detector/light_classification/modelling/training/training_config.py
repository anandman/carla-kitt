# make the config files that we need to test training combinations on.
from jinja2 import Environment, FileSystemLoader
import itertools
import os
import glob

DIRNAME = os.path.dirname(os.path.abspath(__file__))

example_inception_groups = [
    {
        #./data/converted/records/bosch_test
        "train_test_record_name": [{"train": "bosch_test", "test": "bosch_test"}],
        "num_classes": [4],
        "num_steps": [10000],
        "dimensions": [(720, 1280), (1096, 1368), (600, 800)],
        # "min_dimension": [720],
        # "max_dimension": [1280],
    },
    {
        "train_test_record_name": [{"train": "udacity_real", "test": "udacity_real"}],
        "num_classes": [4],
        "num_steps": [10000],
        "dimensions": [(1096, 1368)],
    },
    {
        "train_test_record_name": [{"train": "udacity_sim", "test": "udacity_sim"}],
        "num_classes": [4],
        "num_steps": [10000],
        "dimensions": [(600, 800)],
    },
    {
        "train_test_record_name": [{"train": "udacity_all", "test": "udacity_all"}],
        "num_classes": [4],
        "num_steps": [10000],
        "dimensions": [(600, 800), (1096, 1368)],
    },
    {
        "train_test_record_name": [{"train": "bosch_test_and_udacity_real", "test": "bosch_test_and_udacity_real"}],
        "num_classes": [4],
        "num_steps": [10000],
        "dimensions": [(720, 1280), (1096, 1368)],
    },
    {
        "train_test_record_name": [{"train": "bosch_test_and_udacity_all", "test": "bosch_test_and_udacity_all"}],
        "num_classes": [4],
        "num_steps": [10000],
        "dimensions": [(720, 1280), (1096, 1368), (600, 800)],
    }
]

example_mobile_groups = [
    {
        #./data/converted/records/bosch_test
        "train_test_record_name": [{"train": "bosch_test", "test": "bosch_test"}],
        "num_classes": [4],
        "num_steps": [10000],
        # "min_dimension": [720],
        # "max_dimension": [1280],
    },
    {
        "train_test_record_name": [{"train": "udacity_real", "test": "udacity_real"}],
        "num_classes": [4],
        "num_steps": [10000],
    },
    {
        "train_test_record_name": [{"train": "udacity_sim", "test": "udacity_sim"}],
        "num_classes": [4],
        "num_steps": [10000],
    },
    {
        "train_test_record_name": [{"train": "udacity_all", "test": "udacity_all"}],
        "num_classes": [4],
        "num_steps": [10000],
    },
    {
        "train_test_record_name": [{"train": "bosch_test_and_udacity_real", "test": "bosch_test_and_udacity_real"}],
        "num_classes": [4],
        "num_steps": [10000],
    },
    {
        "train_test_record_name": [{"train": "bosch_test_and_udacity_all", "test": "bosch_test_and_udacity_all"}],
        "num_classes": [4],
        "num_steps": [10000],
    }
]

def make_faster_rcnn_inception_v2_configs(groups):
    template_name = "faster_rcnn_inception_v2.j2"
    # Read Jinja 2 template
    j2_env = Environment(loader=FileSystemLoader(DIRNAME + '/config/templates'))
    # Load j2 file
    for group in groups:
        iterations = itertools.product(group['train_test_record_name'],
                          group['num_classes'],
                          group['num_steps'],
                          group['dimensions'])

        for iteration in iterations:
            fname = "_".join(["faster_rcnn_inception_v2",
                iteration[0]["train"] + "-" + iteration[0]["test"],
                "ncls_" + str(iteration[1]),
                "nstep_" + str(iteration[2]),
                str(iteration[3][0]), \
                str(iteration[3][1])]) + ".config"

            train_record = "./data/converted/records/" + iteration[0]["train"] + ".record"
            train_pbtxt = "./data/converted/labels/" + iteration[0]["train"] + ".pbtxt"
            test_record = "./data/converted/records/" + iteration[0]["test"] + ".record"
            test_pbtxt = "./data/converted/labels/" + iteration[0]["test"] + ".pbtxt"
            num_classes = iteration[1]
            num_steps = iteration[2]
            min_dimension = iteration[3][0]
            max_dimension = iteration[3][1]

            with open(DIRNAME + "/config/output/" + fname, "w") as f:
                f.write(j2_env.get_template(template_name).render(
                    train_record = train_record,
                    train_pbtxt = train_pbtxt,
                    test_record = test_record,
                    test_pbtxt = test_pbtxt,
                    num_classes = num_classes,
                    num_steps = num_steps,
                    min_dimension = min_dimension,
                    max_dimension = max_dimension,
                ))

    # TODO Later - option to resume training.


def make_ssd_mobilenet_v2_configs(groups):
    # Read Jinja 2 template
    template_name = "ssd_mobilenet_v2.j2"
    j2_env = Environment(loader=FileSystemLoader(DIRNAME + '/config/templates'))
    # Load j2 file
    for group in groups:
        iterations = itertools.product(group['train_test_record_name'],
                          group['num_classes'],
                          group['num_steps'])

        for iteration in iterations:
            fname = "_".join(["ssd_mobilenet_v2",
                iteration[0]["train"] + "-" + iteration[0]["test"],
                "ncls_" + str(iteration[1]),
                "nstep_" + str(iteration[2])]) + ".config"

            train_record = "./data/converted/records/" + iteration[0]["train"] + ".record"
            train_pbtxt = "./data/converted/labels/" + iteration[0]["train"] + ".pbtxt"
            test_record = "./data/converted/records/" + iteration[0]["test"] + ".record"
            test_pbtxt = "./data/converted/labels/" + iteration[0]["test"] + ".pbtxt"
            num_classes = iteration[1]
            num_steps = iteration[2]

            with open(DIRNAME + "/config/output/" + fname,"w") as f:
                f.write(j2_env.get_template(template_name).render(
                    train_record = train_record,
                    train_pbtxt = train_pbtxt,
                    test_record = test_record,
                    test_pbtxt = test_pbtxt,
                    num_classes = num_classes,
                    num_steps = num_steps,
                ))

if __name__ == "__main__":
    # Clear output directory
    try:
        os.mkdir(DIRNAME + "/config/output/")
    except OSError:
        pass
    for filename in glob.glob(DIRNAME + "/config/output/*"):
        os.remove(filename)
    make_faster_rcnn_inception_v2_configs(example_inception_groups)
    make_ssd_mobilenet_v2_configs(example_mobile_groups)
