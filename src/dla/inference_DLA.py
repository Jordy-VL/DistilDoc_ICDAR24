import os
import logging
from collections import OrderedDict
from typing import List, Dict
from tqdm import tqdm
import numpy as np
import time
import cv2
import torch


from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    register_coco_instances,
    build_detection_test_loader,
    DatasetCatalog,
    MetadataCatalog,
    DatasetMapper,
)
from detectron2.data.samplers import InferenceSampler
import detectron2.data.transforms as T
from detectron2.engine import default_argument_parser, launch
from detectron2.evaluation import (
    inference_on_dataset,
    print_csv_format,
    DatasetEvaluator,
    COCOEvaluator,
)
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
import json
from train_vanilla_teacher_net import setup, do_test


from detectron2.utils.analysis import FlopCountAnalysis
from fvcore.nn import flop_count_str, flop_count_table, parameter_count

DATAROOT = "./data/DocLayNet_core"

from fvcore.nn.print_model_statistics import _format_size

logger = logging.getLogger("detectron2")


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


# inference statistics
def warmup(model, inputs, N=10):
    for i in range(N):
        out = model(inputs)
    torch.cuda.synchronize()


def measure_time(model, inputs, N=10):
    warmup(model, inputs, N=N)
    ts = []
    for i in range(N):
        s = time.time()
        out = model(inputs)
        ts.append(time.time() - s)
    torch.cuda.synchronize()
    return np.array(ts)


def fmt_res(data):
    # format as dict
    res = {
        "mean": float(data.mean()),
        "std": float(data.std()),
        "prep": str(data.mean().round(2)) + "±" + str(data.std().round(3)),
        "min": float(data.min()),
        "max": float(data.max()),
        "throughput": int(1.0 // data.mean()),
    }  # images per second
    return res


def inference_statistics(model, inputs):
    if inputs is None:
        input_shape = (3, 224, 224)
        inputs = (torch.randn((1, *input_shape)),)

    flops_ = FlopCountAnalysis(model, inputs)
    flops = _format_size(flops_.total())
    params = _format_size(parameter_count(model)[""])

    # wont work
    # activations_ = ActivationCountAnalysis(model, inputs)
    # activations = _format_size(activations_.total())

    flop_table = flop_count_table(
        flops=flops_,
        # activations=activations_,
        show_param_shapes=True,
    )
    flop_str = flop_count_str(flops=flops_, activations=None)

    print("\n" + flop_str)
    print("\n" + flop_table)

    split_line = "=" * 30
    print(f'{split_line}\nInput shape: {inputs[0]["image"].shape}\n' f"Flops: {flops}\nParams: {params}\n{split_line}")

    # timings
    timings = measure_time(model, inputs, N=100)

    return {"flops": flops, "params": params, "timings": fmt_res(timings)}


def load_rvl_cdip_images():
    path = os.environ["INFERENCE_DATASET_PATH"]
    common = []
    count = 0
    for folder in sorted(os.listdir(path)):
        for filename in sorted(os.listdir(os.path.join(path, folder))):
            img = cv2.imread(os.path.join(path, folder, filename))
            if img is not None:
                data = dict(
                    file_name=os.path.join(path, folder, filename),
                    # height=img.shape[0],
                    # width=img.shape[1],
                    image_id=count,
                )
                common.append(data)
                count += 1
    return common


def load_single_image():
    path = os.environ["INFERENCE_DATASET_PATH"]
    img = cv2.imread(path)
    return [
        dict(
            file_name=path,
            # height=img.shape[0],
            # width=img.shape[1],
            image_id=0,
        )
    ]


def load_images_from_path():
    # path="/data2/users/XXX/DocVQA/images"\
    path = os.environ["INFERENCE_DATASET_PATH"]
    common = []
    count = 0
    for filename in sorted(os.listdir(path)):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            data = dict(
                file_name=os.path.join(path, filename),
                # height=img.shape[0],
                # width=img.shape[1],
                image_id=count,
            )
            common.append(data)
            count += 1
    return common


def category_id_map(i, return_map=False):
    map = [
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title",
    ]
    if return_map:
        return map
    return map[i]


def filter_instances(predictions, conf_threshold=0.5):
    image_size = (predictions.image_size[0], predictions.image_size[1])
    ret = Instances(image_size)
    chosen = torch.where((predictions.scores > conf_threshold))[0]
    ret.scores = predictions.scores[chosen]
    ret.pred_boxes = predictions.pred_boxes[chosen]
    ret.pred_classes = predictions.pred_classes[chosen]
    ret.labels = [category_id_map(x) for x in ret.pred_classes]
    return ret


def localize_datasplit(dataset_name, image_sample):
    if "docvqa" in dataset_name:
        path = "/data2/users/XXX/DocVQA/images"  # ALL
    elif "infographics_vqa" in dataset_name:
        path = "downstream/InfographicsVQA"
    elif "rvl_cdip" in dataset_name:
        if "test" in dataset_name:
            path = "/data/users/XXX/datasets/rvlcdip/Image/Test_Data"
        elif "train" in dataset_name:
            path = "/data/users/XXX/datasets/rvlcdip/Image/Train_Data"
        elif "val" in dataset_name:
            path = "/data/users/XXX/datasets/rvlcdip/Image/Valid_Data"
    elif image_sample:
        path = image_sample
    os.environ["INFERENCE_DATASET_PATH"] = path


def main(args):
    cfg = setup(args, update_output=True)

    dataset_name = args.inference_dataset  #'rvl_cdip'
    if not dataset_name and args.image_sample:
        dataset_name = "sample"
    localize_datasplit(dataset_name, args.image_sample)

    if "docvqa" in dataset_name:
        fx = load_images_from_path
    elif "infographics_vqa" in dataset_name:
        fx = load_images_from_path
    elif "rvl_cdip" in dataset_name:
        fx = load_rvl_cdip_images
    elif "sample" in dataset_name:
        fx = load_single_image
    elif "doclaynet" in dataset_name:
        # load doclaynet and evaluator

        register_coco_instances("doclaynet_train", {}, f"{DATAROOT}/COCO/train.json", f"{DATAROOT}/PNG")

        register_coco_instances("doclaynet_val", {}, f"{DATAROOT}/COCO/val.json", f"{DATAROOT}/PNG")

        register_coco_instances("doclaynet_test", {}, f"{DATAROOT}/COCO/test.json", f"{DATAROOT}/PNG")
    else:
        raise NotImplementedError("SOMEONE TO IMPLEMENT THIS")

    # # need to swap to checkpoint path; where the model_final.pth and config.yml are saved
    # cfg.defrost()
    # cfg.MODEL.WEIGHTS = os.path.join(os.path.dirname(args.config_file), "model_final.pth")
    # cfg.OUTPUT_DIR = os.path.join(os.path.dirname(args.config_file), "inference", args.inference_dataset)
    # cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    IMAGES_OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, "images")
    os.makedirs(IMAGES_OUTPUT_DIR, exist_ok=True)

    # Register datasets
    MetadataCatalog.get("doclaynet_train").thing_classes = category_id_map(None, return_map=True)
    MetadataCatalog.get("doclaynet_val").thing_classes = category_id_map(None, return_map=True)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)  # this will load model in place

    if "doclaynet" in dataset_name:
        print("evaluating doclaynet")
        do_test(cfg, model)

    metadata = MetadataCatalog.get("doclaynet_train")

    DatasetCatalog.register(dataset_name, fx)
    mapper = DatasetMapper(cfg, is_train=False, augmentations=[T.Resize(cfg.MODEL.VIT.IMG_SIZE)])
    data_loader = build_detection_test_loader(
        cfg,
        dataset_name,
        mapper=mapper,
        sampler=InferenceSampler(len(fx())),
    )
    model.eval()

    # do flops analysis here
    inference_results = inference_statistics(model, inputs=next(iter(data_loader)))
    print(json.dumps(inference_results, indent=4))
    dump_json_path = os.path.join(cfg.OUTPUT_DIR, "inference_results.json")
    save_json(inference_results, dump_json_path)
    # next(iter(data_loader))).shape

    predictions = []
    for i, batch in tqdm(enumerate(iter(data_loader))):
        with torch.no_grad():
            # predictions and save to disk
            outputs = model(batch)
            outputs = outputs[0]["instances"].to("cpu")
            outputs = filter_instances(outputs, conf_threshold=args.confidence_threshold)
            outputs.filename = [batch[0]["file_name"]] * len(outputs)
            predictions.append(outputs)

            if "rvl_cdip" in dataset_name or (i % 10) == 0:  # only needed for classification
                # visualization and write to disk
                img = cv2.imread(batch[0]["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
                basename = os.path.basename(batch[0]["file_name"])
                vis = Visualizer(img, metadata)
                vis_pred = vis.draw_instance_predictions(outputs).get_image()
                cv2.imwrite(os.path.join(IMAGES_OUTPUT_DIR, basename), vis_pred[:, :, ::-1])

    file_path = os.path.join(cfg.OUTPUT_DIR, "instances_predictions.pth")
    with open(file_path, "wb") as f:
        torch.save(predictions, f)


if __name__ == "__main__":
    # python3 inference_DLA.py --config-file ./output/teacher/vitb_imagenet_doclaynet_tecaher/config.yaml --num-gpus 1
    parser = default_argument_parser()
    parser.add_argument("--inference_dataset", type=str, default="")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--image_sample", type=str, default="")
    parser.add_argument("--evaluate_doclaynet", action="store_true", help="evaluate doclaynet")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
