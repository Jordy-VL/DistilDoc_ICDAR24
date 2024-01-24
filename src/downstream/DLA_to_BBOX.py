import os
from PIL import Image
import json
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
import xml.etree.ElementTree as ET
from space_layout import space_layout


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def dump_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def read_content(xml_file: str):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []
    labels = []

    for boxes in root.iter("object"):
        filename = root.find("filename").text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)
        labels.append(boxes.find("name").text)

    return filename, list_with_all_boxes, labels


DLA_LABELS = [
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


def intersection_over_union(bbox, bbox2):
    x1, y1, x2, y2 = bbox
    x3, y3, x4, y4 = bbox2

    xA = max(x1, x3)
    yA = max(y1, y3)
    xB = min(x2, x4)
    yB = min(y2, y4)

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (x2 - x1) * (y2 - y1)
    boxBArea = (x4 - x3) * (y4 - y3)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def rectContains(bbox, pt, tolerance=20):  # 20 piels tolerance
    # FIXED: this assumes a,b are the top-left coordinate of the rectangle and (c,d) be its width and height. OpenCV Contour Features
    # a < x0 < a+c and b < y0 < b + d
    rect = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
    logic = rect[0] < pt[0] < rect[0] + rect[2] and rect[1] < pt[1] < rect[1] + rect[3]
    # logic = (
    #     rect[0] - tolerance < pt[0] < rect[0] + rect[2] + tolerance
    #     and rect[1] - tolerance < pt[1] < rect[1] + rect[3] + tolerance
    # )
    return logic


def bbox_area_overlap_for_P_percent(smaller_bbox, larger_bbox, P=0.5):
    """Checks if a smaller bounding box falls for 50% of its area inside another larger bounding box.

    Args:
      smaller_bbox: A tuple containing the coordinates of the smaller bounding box, in the format `(x1, y1, x2, y2)`, where `x1, y1` form the top-left corner and `x2, y2` form the right-bottom corner.
      larger_bbox: A tuple containing the coordinates of the larger bounding box, in the format `(x1, y1, x2, y2)`, where `x1, y1` form the top-left corner and `x2, y2` form the right-bottom corner.

    Returns:
      True if the smaller bounding box falls for 50% of its area inside the larger bounding box, False otherwise.
    """

    # Calculate the area of the smaller bounding box.
    smaller_bbox_area = (smaller_bbox[2] - smaller_bbox[0]) * (smaller_bbox[3] - smaller_bbox[1])

    # Calculate the area of the intersection between the two bounding boxes.
    intersection_area = calculate_intersection_area_between_two_bounding_boxes_with_bounding_box_format_of_x1_y1_x2_y2(
        smaller_bbox, larger_bbox
    )

    # If the area of the intersection is greater than 50% of the area of the smaller bounding box, then the smaller bounding box falls for 50% of its area inside the larger bounding box.
    return intersection_area > smaller_bbox_area * P


def calculate_intersection_area_between_two_bounding_boxes_with_bounding_box_format_of_x1_y1_x2_y2(bbox1, bbox2):
    """Calculates the area of the intersection between two bounding boxes.

    Args:
      bbox1: A tuple containing the coordinates of the first bounding box, in the format `(x1, y1, x2, y2)`, where `x1, y1` form the top-left corner and `x2, y2` form the right-bottom corner.
      bbox2: A tuple containing the coordinates of the second bounding box, in the format `(x1, y1, x2, y2)`, where `x1, y1` form the top-left corner and `x2, y2` form the right-bottom corner.

    Returns:
      The area of the intersection between the two bounding boxes.
    """

    # Calculate the top-left and bottom-right coordinates of the intersection.
    intersection_top_left_x = max(bbox1[0], bbox2[0])
    intersection_top_left_y = max(bbox1[1], bbox2[1])
    intersection_bottom_right_x = min(bbox1[2], bbox2[2])
    intersection_bottom_right_y = min(bbox1[3], bbox2[3])

    # If the intersection has a negative width or height, then it is empty, so return an area of 0.
    if (
        intersection_bottom_right_x - intersection_top_left_x <= 0
        or intersection_bottom_right_y - intersection_top_left_y <= 0
    ):
        return 0

    # Otherwise, return the area of the intersection.
    return (intersection_bottom_right_x - intersection_top_left_x) * (
        intersection_bottom_right_y - intersection_top_left_y
    )


def check_box_closest(layout_box, bbox, i, closest_start, closest_end):
    """Check if bbox is the closest to layout_box
    Heuristic: closest in terms of diagonal to top-left corner and closest to bottom-right corner

    TODO: alternative (follow reading order left-to-right, top-to-bottom)
    1) highest/lowest fully enclosed bbox
    2) leftmost/rightmost fully enclosed bbox from the top-k highest
    """
    x1, y1, x2, y2 = layout_box
    start_distance = abs(x1 - bbox[0]) + abs(y1 - bbox[1])
    end_distance = abs(x2 - bbox[2]) + abs(y2 - bbox[3])

    # set condition on being fully captured within or at least to a degree of tolerance [iou of box]
    if start_distance < closest_start[1]:
        # if rectContains(layout_box, (bbox[0], [bbox1])):
        closest_start = (i, start_distance)
    if end_distance < closest_end[1]:
        # if rectContains(layout_box, (bbox[2], bbox[3])):
        closest_end = (i, end_distance)
    return closest_start, closest_end


def torch_iou(layout_box, bbox):
    import torch
    import torchvision.ops.boxes as bops

    box1 = torch.tensor([layout_box], dtype=torch.float)
    box2 = torch.tensor([bbox], dtype=torch.float)
    return bops.box_iou(box1, box2)


def check_box_within(layout_box, bbox, i, closest_start, closest_end, debug=False):
    """alternative heuristic (follow reading order left-to-right, top-to-bottom)

    The first and last index falling within the layout box is the one that is used

    Check with IoU if the bbox is fully contained within the layout box

    Current version is strict; TODO: iou check
    """
    x3, y3, x4, y4 = bbox

    iou = intersection_over_union(layout_box, bbox)
    # one of the corners is within the layout box
    captured = iou > 0.3  # some token bboxes are really wonky

    rect_check_start, rect_check_end = rectContains(layout_box, (x3, y3)), rectContains(layout_box, (x4, y4))
    # torch_iou_score = torch_iou(layout_box, bbox) #nothing wrong with my iou calculation

    overlap_check = bbox_area_overlap_for_P_percent(bbox, layout_box, P=0.5)
    if debug:
        print("IOU", iou, i, bbox, layout_box, captured, rect_check_start, rect_check_end, overlap_check)

    # first box falling within the layout box = start; assuming monotonicity in bboxes
    if closest_start is None:
        if captured or rect_check_start or overlap_check:
            closest_start = i

    if captured or rect_check_end or overlap_check:
        closest_end = i
    return closest_start, closest_end


def place_tokens(DLA, OCR, args=None):  # with args not very clean..
    """For a given set of DLA predictions, where each is an instance of (label, bbox),
        input XNL-like tags into the original OCRed tokens of a document.

        To do so, we will iterate over the token bounding boxes and for the given
        DLA bounding box (x1, y1, x2, y2), we will find the closest start and end

        This involves finding the enclosed (x1, y1) and (x2, y2) tokens that minimize L1-distance
        to respectively, the top-left and top-right corner of the DLA box.

        As each layout bbox (upto B different) requires ~BN time complexity, it would be better to iterate once over all tokens
        and keep a dictionary for closest start and end position per layout box. This would be ~N/B time complexity

        Finally, insert DLA start token and end token into the OCR tokens at the respective positions (one before and one after).

    Args:
        DLA (tuple): contains (x1, y1, x2, y2) for each layout box (B)
        OCR (dict): contains positions and tokens in DUE format
    """
    if "common_format" in OCR:
        OCR = OCR["common_format"]

    original_tokens = deepcopy(OCR["tokens"])
    boxes, labels = DLA["boxes"], DLA["labels"]

    if args.ignore_tokens:  # remove tokens from DLA
        remove_idx = [i for i, token in enumerate(labels) if token in args.ignore_tokens]
        boxes, labels = [box for i, box in enumerate(boxes) if i not in remove_idx], [
            token for i, token in enumerate(labels) if i not in remove_idx
        ]

    # starts, ends = [(None, 1000000)] * len(boxes), [(None, 1000000)] * len(boxes)
    starts, ends = [(None)] * len(boxes), [(None)] * len(boxes)

    for i, bbox in enumerate(OCR["positions"]):
        # find closest start and end for each bbox of each type
        for j, layout_box in enumerate(boxes):
            # TODO: this would not work for multi-page documents like DUDE (but we don't have those yet)
            starts[j], ends[j] = check_box_within(layout_box, bbox, i, starts[j], ends[j])
            # starts[j], ends[j] = check_box_closest(layout_box, bbox, i, starts[j], ends[j])

    # check for None values in starts and ends
    for j, layout_box in enumerate(boxes):
        if starts[j] is None or ends[j] is None:
            keep_start = deepcopy(starts[j]) if starts[j] is not None else None
            keep_end = deepcopy(ends[j]) if ends[j] is not None else None
            print("Missing start or end", layout_box, keep_start, keep_end)

            # closest diagonal heuristic as backup
            starts[j], ends[j] = [None, 1000000], [None, 1000000]  # index, distance
            for i, bbox in enumerate(OCR["positions"]):
                starts[j], ends[j] = check_box_closest(layout_box, bbox, i, starts[j], ends[j])
            starts[j] = keep_start if keep_start is not None else starts[j][0]
            ends[j] = keep_end if keep_end is not None else ends[j][0]
            print("New start and end", starts[j], ends[j])

    # TODO: optionally: check nesting so that we need to index <table_1> <table_2> etc.
    # print(starts, ends)

    # insert start and end tokens into new_tokens; should do in reversed order or keep track of what was inserted
    num_inserted = 0
    iterator = []
    for j in range(len(boxes)):  # layout boxes
        iterator.append((starts[j], labels[j], "start", j))
        iterator.append((ends[j], labels[j], "end", j))
    iterator = sorted(iterator, key=lambda x: x[0])

    line_ranges = [range(*word_range) for word_range in OCR["structures"]["lines"]["structure_value"]]
    # print(line_ranges)
    # print(iterator)
    L = len(original_tokens)
    for insertion_index, label, type_index, j in iterator:  # sort fromn first insertion point to last (based on start)
        # print("Inserting tokens for", label, insertion_index, type_index, j)
        if type_index == "start":
            dynamic_index = insertion_index + num_inserted
            OCR["tokens"].insert(dynamic_index, f"<{label}>")
            OCR["positions"].insert(
                dynamic_index, [coord - (0.01 * coord) for coord in OCR["positions"][dynamic_index]]
            )  # just before token after
        else:
            dynamic_index = min(
                L + num_inserted, insertion_index + num_inserted + 1
            )  # can be an issue if there are no more tokens after bbox
            OCR["tokens"].insert(dynamic_index, f"</{label}>")
            dynamic_index -= 1  # FIX for updating line ranges [token touched needs to be used for finding line range]
            OCR["positions"].insert(
                dynamic_index, [coord + (0.01 * coord) for coord in OCR["positions"][dynamic_index]]
            )  # just after token before

        # update line ranges
        # find index l within line ranges, then increment end with 1
        # followed by incrementing all l+1 line ranges with 1
        for l, line_range in enumerate(line_ranges):
            if dynamic_index in line_range:
                line_ranges[l] = range(line_range[0], line_range[-1] + 1 + 1)  # for non inclusive index ranges
                for k in range(l + 1, len(line_ranges)):  # not creating new lines, just extending the existing ones
                    line_ranges[k] = range(
                        line_ranges[k][0] + 1, line_ranges[k][-1] + 1 + 1
                    )  # for non inclusive index ranges
                break
        num_inserted += 1

    token_inserted_indices = [
        insertion_index + j if type_index == "start" else insertion_index + j + 1
        for j, (insertion_index, label, type_index, _) in enumerate(iterator)
    ]
    OCR["structures"]["lines"]["structure_value"] = [[line_range[0], line_range[-1] + 1] for line_range in line_ranges]
    # print(line_ranges)
    # print(iterator, token_inserted_indices)
    return token_inserted_indices


def plot_boxes(image, boxes, labels, token_boxes=None, title=""):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # plot boxes with different colors per label on the image
    for box, label in zip(boxes, labels):
        # Create a Rectangle patch defined via an anchor point xy and its width and height.
        ax.add_patch(
            patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="r", facecolor="none"
            )
        )
        ax.text(box[0], box[1], label, fontsize=12, color="white")

    if token_boxes:
        for box in token_boxes:
            ax.add_patch(
                patches.Rectangle(
                    (box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor="b", facecolor="none"
                )
            )
    plt.title(title)
    plt.show()


def rescale_DLA_bboxes(boxes, img_shape, reshaped_img_shape):
    # Get the scaling factor
    # img_shape = (y, x)
    # reshaped_img_shape = (y1, x1)
    # the scaling factor = (y1/y, x1/x)

    scale = np.flipud(
        np.divide(reshaped_img_shape, img_shape)
    )  # you have to flip because the image.shape is (y,x) but your corner points are (x,y)
    new_boxes = []
    for box in boxes:
        top_left_corner = [box[0], box[1]]
        bottom_right_corner = [box[2], box[3]]
        # use this on to get new top left corner and bottom right corner coordinates
        new_top_left_corner = np.multiply(top_left_corner, scale)
        new_bottom_right_corner = np.multiply(bottom_right_corner, scale)
        new_box = [
            new_top_left_corner[0],
            new_top_left_corner[1],
            new_bottom_right_corner[0],
            new_bottom_right_corner[1],
        ]
        new_boxes.append(new_box)
    return new_boxes


def load_OCR_due(ocr):
    # doc_path =  ocr/"documents.json"
    content_path = os.path.join(ocr, "documents_content.jsonl")

    document_contents = {}
    with open(content_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            document_contents[data["name"]] = data["contents"][1]
    return document_contents


def load_DLA_torch(dla):
    DLA = torch.load(dla)
    DLA_dict = {}
    for value in DLA:
        try:
            identifier = value.filename[0].split("/")[-1].replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
        except Exception as e:
            # print(e)
            # print(value.filename)
            continue

        # FIX: need to convert boxes from [x,y,w,h] in COCO format to [x1,y1,x2,y2] in DUE format
        boxes = value.pred_boxes.tensor.tolist()
        # for box in boxes:
        #     box[2] = box[0] + box[2]
        #     box[3] = box[1] + box[3]
        DLA_dict[identifier] = {"boxes": boxes, "labels": value.labels, "image_size": value.image_size}
    return DLA_dict


def test_rescaling():
    original_size = (1764, 2257)
    original_image = Image.open("pybv0228_81.png")
    print(original_image.size)  # w x h

    # without the coco garbage
    original_boxes = [
        [273.88812255859375, 357.46728515625, 1974.80126953125, 1517.7734375],
        [1079.2783203125, 228.21287536621094, 1297.0621337890625, 257.201904296875],
    ]

    # plot original boxes first
    plot_boxes(original_image, original_boxes, ["Picture", "Text"], title="original")

    labels = ["Picture", "Text"]

    rescale_size = (2646, 3383)  # (y, x)
    new_boxes = rescale_DLA_bboxes(original_boxes, original_size, rescale_size)  # might be an issue here, passing (y,x)
    reshaped_image = original_image.resize(tuple(reversed(rescale_size)), Image.BICUBIC)  # (x, y) assumed

    print(reshaped_image.size)  # w x h

    OCR = load_OCR_due("./DocVQA/aws_neurips_time/docvqa/dev")["pybv0228_81"]
    dump_json(OCR, "pybv0228_81_due.json")
    print(new_boxes)
    print(OCR["common_format"]["positions"])

    plot_boxes(reshaped_image, new_boxes, labels, token_boxes=OCR["common_format"]["positions"], title="DUE")

    print("Before", "\n".join(space_layout(OCR["common_format"]["tokens"], OCR["common_format"]["positions"])))

    results = place_tokens({"boxes": new_boxes, "labels": labels}, OCR)

    tokens = OCR["common_format"]["tokens"]
    lines = OCR["common_format"]["structures"]["lines"]["structure_value"]
    line_texts = []
    for words_range in lines:
        line_words = [tokens[idx] for idx in range(words_range[0], words_range[1])]
        line_texts.append(" ".join(line_words))
    print("\n".join(line_texts))
    print("After", "\n".join(space_layout(OCR["common_format"]["tokens"], OCR["common_format"]["positions"])))

    plot_boxes(
        reshaped_image, new_boxes, labels, token_boxes=OCR["common_format"]["positions"], title="post transformation"
    )


def test_line_text(OCR, key):
    tokens = OCR[key]["common_format"]["tokens"]
    lines = OCR[key]["common_format"]["structures"]["lines"]["structure_value"]
    line_boxes = OCR[key]["common_format"]["structures"]["lines"]["positions"]

    # assert it covers the range
    assert len(tokens) == sum(lines, [])[-1]  # safe?

    line_texts = []
    for words_range in lines:
        line_words = [tokens[idx] for idx in range(words_range[0], words_range[1])]
        line_texts.append(line_words)

    print("\n".join([" ".join(line_words) for line_words in line_texts]))
    line_texts = [" ".join(line_words) for line_words in line_texts]
    print()
    # print("After", "\n".join(space_layout(line_texts, line_boxes)))


def main(args):
    # torch load instances coco and OCR from filenames - save OCR with new tokens for integration into LATIN-prompt

    OCR = load_OCR_due(args.ocr)
    DLA = load_DLA_torch(args.dla)

    # check if all from OCR are in DLA
    for key in tqdm(OCR):
        if key not in DLA:
            print(
                f"Missing: {key}"
            )  # just use without DLA extension then; negatively biases the results; prioritize later if needed
            DLA[key] = {"boxes": [], "labels": []}
        else:
            OCR_image_size = tuple(
                reversed(OCR[key]["common_format"]["structures"]["pages"]["positions"][0][2:])
            )  # width, height -> reverse
            DLA_image_size = DLA[key]["image_size"]  # height, width
            if OCR_image_size != DLA_image_size:
                # print(f"OCR: {OCR_image_size} DLA: {DLA_image_size} in h x w format")
                # print("old boxes", DLA[key]["boxes"], DLA[key]["labels"])
                new_boxes = rescale_DLA_bboxes(DLA[key]["boxes"], DLA_image_size, OCR_image_size)
                DLA[key]["boxes"] = new_boxes
                # print("new boxes", new_boxes)

            # inplace updates?
            results = place_tokens(DLA[key], OCR[key], args)
            print(f"Results: {results} inserted at {OCR[key]['common_format']['tokens']}")
            # test_line_text(OCR, key)#=> beware, running the test will change the OCR[key] inplace
            """
            if key == "pybv0228_81":
                tokens = OCR[key]["common_format"]["tokens"]
                lines = OCR[key]["common_format"]["structures"]["lines"]["structure_value"]
                line_texts = []
                for words_range in lines:
                    line_words = [tokens[idx] for idx in range(words_range[0], words_range[1])]
                    line_texts.append(" ".join(line_words))
                print("\n".join(line_texts))
                print(
                    "After",
                    "\n".join(
                        space_layout(OCR[key]["common_format"]["tokens"], OCR[key]["common_format"]["positions"])
                    ),
                )
            """

    # save OCR with new tokens and boxes and lines
    identifier = args.origin if args.origin else args.dla.replace(".pth", "").split("/")[-1]
    dataset = args.ocr.split("/")[-2]
    identifier = f"{dataset}_{identifier}"

    # extend identifier with options
    if args.ignore_tokens:
        identifier += f"_ignore_tokens-{'_'.join(args.ignore_tokens)}"
    if args.markup != "xml":
        identifier += f"_markup-{args.markup}"
    if args.ignore_nesting:
        identifier += f"_ignore_nesting"
    if args.DLA_isolated_line:
        identifier += f"_DLA_isolated_line"

    out_path = os.path.join(args.ocr, f"documents_content-{identifier}.jsonl")
    with open(out_path, "w") as outfile:
        for key in OCR:
            entry = {}
            entry["name"] = key
            entry["contents"] = [None, OCR[key]]  # index of Azure is 1
            json.dump(entry, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    # usage: py DLA_to_BBOX.py --ocr ./DocVQA/aws_neurips_time/docvqa/dev --dla docvqa_vitb_imagenet_doclaynet_tecaher_instances_predictions.pth --origin ViT_teacher

    """Extensions

    1. use only a subset of DLA tokens into OCR (skipping text)
    1.1 skip agrammatical starts and ends (when end is before start or no end is found)
    put DLA tokens on separate lines to not confuse the model (requires mangling the line boxes, but will check; might upset placement; which might already be happening due to more tokens on a line than before)
    2.1 ignore layout tokens in calculating space layout
    make the prompt aware of DLA tokens in the text
    check for nesting and update labels depending on nesting (table_1) ...
    4.1 make labels order aware by always adding _\d
    5. low-confidence filtering at 0.5 might be too strict?
    6 Llama2 might be confused about the XML tags, swap to [TABLE] or just TABLE and table to indicate start-stop

    """

    from argparse import ArgumentParser

    parser = ArgumentParser()
    # load OCR and DLA
    parser.add_argument("--ocr", type=str, help="path to OCR json")
    parser.add_argument("--dla", type=str, help="path to DLA predictions")
    parser.add_argument("--origin", type=str, default="", help="some DLA model identifier")

    parser.add_argument("--ignore_tokens", action="append", default=[], help="tokens to ignore")  # 1
    parser.add_argument(
        "--markup",
        choices=["xml", "llama", "upper"],
        default="xml",
        help="how to indicate tokens in the text like <> or [] or uppercase",
    )
    parser.add_argument("--ignore_nesting", action="store_true", help="ignore nesting of tokens")
    parser.add_argument(
        "--DLA_isolated_line", action="store_true", default=False, help="put DLA tokens on separate lines"
    )

    args = parser.parse_args()

    # test_rescaling()
    # test_placement()
    main(args)
