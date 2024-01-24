import json
import os
from pathlib import Path

import datasets

from PIL import Image

logger = datasets.logging.get_logger(__name__)


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def quad_to_box(quad):
    # test 87 is wrongly annotated
    box = (max(0, quad["x1"]), max(0, quad["y1"]), quad["x3"], quad["y3"])
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box


def box4point_to_box2point(box4point):
    # bounding box = [x0, y0, x1, y1, x2, y2, x3, y3]
    all_x = [box4point[2 * i] for i in range(4)]
    all_y = [box4point[2 * i + 1] for i in range(4)]
    # print(box4point)
    box2point = [min(all_x), min(all_y), max(all_x), max(all_y)]
    # print(box2point)
    return box2point


_DESCRIPTION = """
InfographicsVQA dataset
"""


class InfographicsVQAConfig(datasets.BuilderConfig):
    """BuilderConfig for InfographicsVQA"""

    def __init__(self, dla_model: str = "", **kwargs):
        """BuilderConfig for CORD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(InfographicsVQAConfig, self).__init__(**kwargs)
        self.dla_model = f"-{dla_model}" if dla_model else ""


class InfographicsVQA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        InfographicsVQAConfig(name="InfographicsVQA", version=datasets.Version("1.0.0"), description=_DESCRIPTION),
    ]

    def __init__(self, dla_model: str = "", **kwargs):
        """BuilderConfig for CORD.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        self.dla_model = f"-{dla_model}" if dla_model else ""
        super(InfographicsVQA, self).__init__(**kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION + f" with dla_model {self.dla_model}",
            features=datasets.Features(
                {
                    "questionId": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence(datasets.Value("string")),
                    # "words": datasets.Sequence(datasets.Value("string")),
                    # "word_boxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "texts": datasets.Sequence(datasets.Value("string")),
                    "text_boxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "question_types": datasets.Sequence(datasets.Value("string")),
                    # "image": datasets.features.Image(),
                }
            ),
            supervised_keys=None,
            # citation=_CITATION,
            homepage="xxx",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Uses local files located with data_dir"""
        dest = Path(os.path.join(os.environ["DATAS_DIR"], "aws_neurips_time/infographics_vqa"))
        # dest = Path("./datas/InfographicsVQA")
        return [
            # datasets.SplitGenerator(
            #     name=datasets.Split.TRAIN,
            # gen_kwargs={
            # "doc_path": dest / "documents.json",
            # "content_path": dest / f"train/documents_content{self.dla_model}.jsonl",
            # "split": "train",
            #     }
            # ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    # "doc_path": dest/"dev/document.jsonl",
                    "doc_path": dest / "documents.json",
                    "content_path": dest / f"dev/documents_content{self.dla_model}.jsonl",
                    "split": "dev",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    # "doc_path": dest/"test/document.jsonl",
                    "doc_path": dest / "documents.json",
                    "content_path": dest / f"test/documents_content.jsonl",
                    "split": "test",
                },
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, doc_path, content_path, split):
        logger.info(f"⏳ Generating examples from = {doc_path} and {content_path}")

        document_contents = {}
        with open(content_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                document_contents[data["name"]] = data["contents"][1]  # microsoft_cv
                # print(data["name"])
                # print(data["contents"][1]["common_format"])

        # while True:
        #     continue

        # NEW: open annotations for question types
        def join_annotations(a):
            question_types = []
            keys = ["answer_type", "evidence", "operation/reasoning"]
            for key in keys:
                if a[key]:
                    question_types.extend(a[key])
            return question_types

        with open(os.path.join(os.environ["DATAS_DIR"], "infographicsVQA_val_v1.0_withQT.json"), "r") as f:
            valdata = json.load(f)["data"]  # transform into dict with question_id as key
            val_question_types = {valdata[i]["questionId"]: join_annotations(valdata[i]) for i in range(len(valdata))}

        count = 0
        with open(doc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for doc in data:
                if doc["split"] != split:
                    continue

                for item in doc["client_features"]:
                    question = item["name"]
                    question_id = int(item["metadata"]["question_id"])
                    answers = item["value_variants"] if doc["split"] != "test" else None
                    question_types = val_question_types[question_id] if doc["split"] == "dev" else None
                    words = document_contents[doc["name"]]["common_format"]["tokens"]
                    # word_boxes = document_contents[data["name"]]["common_format"]["positions"]
                    lines = document_contents[doc["name"]]["common_format"]["structures"]["lines"]["structure_value"]
                    line_boxes = document_contents[doc["name"]]["common_format"]["structures"]["lines"]["positions"]

                    try:
                        line_texts = []
                        for words_range in lines:
                            line_words = [words[idx] for idx in range(words_range[0], words_range[1])]
                            line_texts.append(" ".join(line_words))
                    except Exception as e:
                        print(e)
                        print(doc["name"])
                        print(lines)
                        print(words)
                        print("=" * 50)
                        continue

                    example = {
                        "questionId": question_id,
                        "question": question,
                        "answers": answers,
                        "texts": line_texts,
                        "text_boxes": line_boxes,
                        "question_types": question_types,
                    }

                    # if count > 10:
                    #     break

                    # print("="*50)
                    # for k, v in example.items():
                    #     print(k, v)
                    # print(example["questionId"])
                    # print(example["texts"])
                    # print(document_contents[data["name"]]["common_format"].keys())
                    # print(document_contents[data["name"]]["common_format"]["structures"]["lines"])

                    yield count, example

                    count += 1

                # if count > 10:
                #     break


if __name__ == "__main__":
    dataset = datasets.load_dataset(
        os.path.abspath(__file__),
        # dla_model="vitb_imagenet_doclaynet_tecaher",
        download_mode="force_redownload",
        ignore_verifications=True,
    )

    print("extended", len(dataset["validation"]))
    print(dataset["validation"][0])
    # load_dataset_builder, config_kwargs
    dataset = datasets.load_dataset(
        os.path.abspath(__file__),
    )
    print("original", len(dataset["validation"]))
    print(dataset["validation"][0])
    # print(len(dataset["test"]))
