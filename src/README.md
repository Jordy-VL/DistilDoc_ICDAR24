# DistilDoc

This project consists of 3 major subprojects: CLF, DLA and Downstream.

## Installation

For CLF and Downstream, follow instructions in clf/README.md and additionally conda install the downstream/llm.yaml
For DLA, conda install the dla/dla.yaml. Beware of additional detectron2 based installation issues.

### other reqs

* ensure you have wandb setup: https://docs.wandb.ai/quickstart
* ensure you are logged into HuggingFace using the cli: https://huggingface.co/docs/huggingface_hub/main/quick-start#login 
* ensure you have git-lfs setup to save training models to the HuggingFace hub

## Reproducing experiments

Check each subproject's README.md, linked below
 
For CLF, check [here](clf/README.md)

For DLA, check [here](dla/README.md)

For Downstream, check [here](downstream/README.md)
