# Downstream

The flow for the downstream experiments is as follows:

1. bash [get_docvqa_datasets.sh](get_docvqa_datasets.sh) to obtain DocVQA, Infographic and even DUDE
2. run [dla/inference_DLA](../dla/inference_DLA.py) on every DLA model you would like to test
3. bash [DLA enriched OCR](DLA_OCR_enrichment.sh)
4. bash [Downstream experiments with llama2](downstream_experiments_llama2.sh)