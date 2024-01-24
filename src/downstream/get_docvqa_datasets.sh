
## SP-DocVQA
wget --no-check-certificate 'https://datasets.cvc.uab.es/rrc/DocVQA/Task1/spdocvqa_qas.zip'
wget --no-check-certificate 'https://datasets.cvc.uab.es/rrc/DocVQA/Task1/spdocvqa_images.zip'
wget --no-check-certificate 'https://datasets.cvc.uab.es/rrc/DocVQA/Task1/spdocvqa_ocr.tar.gz'
#imdb optional
#wget --no-check-certificate 'https://datasets.cvc.uab.es/rrc/DocVQA/Task1/spdocvqa_imdb.zip'

#DUE format
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/DocVQA.tar.gz; tar -xvzf DocVQA.tar.gz
##PDFS  (optional)
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/pdfs/DocVQA.tar.gz; tar -xvzf DocVQA.tar.gz DocVQA/pdfs

## InfographicsVQA
wget --no-check-certificate 'https://datasets.cvc.uab.es/rrc/DocVQA/Task3/infographicsvqa_qas.zip'
wget --no-check-certificate 'https://datasets.cvc.uab.es/rrc/DocVQA/Task3/infographicsvqa_images.tar.gz'
wget --no-check-certificate 'https://datasets.cvc.uab.es/rrc/DocVQA/Task3/infographicsvqa_ocr.tar.gz'
#imdb optional
#wget --no-check-certificate 'https://datasets.cvc.uab.es/rrc/DocVQA/Task3/infographicsvqa_imdb.zip'

## DUE format
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/datasets/InfographicsVQA.tar.gz #; tar -xvzf InfographicsVQA.tar.gz
## PDFs (optional)
wget https://applica-public.s3.eu-west-1.amazonaws.com/due/pdfs/InfographicsVQA.tar.gz #; tar -xvzf InfographicsVQA.tar.gz InfographicsVQA/pdfs

## DUDE
python3 -c 'import datasets; ds=datasets.load_dataset("jordyvl/DUDE_loader", "Amazon_original")'
