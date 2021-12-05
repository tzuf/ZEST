-install:
conda create -n ZEST python=3.7.4 anaconda
conda activate ZEST 
pip install -r requirements.txt 

-data-
follow instructions under data\README.md 


-RUN
python train_CUB.py --split easy --model similarity 
python train_NAB.py --split easy --model similarity 