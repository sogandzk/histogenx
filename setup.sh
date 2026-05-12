pip install gdown
pip install datasets
pip install pytorch-lightning
pip install optuna
pip install optuna optuna-integration[pytorch_lightning]

mkdir data
cd data

gdown https://drive.google.com/file/d/1kAaQerv-dy7U8RIYayHFyXITdvoG29iV/view?usp=sharing

tar -xzf dataset.tar.gz
rm dataset.tar.gz
