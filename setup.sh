pip install gdown
pip install datasets
pip install pytorch-lightning
pip install optuna
pip install optuna optuna-integration[pytorch_lightning]

mkdir data
cd data

gdown --fuzzy https://drive.google.com/file/d/1QMjQzndn7ym5zcuaHzKhJcrdMOxvMhgL/view?usp=sharing

tar -xzf dataset.tar.gz
rm dataset.tar.gz
