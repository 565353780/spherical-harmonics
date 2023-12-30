cd ..
git clone https://github.com/565353780/data-convert.git

cd data-convert
./setup.sh

sudo apt install python3-dev libomp-dev

conda install -c conda-forge gcc=12.1.0

pip install scipy matplotlib
