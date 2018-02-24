# RLCryptocurrency


### Environment setup on OCIO (CUDA driver already installed)

## Install anaconda and tensorflow
wget https://repo.continuum.io/archive/Anaconda2-5.1.0-Linux-x86_64.sh
bash Anaconda2-5.1.0-Linux-x86_64.sh

restart terminal

pip install tensorflow==1.4.1

pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp27-none-linux_x86_64.whl

pip install gym

## Start environment
source activate

export CUDA_VISIBLE_DEVICES=X  (remember to check the usage of GPU before starting with nvidia-smi)


