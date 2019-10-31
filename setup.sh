


rm -rf .python_dir
mkdir .python_dir
cd .python_dir
ln -s ../python rpnet
ln -s ../external/keras-lr-multiplier/keras_lr_multiplier 


export PYTHONPATH=$PWD:$PYTHONPATH
cd ..
