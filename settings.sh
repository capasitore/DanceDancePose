cd ..
pip3 install tensorflow opencv-python
git clone https://www.github.com/ildoonet/tf-openpose
cd tf-openpose
pip3 install -r requirements.txt
cd tf_pose/pafprocess
swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace

cd ../../
cd DanceDancePose
python3 ddp.py