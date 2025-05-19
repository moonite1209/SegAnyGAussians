unzip -q SegAnyGAussians.zip
cd SegAnyGAussians
conda create --name saga python=3.12 -c conda-forge
conda activate saga
pip install torch==2.4.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
mkdir weights
cd weights
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ..
cp third_party/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py weights/GroundingDINO_SwinT_OGC.py
cd third_party/GroundingDINO
export CUDA_HOME=/usr/local/cuda
pip install .
cd ../../..