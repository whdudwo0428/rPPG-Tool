# setup2.sh

cd ./setup
pip install -r requirements.txt

cd ./causal-conv1d-main
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install .

cd ../../mamba_ssm
MAMBA_FORCE_BUILD=TRUE pip install .

cd ../..
