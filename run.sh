# check if Python virtual environment env exists
if [ ! -f "env/bin/activate" ]; then
    echo "Environment env does not exist. Please install it via ./install.sh"
    exit 1
fi

# activate Python virtual environment
source env/bin/activate

echo "##### RUNNNING MODEL TRAINING #####"
python script/booster.py --data data/Data.csv


deactivate