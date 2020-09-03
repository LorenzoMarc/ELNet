DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="MRNet-${DATE}-MRNet"
DATA_PATH='/content/MRNet-v1.0/'

EPOCHS=3
PREFIX=MRNet

python3 /content/content/mrnet-baseline-master/train.py -t acl -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS
python3 /content/content/mrnet-baseline-master/train.py -t acl -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS 
python3 /content/content/mrnet-baseline-master/train.py -t acl -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS 

#python3 /content/content/mrnet-baseline-master/train.py -t meniscus -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS 
#python3 /content/content/mrnet-baseline-master/train.py -t meniscus -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS
#python3 /content/content/mrnet-baseline-master/train.py -t meniscus -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS

#python3 /content/content/MRNet-v1.0/train.py -t abnormal -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS
#python3 /content/content/MRNet-v1.0/train.py -t abnormal -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS
#python3 /content/content/MRNet-v1.0/train.py -t abnormal -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS

python3 /content/content/mrnet-baseline-master/train_logistic_regression.py --path-to-model "experiments/${EXPERIMENT}/models/" --data-path $DATA_PATH 
