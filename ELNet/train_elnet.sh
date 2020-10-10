DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="ELNet-${DATE}-ELNet"
DATA_PATH='/content/MRNet-v1.0/'
NORM='contrast'
LR=1e-5
EPOCHS=1
SEED=2 # --seed $SEED
SCHEDULER='plateau' #--lr_scheduler $SCHEDULER
PREFIX=ELNet

#python3 train.py -t acl -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR
#python3 train.py -t acl -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS  --set_norm_type $NORM --lr=$LR 
python3 train.py -t acl -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --lr=$LR 

#python3 train.py -t meniscus -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR 
python3 train.py -t meniscus -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR 
#python3 train.py -t meniscus -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR 

#python3 train.py -t abnormal -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR 
#python3 train.py -t abnormal -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR
python3 train.py -t abnormal -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --lr=$LR 

python3 train_logistic_regression.py --path-to-model "experiments/${EXPERIMENT}/models/" --data-path $DATA_PATH 
