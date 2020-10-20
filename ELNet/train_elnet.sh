DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="ELNet-${DATE}-ELNet"
DATA_PATH='/content/MRNet-v1.0/'
NORM='contrast'
LR=1e-5
EPOCHS=200
SEED=2 # --seed $SEED
SCHEDULER='plateau' #--lr_scheduler $SCHEDULER
PREFIX=ELNet

python3 /content/ELNet/train.py -t acl -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --lr=$LR --lr_scheduler=$SCHEDULER

python3 /content/ELNet/train.py -t meniscus -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type=$NORM --lr=$LR --lr_scheduler=$SCHEDULER

python3 /content/ELNet/train.py -t abnormal -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --lr=$LR --lr_scheduler=$SCHEDULER

