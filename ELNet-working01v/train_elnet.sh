DATE=$(date +"%Y-%m-%d-%H-%M")
EXPERIMENT="ELNet-${DATE}-ELNet"
DATA_PATH='/content/MRNet-v1.0/'
NORM='layer'
LR=1e-5
EPOCHS=20
SEED = 1 # --seed $SEED
SCHEDULER = 'step' #--lr_scheduler $SCHEDULER
PREFIX=ELNet

python3 /content/ELNet-working01v/train.py -t acl -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR
python3 /content/ELNet-working01v/train.py -t acl -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS  --set_norm_type $NORM --lr=$LR 
python3 /content/ELNet-working01v/train.py -t acl -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR 

python3 /content/ELNet-working01v/train.py -t meniscus -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR 
python3 /content/ELNet-working01v/train.py -t meniscus -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR 
python3 /content/ELNet-working01v/train.py -t meniscus -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR 

python3 /content/ELNet-working01v/train.py -t abnormal -p sagittal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR 
python3 /content/ELNet-working01v/train.py -t abnormal -p coronal --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR
python3 /content/ELNet-working01v/train.py -t abnormal -p axial --experiment $EXPERIMENT --data-path $DATA_PATH --prefix_name $PREFIX --epochs=$EPOCHS --set_norm_type $NORM --lr=$LR 

python3 /content/ELNet-working01v/train_logistic_regression.py --path-to-model "/content/experiments/${EXPERIMENT}/models/" --data-path $DATA_PATH 
