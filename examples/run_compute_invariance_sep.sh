aug_ratio=0.6
aug_type=gga_sym
background_graph=tree
method=graphcl
multiplier=4.0
POSTFIX="data.pkl"
PREFIX="../data"
seed=237
ckpt="../logs/A-B-C-D-E-F-${multiplier}/${method}_${multiplier}_${aug_type}_${aug_ratio}_False_${seed}_59.ckpt" \

python compute_invariance_sep.py --dataset_list ${PREFIX}/${background_graph}_a1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_b1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_c1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_d1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_e1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_f1_${multiplier}/${POSTFIX} \
    --aug_ratio $aug_ratio \
    --aug_type $aug_type \
    --ckpt $ckpt \
    --dataset_name A-B-C-D-E-F-${multiplier} \
    --dataset_root "../data" \
    --method $method \
    --multiplier $multiplier \
    --projector \
    --save_path "../logs"