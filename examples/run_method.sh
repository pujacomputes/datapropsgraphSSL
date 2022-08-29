aug_ratio=0.6
background_graph=tree
method=graphcl
multiplier=4.0
POSTFIX="data.pkl"
PREFIX="../data"
seed=237

for aug_type in gga_sym caa_sym
do
    python SpecCL.py --dataset_list ${PREFIX}/${background_graph}_a1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_b1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_c1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_d1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_e1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_f1_${multiplier}/${POSTFIX} \
        --dataset_name A-B-C-D-E-F-${multiplier} \
        --aug_type ${aug_type} \
        --projector \
        --dataset_root "../data" \
        --seed $seed \
        --multiplier $multiplier 
done 