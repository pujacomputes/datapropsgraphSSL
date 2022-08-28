multiplier=4.0
background_graph=tree
PREFIX="../data"
POSTFIX="data.pkl"
python SpecCL.py --dataset_list ${PREFIX}/${background_graph}_a1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_b1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_c1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_d1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_e1_${multiplier}/${POSTFIX} ${PREFIX}/${background_graph}_f1_${multiplier}/${POSTFIX} \
    --dataset_name A-B-C-D-E-F \
    --dataset_root /usr/workspace/trivedi1/Fall2022/datapropsgraphSSL/data \
    --epochs 1 \
    --multiplier $multiplier 