for multiplier in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
    do
    for dataset in a1 b1 c1 d1 e1 f1
    do
        python create_dataset.py --dataset $dataset \
            --multiplier $multiplier
    done
done