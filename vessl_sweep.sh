#!/bin/bash

vessl sweep create \
    --objective-type "maximize" \
    --objective-goal "1.0" \
    --objective-metric "val/accuracy" \
    --num-experiments 16 \
    --num-parallel 2 \
    --algorithm grid \
    --parameter "batch_size int list 4 8 16 32" \ # Search space parameters in the form of [name] [type] [range_type] [values...]
    --parameter "optimizer categorical list SGD Adam" \
    --parameter "weight_decay double space 1e-4 1e-3 5e-4" \
    --parameter "learning_rate double space 1e-4 1e-3 2e-4" \
    --command "python -u -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --master_port 9901 train_classifier.py --distributed --config configs/b7.json --freeze-epochs 0 --test_every 1 --opt-level O1 --label-smoothing 0.01 --folds-csv folds.csv --fold 0 --seed 111 --data-dir $ROOT_DIR --resume weights/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36 --prefix b7_111_"



