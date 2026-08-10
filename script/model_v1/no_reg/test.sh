CUDA_VISIBLE_DEVICES=1 python main.py --in_channel 13 --adding_type 0 --use_layer_norm --dropout 0.3 --batch_size 1 --debug

## baseline
CUDA_VISIBLE_DEVICES=0 python test_only.py --in_channel 13 --adding_type 0 --dropout 0.1 --batch_size 16 --model_type cnn-lstm --group_name baseline