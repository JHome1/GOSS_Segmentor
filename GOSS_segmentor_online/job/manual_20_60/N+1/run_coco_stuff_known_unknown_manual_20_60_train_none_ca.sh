python -m torch.distributed.launch --nproc_per_node=2 ./tools/train_net.py --cfg configs/wholistic_segmentor_coco_stuff_known_unknown_manual_20_60.yaml OUTPUT_DIR "./output/wholistic_segmentor_coco_stuff_known_unknown_manual_20_60_N+1_none_ca"
