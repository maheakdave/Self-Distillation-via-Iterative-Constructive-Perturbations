class Config():
    seed = 42
    batch_size = 2
    num_workers = 0
    size = 640,640
    TRAIN_PATH = ("COCO\\train2017","COCO\\annotations\\instances_train2017.json")
    VAL_PATH =  ("COCO\\val2017","COCO\\annotations\\instances_val2017.json")

    out_dir = 'Result_od_decreasing_weights'
    START_EPOCH = 0
    SAVE_FREQ = 1
    epochs = 300
    LR = 3e-4
    betas = (0.9, 0.999)
    weight_decay = 1e-2
    grad_clip = 9.
    lr_gamma = 0.2
