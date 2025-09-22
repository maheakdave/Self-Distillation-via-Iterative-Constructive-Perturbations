class Config:
    # DATASET
    num_classes = 100
    BATCH_SIZE = 32
    num_workers = 0
    ratio = 0.78
    
    #TRANSFORMS
    ORIGINAL_SIZE = 32,32

    #TRAINING
    SEED = 42
    lr = 3e-4
    weight_decay = 0
    epochs = 100
    SAVE_FREQ = 1
    PRINT_FREQ = 1
    START_EPOCH = 0
    out_dir = 'Result_ic_decreasing_weights'
    modify_model = True
