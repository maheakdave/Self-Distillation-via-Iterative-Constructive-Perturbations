class Config():
    #DATA
    batch_size = 2
    data_path = "D:\\sdist\\data\\ADEChallengeData2016"
    img_size = 224
    num_workers = 0
    output = ''
    ignore_index = 255
    num_classes = 151
    out_dir = 'Result_seg_decreasing_weights'

    # Factors
    num_masks = 4
    mcz_factor = 0.25
    prob_z = 0.2
    scale = 1.15

    #TRAIN
    START_EPOCH = 0
    SAVE_FREQ = 1
    epochs = 300
    LR = 3e-4
    betas = (0.9, 0.999)
    weight_decay = 1e-2
    grad_clip = 9.
    lr_gamma = 0.2
    backbone = 'resnet50'
    epochs = 300

    #AUG
    hflip_prob = 0.5
    brightness = 0.5
    contrast = 0.5
    saturation = 0.5
    hue = 0.5
    degrees = [-30.0, 30.0]
    translate = (0.1, 0.1)
    scale = (0.75, 1.75)
    re_prob = 0.15
    sharpness_prob = 0.2
    
    #MISC
    seed = 42
