class args():
    traindata_dir = "/home/dzgc8/experiment/2LAB5/Elements SE/Dataset/train32/"
    testdata_dir = "/home/dzgc8/experiment/2LAB5/Elements SE/Dataset/test_QK/"
    evaldata_dir = "/home/dzgc8/experiment/2LAB5/Elements SE/Dataset/test_QK/"
    # oritestdata_dir = '/media/hdr/Elements SE/Dataset/originTestGF2128'
    sample_dir = "/home/dzgc8/experiment/2LAB5/sample/Aspa_spe/sam_pmamba/"
    checkpoint_dir = "/home/dzgc8/experiment/2LAB5/checkpoint/" # save_model
    checkpoint_backup_dir = "/home/dzgc8/experiment/2LAB5/checkpoint/backup/" # 每N次保存一下模型，保存的位置
    record_dir = "/home/dzgc8/experiment/2LAB5/log/record/"
    log_dir = "/home/dzgc8/experiment/2LAB5/log/"
    model2_path ="/home/dzgc8/experiment/2LAB5/checkpoint/edge_enhance_multi.pth"
    output_dir = "/home/dzgc8/experiment/2LAB5/output/lab1_pmamba/"
    edge_enhance_multi_pretrain_model ="/home/dzgc8/experiment/2LAB5/checkpoint/edge_enhance_multi.pth"

    max_value = 65535  # QB是2047,1023
    epochs = 700
    lr = 0.0005  # learning rate
    batch_size = 16
    lr_decay_freq = 200
    model_backup_freq = 10
    eval_freq = 5

    data_augmentation = False

    cuda = 1    # use GPU 1
