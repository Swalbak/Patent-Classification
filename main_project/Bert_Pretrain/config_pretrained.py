config = {
    'model_name': '/home/aailab/yjh/model/save_model_5epoch',
    # 'n_labels': len(labels),
    'batch_size': 256,
    'lr': 1.5e-5,
    'warmup': 0.2, 
    # 'train_size': len(ucc_data_module.train_dataloader()),
    'weight_decay': 0.001,
    'n_epochs': 50
}