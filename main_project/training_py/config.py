# import argparse

# def makeargment():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--n_epoch', type=int, required=False, default=10)
#     parser.add_argument('--batch_size', type=int, required=False, default=256)
#     parser.add_argument('--lr', type=float, required=False, default=1.5e-5)
#     parser.add_argument('--warmup', type=float, required=False, default=0.2)
#     parser.add_argument('--weight_decay', type=float, required=False, default=0.001)
#     # not change
#     parser.add_argument('--model_name', type=str, required=False, default='xlm-roberta-base')

#     return parser.parse_args()

# config = makeargment()

config = {
    'model_name': 'xlm-roberta-base',
    # 'n_labels': len(labels),
    'batch_size': 256,
    'lr': 1.5e-5,
    'warmup': 0.2, 
    # 'train_size': len(ucc_data_module.train_dataloader()),
    'weight_decay': 0.001,
    'n_epochs': 50
}