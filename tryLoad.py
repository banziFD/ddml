from cifar.utils_model import CNN as Feature



pretrained = torch.load(path['workPath'] + '/pretraine')
feature = Feature()
feature.load_state_dict(pretrained)