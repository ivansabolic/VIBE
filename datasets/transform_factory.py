from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, RandomResizedCrop, Resize, CenterCrop

def get_transforms(dataset_name:str, train: bool, backbone_type: str) -> Compose:
    if dataset_name == "cifar-10":
        if train:
            T = Compose([
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])
        else:
            T = Compose([
                ToTensor(),
                Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])
    elif dataset_name == "cifar-100":
        if train:
            T = Compose([
                RandomCrop(32, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        else:
            T = Compose([
                ToTensor(),
                Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
    elif "imagenet" in dataset_name:
        if train:
            T = Compose([
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            T = Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
    else:
        raise ValueError(f"Unknown transforms for dataset {dataset_name}.")

    if backbone_type in ["dino", "clip"] and "cifar" in dataset_name:
        T_list = T.transforms
        T_list.append(Resize((224, 224)))
        T = Compose(T_list)

    return T

def fetch_transforms_for_dataset(dataset_name: str, backbone_type: str) -> Compose:
    return get_transforms(dataset_name, True, backbone_type), get_transforms(dataset_name, False, backbone_type)
