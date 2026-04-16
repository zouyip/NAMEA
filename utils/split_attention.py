import torch


from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad


def reshape_swin_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def reshape_vit_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def get_Attention_all(x, y, model, model_name):
    if model_name == 'deit':
        target_layers = [model[-1].blocks[-1].norm1]
    elif model_name == 'vit':
        target_layers = [model[-1].blocks[-1].norm1]
    elif model_name == 'res18':
        target_layers = [model[-1].layer4[-1]]
    elif model_name == 'dense121':
        target_layers = [model[-1].features[-1]]
    elif model_name == 'swin_t':
        target_layers = [model[-1].layers[-1].blocks[-1].norm1]
    elif model_name == 'bit_101':
        target_layers = [model[-1].stages[-1]]
    elif model_name == 'convit_t':
        target_layers = [model[-1].blocks[-1].norm1]
    else:
        target_layers = [model[-1].Mixed_7b]

    targets = [ClassifierOutputTarget(target_class) for target_class in y.tolist()]
    if model_name == 'deit':
        cam = GradCAM(model=model[-1], target_layers=target_layers, reshape_transform=reshape_vit_transform)
    elif 'vit' in model_name:
        cam = GradCAM(model=model[-1], target_layers=target_layers, reshape_transform=reshape_vit_transform)
    # elif 'win' in model_name:
    #     cam = GradCAM(model=model[-1], target_layers=target_layers, reshape_transform=reshape_swin_transform)
    else:
        cam = GradCAM(model=model[-1], target_layers=target_layers)
    # grayscale_cams = cam(input_tensor=x, targets=targets, aug_smooth=True)
    grayscale_cams = cam(input_tensor=x, aug_smooth=True)
    masks = []
    for i, grayscale_cam in enumerate(grayscale_cams):
        grayscale_cam = grayscale_cams[i, :]
        mask = torch.from_numpy(grayscale_cam).cuda()
        mask = mask.repeat(3, 1, 1)
        masks.append(mask)
    att = torch.stack(masks).cuda()
    return att



def get_att_mask(x, y, model, threshold, model_name):
    att = get_Attention_all(x, y, model, model_name)
    attention = att
    attention[attention > threshold] = 1.
    attention[attention <= threshold] = 0.
    return attention


