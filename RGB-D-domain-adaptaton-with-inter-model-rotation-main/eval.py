import torch
from torch.utils.tensorboard import SummaryWriter
from dataloader import MyDataset, MyTransformer
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import EvaluationManager, get_num_correct, load_checkpoint
from net import Obj_classifier, Resnet_extractor, Obj_classifier, Rot_classifier
import os
import torch.optim as optim

device = torch.device('cuda')

# Dataset ROD
rod_evaluation = MyDataset("./ROD-synROD/ROD/wrgbd_40k-split_sync.txt", crop = "center", fold_name = "ROD")

# Dataloader ROD
Rod_evaluation_loader = DataLoader( 
            rod_evaluation,
            shuffle=True,                    # rod for evaluation of the whole model
            batch_size=64,
            num_workers=1,
            
            )

device = torch.device('cuda')
rgb_extrector = Resnet_extractor().to(device)
depth_extrector = Resnet_extractor().to(device)
obj_classifier = Obj_classifier().to(device)
rot_classifier = Rot_classifier().to(device)
net_list = [rgb_extrector, depth_extrector, obj_classifier, rot_classifier]


rgb_extrector_opt = optim.SGD(rgb_extrector.parameters(), lr=0.9, momentum=0.9, weight_decay=0.5)
depth_extrector_opt = optim.SGD(rgb_extrector.parameters(), lr=0.9, momentum=0.9, weight_decay=0.5)
obj_classifier_opt = optim.SGD(rgb_extrector.parameters(), lr=0.9, momentum=0.9, weight_decay=0.5) 
rot_classifier_opt = optim.SGD(rgb_extrector.parameters(), lr=0.9, momentum=0.9, weight_decay=0.5)
opt_lis = [rgb_extrector_opt, depth_extrector_opt, obj_classifier_opt, rot_classifier_opt]

first=0

#Load the checkpoint of the model you want to test (set accordingly name of the folder)
checkpoint_path = os.path.join("./checkpoint", "checkpoint.pth")
first_epoch = load_checkpoint(checkpoint_path, first, net_list, opt_lis)
rgb_extractor = net_list[0]
depth_extractor = net_list[1]
obj_classifier = net_list[2]


with EvaluationManager(net_list):
    Rod_evaluation_loader_iter = iter(Rod_evaluation_loader) 
    correct = 0.0
    num_predictions = 0.0
    val_loss = 0.0
    # for i in range(0, 1000, 64):
    for tmp in Rod_evaluation_loader_iter:
      #tmp = Rod_evaluation_loader_iter.get_next()
      rgb_img = tmp[0].to(device)
      depth_img = tmp[1].to(device)
      label = tmp[2].to(device)

      rgb_out, _ = rgb_extrector(rgb_img)
      depth_out, _ = depth_extrector(depth_img)
      out_concat = torch.cat((rgb_out, depth_out), 1)
      pred_labels = obj_classifier(out_concat) 
      pred_labels =   pred_labels.type(torch.float32)

      correct +=get_num_correct( pred_labels, label) 
      num_predictions += rgb_img.shape[0]
      del rgb_img, depth_img, label, pred_labels,rgb_out, depth_out, out_concat

    accuracy = correct/ num_predictions
    # actual_loss = val_loss/ num_predictions   
    print(" -- validation of source-- accuracy: {}".format( accuracy))