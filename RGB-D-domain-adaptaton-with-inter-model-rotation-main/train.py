from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as opt
from dataloader import MyDataset
from net import Obj_classifier, Resnet_extractor, Obj_classifier, Rot_classifier
from utils import RunBuilder, DataWraper,OptimizerManager, load_checkpoint, save_checkpoint, get_num_correct, entropy_loss_paper, EvaluationManager, RunRecord
import os






# Datasets

syn_Rod_train = MyDataset("./ROD-synROD/synROD/synARID_50k-split_sync_train1.txt", flip=True,fold_name = "synROD",crop="random",rotate=False)
syn_Rod_evaluation = MyDataset("./ROD-synROD/synROD/synARID_50k-split_sync_test1.txt",flip=False, fold_name = "synROD", rotate=False)

Rod_entropy_paper = MyDataset("./ROD-synROD/ROD/wrgbd_40k-split_sync.txt", flip=True, fold_name = "ROD", crop="random", rotate=False)

syn_Rod_train_Rot = MyDataset("./ROD-synROD/synROD/synARID_50k-split_sync_train1.txt",flip=True, rotate = True,crop="random", fold_name = "synROD")
syn_Rod_evaluation_Rot = MyDataset("./ROD-synROD/synROD/synARID_50k-split_sync_test1.txt",flip=False,  rotate = True, fold_name = "synROD")

rod_train_rot = MyDataset("./ROD-synROD/ROD/wrgbd_40k-split_sync.txt", flip=True, rotate = True,crop= "random", fold_name = "ROD")
rod_evaluation_rot = MyDataset("./ROD-synROD/ROD/wrgbd_40k-split_sync.txt",flip=False,  rotate = True, fold_name = "ROD")

tb = SummaryWriter('./log')





networks = []
entropy_loss = nn.CrossEntropyLoss()
recod = RunRecord()
resume = False # set it to True if you want to start from a checkpoint.pth to keep training the save model 


# defining parameters, each configuration of them will be run
parameters = dict(
    lr = [0.0003, 0.00003], 
    batch_size = [32, 64], 
    weight_decay = [0.5, 0.05, 0.04], 
    num_workers=[2],
    epoch = [40],
    weight_entropy = [0.1],
    weight_rot = [1, 0.8]
)



device = torch.device('cuda')
networks = []
entropy_loss = nn.CrossEntropyLoss()
recod = RunRecord()
for runs in RunBuilder.get_runs(parameters):
    
    
    checkpoint_path = os.path.join("./checkpoint", "checkpoint.pth")

    # DataLoader
    syn_rod_train_loader = DataLoader( #
                syn_Rod_train,
                shuffle=True,                    # syn_rod for training the Main head  (main cross entropy)
                batch_size=runs.batch_size,
                num_workers=runs.num_workers,
                
                )

    
    rod_entropy_paper_loader = DataLoader(  #
                Rod_entropy_paper,
                shuffle=True,                     # rod for evluating the Main head to compute entropy according to paper
                batch_size=runs.batch_size,
                num_workers=runs.num_workers
                )

    rod_train_rot_loader = DataLoader(#
                rod_train_rot,
                shuffle=True,                     # for pretext training  on rod  (pretext loss on rod)
                batch_size=runs.batch_size,
                num_workers=runs.num_workers
                )
    syn_rod_train_rot_loader = DataLoader( # 
                syn_Rod_train_Rot,
                shuffle=True,                     # for pretext loss calculation on syn_rod
                batch_size=runs.batch_size,
                num_workers=runs.num_workers
                )
    syn_Rod_evaluation_rot_loader = DataLoader(  #
                syn_Rod_evaluation_Rot,
                shuffle=True,                     # for evaluation of rotation with syn_rod
                batch_size=runs.batch_size,
                num_workers=runs.num_workers
                )
    syn_Rod_evaluation_loader = DataLoader(  #
                syn_Rod_evaluation,
                shuffle=True,                     # for main evaluation without rotation
                batch_size=runs.batch_size,
                num_workers=runs.num_workers
                )
    rod_evaluation_Rot_loader = DataLoader(  #
                rod_evaluation_rot,
                shuffle=True,                     # for pretext evaluation on rod
                batch_size=runs.batch_size,
                num_workers=runs.num_workers
                )
    
    # Network architecture
    rgb_extrector = Resnet_extractor().to(device)
    depth_extrector = Resnet_extractor().to(device)
    obj_classifier = Obj_classifier().to(device)
    rot_classifier = Rot_classifier().to(device)
    net_list = [rgb_extrector, depth_extrector, obj_classifier, rot_classifier]
    
    first_epoch = None

    rgb_extrector_opt = optim.SGD(rgb_extrector.parameters(), lr=runs.lr, momentum=0.9, weight_decay=runs.weight_decay)
    depth_extrector_opt = optim.SGD(depth_extrector.parameters(), lr=runs.lr, momentum=0.9, weight_decay=runs.weight_decay)
    obj_classifier_opt = optim.SGD(obj_classifier.parameters(), lr=runs.lr, momentum=0.9, weight_decay=runs.weight_decay) 
    rot_classifier_opt = optim.SGD(rot_classifier.parameters(), lr=runs.lr, momentum=0.9, weight_decay=runs.weight_decay)
    opt_lis = [rgb_extrector_opt, depth_extrector_opt, obj_classifier_opt, rot_classifier_opt]
    if not resume:
      first_epoch = 1
    else:
      first_epoch = load_checkpoint(checkpoint_path, first_epoch, net_list, opt_lis)
      rgb_extrector = net_list[0]
      depth_extrector = net_list[1]
      obj_classifier = net_list[2]
      rot_classifier = net_list[3]



    for epoch in range(first_epoch, runs.epoch +1):
        
        recod.start_epoch(epoch, runs)
        rod_entropy_paper_loader_iter = DataWraper(rod_entropy_paper_loader)  # for entropy in paper
        syn_rod_train_rot_loader_iter = DataWraper(syn_rod_train_rot_loader) # pretext loss on syn_rod
        syn_rod_train_loader_iter = DataWraper(syn_rod_train_loader) # pretext loss on syn_rod
        syn_Rod_evaluation_rot_loader_iter = DataWraper(syn_Rod_evaluation_rot_loader) # pretext loss on syn_rod
        rod_train_rot_loader_iter = DataWraper(rod_train_rot_loader) # pretext loss on syn_rod
        syn_Rod_evaluation_loader_iter = DataWraper(syn_Rod_evaluation_loader) # pretext loss on syn_rod
        rod_evaluation_Rot_loader_iter = DataWraper(rod_evaluation_Rot_loader)   # pretext loss on rod
        print(epoch)
        correct_syn_rod = 0
        correct_syn_rod_rot = 0
        correct_rod_rot = 0
        total_syn = 0
        total_rod = 0

        #for i in range(0, 500, 62):   # use this while debugging
        for tmp in syn_rod_train_loader:
          #tmp = syn_rod_train_loader_iter.get_next()
          rgb_img = tmp[0].to(device)
          # print(rgb_img.shape)
          depth_img = tmp[1].to(device)
          # print(depth_img.is_cuda)
          label = tmp[2]
          label = label.type(torch.int64)
          label_img = tmp[2].to(device)
          # print(label_img.is_cuda)
          total_syn += rgb_img.shape[0]

          with OptimizerManager(opt_lis):
                rgb_out, _ = rgb_extrector(rgb_img)
                depth_out, _ = depth_extrector(depth_img)
                out_concat = torch.cat((rgb_out, depth_out), 1)
                pred_labels = obj_classifier(out_concat) 
                pred_labels =  pred_labels.type(torch.float32) 
                class_loss = entropy_loss(pred_labels,label_img )
                correct_syn_rod +=get_num_correct( pred_labels, label_img)
                #print("num correct", correct_syn_rod)
                

                if runs.weight_entropy > 0.0:
                    tmp = rod_entropy_paper_loader_iter.get_next()
                    rgb_img= tmp[0].to(device)
                    depth_img= tmp[1].to(device)
                    rgb_out, _ = rgb_extrector(rgb_img)
                    depth_out, _ = depth_extrector(depth_img)
                    out_concat = torch.cat((rgb_out, depth_out), 1)
                    pred_labels = obj_classifier(out_concat)
                    pred_labels =   pred_labels.type(torch.float32)  
                    loss_paper = entropy_loss_paper(pred_labels)
                else:
                    loss_paper = 0

                loss = class_loss + runs.weight_entropy * loss_paper
                # print(class_loss)
                
                loss.backward()
                tb.add_scalar("Entropy Loss", loss, epoch)
                del rgb_img, depth_img, label_img, rgb_out, depth_out, out_concat, pred_labels

                if runs.weight_rot > 0.0:                    
                    tmp1 = syn_rod_train_rot_loader_iter.get_next()
                    rgb_img = tmp1[0].to(device)
                    depth_img  = tmp1[1].to(device)
                    label_rot = tmp1[3].to(device)
                    _, unpooled_rgb = rgb_extrector(rgb_img)
                    _, unpooled_depth = depth_extrector(depth_img)
                    pred_labels = rot_classifier(torch.cat((unpooled_rgb, unpooled_depth), 1))
                    pred_labels =   pred_labels.type(torch.float32)  
                    rot_loss = entropy_loss(pred_labels, label_rot)

                    loss = rot_loss * runs.weight_rot

                    # print(rot_loss)
                    correct_syn_rod_rot +=get_num_correct(pred_labels, label_rot)

                    #print("rotation correct ", correct_syn_rod_rot)                    


                    loss.backward()
                    tb.add_scalar("syn-ROD Rotation Loss", rot_loss, epoch)
                    del rgb_img, depth_img, unpooled_rgb, unpooled_depth, pred_labels, loss,label_rot,rot_loss

                    tmp = rod_train_rot_loader_iter.get_next()
                    rgb_img=tmp[0].to(device)
                    depth_img=tmp[1].to(device)
                    label_img=tmp[3].to(device)
                    _, unpooled_rgb = rgb_extrector(rgb_img)
                    _, unpooled_depth = depth_extrector(depth_img)
                    pred_labels = rot_classifier(torch.cat((unpooled_rgb, unpooled_depth), 1))
                    rot_loss = entropy_loss(pred_labels, label_img)
                    loss = rot_loss * runs.weight_rot   # 0.5, 1
                    correct_rod_rot  += get_num_correct(pred_labels, label_img)
                    total_rod += rgb_img.shape[0]
                    #print("rotation correct ", get_num_correct(pred_labels, label_img))

                    loss.backward()
                    tb.add_scalar("ROD Rotation Loss", rot_loss)
                    del rgb_img, depth_img, label_img, unpooled_rgb, unpooled_depth, pred_labels, loss
        if epoch % 4 == 0:
          save_checkpoint(checkpoint_path, epoch, net_list, opt_lis)
          print("checkpoint saved")
        tb.add_scalar("syn_rod_accuracy ", correct_syn_rod/total_syn, epoch)
        tb.add_scalar("syn_rod_accuracy_rot ", correct_syn_rod_rot/total_syn, epoch)
        tb.add_scalar("srod_accuracy rot ", correct_rod_rot/total_rod, epoch)
        print("syn_rod_accuracy ", correct_syn_rod/total_syn)
        print("syn_rod_accuracy_rot ", correct_syn_rod_rot/total_syn)
        print("srod_accuracy rot ", correct_rod_rot/total_rod)


        # Evaluation of object calssifier on synROD
        with EvaluationManager(net_list):
           #syn_Rod_evaluation_loader_iter = iter(syn_Rod_evaluation_loader) 
           correct = 0.0
           num_predictions = 0.0
           val_loss = 0.0
           for i in range(0, 300):
           #for tmp in syn_Rod_evaluation_loader_iter:   # use this to test on the whole ROD dataset
             tmp = syn_Rod_evaluation_loader_iter.get_next()
             rgb_img = tmp[0].to(device)
             depth_img = tmp[1].to(device)
             label = tmp[2].to(device)

             rgb_out, _ = rgb_extrector(rgb_img)
             depth_out, _ = depth_extrector(depth_img)
             out_concat = torch.cat((rgb_out, depth_out), 1)
             pred_labels = obj_classifier(out_concat) 
             pred_labels =   pred_labels.type(torch.float32)

             val_loss += entropy_loss(pred_labels, label).item() 
             correct +=get_num_correct( pred_labels, label) 
             num_predictions += rgb_img.shape[0]
             del rgb_img, depth_img, label, pred_labels,rgb_out, depth_out, out_concat

           accuracy = correct/ num_predictions
           actual_loss = val_loss/ num_predictions 
           print("Epoch: {} -- validation of source-- accuracy: {}".format(epoch, accuracy))
        
        recod.update_classification(actual_loss, accuracy)

        # Evaluation of rotation classifier on ROD
        with EvaluationManager(net_list):
           #rod_evaluation_Rot_loader_iter = iter(rod_evaluation_Rot_loader) 
           correct = 0.0
           num_predictions = 0.0
           val_loss = 0.0
           for i in range(0 ,150):
           #for tmp in rod_evaluation_Rot_loader_iter:   # use this to test on the whole ROD dataset
             tmp = rod_evaluation_Rot_loader_iter.get_next()
             rgb_img = tmp[0].to(device)
             depth_img = tmp[1].to(device)
             label = tmp[3].to(device)

             _ , rgb_out = rgb_extrector(rgb_img)
             _ , depth_out = depth_extrector(depth_img)
             out_concat = torch.cat((rgb_out, depth_out), 1)
             pred_labels = rot_classifier(out_concat) 
             pred_labels =   pred_labels.type(torch.float32)

             val_loss = entropy_loss(pred_labels, label) 
             correct +=get_num_correct( pred_labels, label) 
             num_predictions += rgb_img.shape[0]
             del rgb_img, depth_img, label, pred_labels,rgb_out, depth_out, out_concat

           accuracy = correct/ num_predictions
           actual_loss = val_loss/ num_predictions  
           print("Epoch: {} -- validation of target -- rotation accuracy: {}".format(epoch, accuracy))
        
        recod.update_test_rot(actual_loss, accuracy)

        # Evaluation of rotation classifier on synROD
        with EvaluationManager(net_list):
           #syn_Rod_evaluation_rot_loader_iter = iter(syn_Rod_evaluation_rot_loader) 
           correct = 0.0
           num_predictions = 0.0
           val_loss = 0.0
           for i in range(0 ,150):
           #for tmp in syn_Rod_evaluation_rot_loader_iter:   # use this to test on the whole ROD dataset
             tmp = syn_Rod_evaluation_rot_loader_iter.get_next()
             rgb_img = tmp[0].to(device)
             depth_img = tmp[1].to(device)
             label = tmp[3].to(device)

             _ , rgb_out = rgb_extrector(rgb_img)
             _ , depth_out = depth_extrector(depth_img)
             out_concat = torch.cat((rgb_out, depth_out), 1)
             pred_labels = rot_classifier(out_concat) 
             pred_labels =   pred_labels.type(torch.float32)

             val_loss += entropy_loss(pred_labels, label) 
             correct +=get_num_correct( pred_labels, label) 
             num_predictions += rgb_img.shape[0]
             del rgb_img, depth_img, label, pred_labels,rgb_out, depth_out, out_concat

           accuracy = correct/ num_predictions
           actual_loss = val_loss/ num_predictions   # need to confirm form prof, either to divide by batch size or total no of imgs
           print("Epoch: {} -- validation of source -- rotation accuracy: {}".format(epoch, accuracy))    
        recod.update_source_rot(actual_loss, accuracy)
        recod.end_epoch()
    

#make folder result_csv in to save results, save with your own name
#recod.save_to_csv("result.csv")
recod.df.to_csv("./result_csv")

del rgb_extrector, depth_extrector, obj_classifier, rot_classifier


torch.cuda.empty_cache()