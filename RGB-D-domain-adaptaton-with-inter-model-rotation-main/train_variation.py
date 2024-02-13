from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.optim as opt
from dataloader import MyDataset, MyTransformer
from net import Obj_classifier, Resnet_extractor, Obj_classifier, Rot_classifier, Rot_regressor
from utils import RunBuilder, DataWraper,OptimizerManager, load_checkpoint, save_checkpoint, get_num_correct, entropy_loss_paper, EvaluationManager, RunRecord
import os



syn_Rod_train = MyDataset("./ROD-synROD/synROD/synARID_50k-split_sync_train1.txt", fold_name = "synROD")
syn_Rod_evaluation = MyDataset("./ROD-synROD/synROD/synARID_50k-split_sync_test1.txt", fold_name = "synROD")

Rod_entropy_paper = MyDataset("./ROD-synROD/ROD/wrgbd_40k-split_sync.txt", fold_name = "ROD")

syn_Rod_train_Rot = MyDataset("./ROD-synROD/synROD/synARID_50k-split_sync_train1.txt",rotate = True, fold_name = "synROD", discrete = False )
syn_Rod_evaluation_Rot = MyDataset("./ROD-synROD/synROD/synARID_50k-split_sync_test1.txt", rotate = True, fold_name = "synROD", discrete = False)

rod_train_rot = MyDataset("./ROD-synROD/ROD/wrgbd_40k-split_sync.txt", rotate = True, fold_name = "ROD", discrete = False)
rod_evaluation_rot = MyDataset("./ROD-synROD/ROD/wrgbd_40k-split_sync.txt", rotate = True, fold_name = "ROD", discrete = False)

tb = SummaryWriter('./log')

resume = False



parameters = dict(
    lr = [0.0003], 
    batch_size = [32, 64], #64
    weight_decay = [0.05, 0.5], #0.5
    num_workers=[2,4],
    epoch = [40],
    weight_entropy = [0.1],
    weight_rot = [1]
)



device = torch.device('cuda')
networks = []
entropy_loss = nn.CrossEntropyLoss()
recod = RunRecord()
cos_loss = nn.MSELoss()
sin_loss = nn.MSELoss()
for runs in RunBuilder.get_runs(parameters):
    
    checkpoint_path = os.path.join("./checkpoint", "checkpoint.pth")
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
    rgb_extrector = Resnet_extractor().to(device)
    depth_extrector = Resnet_extractor().to(device)
    obj_classifier = Obj_classifier().to(device)
    rot_regressor = Rot_regressor().to(device)
    net_list = [rgb_extrector, depth_extrector, obj_classifier, rot_regressor]
    
    first_epoch = None

    rgb_extrector_opt = optim.SGD(rgb_extrector.parameters(), lr=runs.lr, momentum=0.9, weight_decay=runs.weight_decay)
    depth_extrector_opt = optim.SGD(depth_extrector.parameters(), lr=runs.lr, momentum=0.9, weight_decay=runs.weight_decay)
    obj_classifier_opt = optim.SGD(obj_classifier.parameters(), lr=runs.lr, momentum=0.9, weight_decay=runs.weight_decay) 
    rot_regressor_opt = optim.SGD(rot_regressor.parameters(), lr=runs.lr, momentum=0.9, weight_decay=runs.weight_decay)
    
    opt_lis = [rgb_extrector_opt, depth_extrector_opt, obj_classifier_opt, rot_regressor_opt]
    if not resume:
      first_epoch = 1
    else:
      first_epoch = load_checkpoint(checkpoint_path, first_epoch, net_list, opt_lis)
      rgb_extrector = net_list[0]
      depth_extrector = net_list[1]
      obj_classifier = net_list[2]
      rot_regressor = net_list[3]
                                    

    # Network architecture

    for epoch in range(first_epoch, first_epoch+1 ):
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

        # for i in range(0, 1000, 62):
        count = 0
        for tmp in syn_rod_train_loader:
          count+=1
          # tmp = syn_rod_train_loader_iter.get_next()
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
                # print("pred_labels ", pred_labels.shape)
                # print("labels ", label_img.shape)
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
                # print("okkk", loss.dtype)
                loss.backward()
                tb.add_scalar("Entropy Loss", loss, epoch)
                del rgb_img, depth_img, label_img, rgb_out, depth_out, out_concat, pred_labels,loss

                if runs.weight_rot > 0.0:                    
                    tmp1 = syn_rod_train_rot_loader_iter.get_next()
                    rgb_img = tmp1[0].to(device)
                    depth_img  = tmp1[1].to(device)
                    label_cos = tmp1[3].to(device)
                    label_sin = tmp1[4].to(device)
                    _, unpooled_rgb = rgb_extrector(rgb_img)
                    _, unpooled_depth = depth_extrector(depth_img)
                    pred_cos, pred_sin = rot_regressor(torch.cat((unpooled_rgb, unpooled_depth), 1))

                    pred_cos = torch.reshape(pred_cos, (-1,))
                    cos_los = cos_loss(pred_cos, label_cos)
                    # print(pred_cos.shape)
                    # print(label_cos.shape)
                    # print(cos_los.shape)
                    # print(loss.dtype)
                    pred_sin = torch.reshape(pred_sin, (-1,))
                    sin_los = sin_loss(pred_sin, label_sin)
                    # print(loss)

                    loss = cos_los + sin_los
                    # loss = loss.type(torch.float32)

                    # print(rot_loss)
                    # correct_syn_rod_rot +=get_num_correct(pred_cos, label_cos)
                    # correct_syn_rod_rot +=get_num_correct(pred_sin, label_sin)

                    #print("rotation correct ", correct_syn_rod_rot)  

                    # print("loss type", loss.dtype)
                    loss.backward()
                    tb.add_scalar("syn-ROD Rotation Loss", loss, epoch)
                    del rgb_img, depth_img, unpooled_rgb, unpooled_depth, pred_cos, pred_sin, loss,label_cos, label_sin

                    tmp1 = rod_train_rot_loader_iter.get_next()
                    rgb_img = tmp1[0].to(device)
                    depth_img  = tmp1[1].to(device)
                    label_cos = tmp1[3].to(device)
                    label_sin = tmp1[4].to(device)
                    _, unpooled_rgb = rgb_extrector(rgb_img)
                    _, unpooled_depth = depth_extrector(depth_img)
                    pred_cos, pred_sin = rot_regressor(torch.cat((unpooled_rgb, unpooled_depth), 1))
                    cos_los = cos_loss(pred_cos, label_cos)
                    sin_los = sin_loss(pred_sin, label_sin)

                    loss = (cos_los + sin_los) # shouldn't we assign the same weight of the loss of the pretext task?
                    total_rod += rgb_img.shape[0]
                    #print("rotation correct ", get_num_correct(pred_labels, label_img))
                    loss.backward()         
                    tb.add_scalar("ROD Rotation Loss", loss)
                    del rgb_img, depth_img, unpooled_rgb, unpooled_depth, pred_cos, pred_sin, loss,label_cos, label_sin
        if epoch % 5 == 0:
          save_checkpoint(checkpoint_path, epoch, net_list, opt_lis)
          print("checkpoint saved")
        tb.add_scalar("syn_rod_accuracy ", correct_syn_rod/total_syn, epoch)
        tb.add_scalar("syn_rod_accuracy_rot ", correct_syn_rod_rot/total_syn, epoch)
        tb.add_scalar("srod_accuracy rot ", correct_rod_rot/total_rod, epoch)
        print("syn_rod_accuracy ", correct_syn_rod/total_syn)
        # print("syn_rod_accuracy_rot ", correct_syn_rod_rot/total_syn)
        # print("srod_accuracy rot ", correct_rod_rot/total_rod)
        with EvaluationManager(net_list):
          #  syn_Rod_evaluation_loader_iter = iter(syn_Rod_evaluation_loader) 
           correct = 0.0
           num_predictions = 0.0
           val_loss = 0.0
           for i in range(0, 100):
          #  for tmp in syn_Rod_evaluation_loader_iter:
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
           actual_loss = val_loss/ num_predictions   # need to confirm form prof, either to divide by batch size or total no of imgs
           print("Epoch: {} -- validation of source-- accuracy: {}".format(epoch, accuracy))
        
        recod.update_classification(actual_loss, accuracy)
        with EvaluationManager(net_list):
          #  rod_evaluation_Rot_loader_iter = iter(rod_evaluation_Rot_loader) 
           correct = 0.0
           num_predictions = 0.0
           val_loss = 0.0
           for i in range(0 ,100):
          #  for tmp in rod_evaluation_Rot_loader_iter:
             tmp = rod_evaluation_Rot_loader_iter.get_next()
             rgb_img = tmp[0].to(device)
             depth_img = tmp[1].to(device)
             label = tmp[3].to(device)

             _ , rgb_out = rgb_extrector(rgb_img)
             _ , depth_out = depth_extrector(depth_img)
             out_concat = torch.cat((rgb_out, depth_out), 1)
             pred_labels = rot_regressor(out_concat) 
             pred_labels =   pred_labels.type(torch.float32)

             val_loss = entropy_loss(pred_labels, label) 
             correct +=get_num_correct( pred_labels, label) 
             num_predictions += rgb_img.shape[0]
             del rgb_img, depth_img, label, pred_labels,rgb_out, depth_out, out_concat

           accuracy = correct/ num_predictions
           actual_loss = val_loss/ num_predictions   # need to confirm form prof, either to divide by batch size or total no of imgs
           print("Epoch: {} -- validation of target -- rotation accuracy: {}".format(epoch, accuracy))

        recod.update_test_rot(actual_loss, accuracy)
        with EvaluationManager(net_list):
          #  syn_Rod_evaluation_rot_loader_iter = iter(syn_Rod_evaluation_rot_loader) 
           correct = 0.0
           num_predictions = 0.0
           val_loss = 0.0
           for i in range(0 ,100):
          #  for tmp in syn_Rod_evaluation_rot_loader_iter:
             tmp = syn_Rod_evaluation_rot_loader_iter.get_next()
             rgb_img = tmp[0].to(device)
             depth_img = tmp[1].to(device)
             label = tmp[3].to(device)

             _ , rgb_out = rgb_extrector(rgb_img)
             _ , depth_out = depth_extrector(depth_img)
             out_concat = torch.cat((rgb_out, depth_out), 1)
             pred_labels = rot_regressor(out_concat) 
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
recod.save_to_csv("giuseppe1.csv")
recod.df.to_csv("./result_csv")

del rgb_img, depth_img, unpooled_rgb, unpooled_depth, pred_cos, pred_sin, loss,label_cos, label_sin

torch.cuda.empty_cache()

print(count)