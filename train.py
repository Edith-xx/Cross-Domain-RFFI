import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from confusion_matrix import confusion
from MixStyle_model import my_resnet
from raw_data_loader import *
#from thop import profile as thop_profile
#from torchsummary import summary
from torchvision import transforms, models
import random
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
class Config:
    def __init__(
        self,
        batch_size: int = 32,
        test_batch_size: int = 32,
        epochs: int = 100,
        lr: float = 0.001,
        save_path: str = 'model_weight/Raw2.pth',
        device_num: int = 0,
        rand_num: int = 30,
        ):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.save_path = save_path
        self.device_num = device_num
        self.rand_num = rand_num
        self.train_lossses = []
        self.val_accuracies = []
def train(model, train_dataloader, optimizer, epoch, writer, device_num, alpha=0.7):
    model.train()
    device = torch.device("cuda:"+str(device_num))
    correct = 0
    classifier_loss =0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    cpu_start_time = time.time()
    start_event.record()
    for data_nnl in train_dataloader:
        data, target = data_nnl
        target = target.squeeze().long()
        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
        optimizer.zero_grad()
        output = features = model(data)
        classifier_output = F.log_softmax(output, dim=1)
        #classifier_loss_batch = mixup_criterion(loss, classifier_output, target_a, target_b, lam)
        classifier_loss_batch = loss(classifier_output, target)
        result_loss_batch = classifier_loss_batch
        result_loss_batch.backward()
        optimizer.step()
        classifier_loss += classifier_loss_batch.item()
        pred = classifier_output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    end_event.record()
    torch.cuda.synchronize()
    cpu_end_time = time.time()
    gpu_time = start_event.elapsed_time(end_event) / 1000  
    cpu_time = cpu_end_time - cpu_start_time 
    print(f"Epoch {epoch}: GPU time = {gpu_time:.2f} seconds, CPU time = {cpu_time:.2f} seconds")
    classifier_loss /= len(train_dataloader)
    print('Train Epoch: {} \tLoss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
        epoch,
        classifier_loss,
        correct,
        len(train_dataloader.dataset),
        100.0 * correct / len(train_dataloader.dataset))
    )            
    writer.add_scalar('Accuracy/train', 100.0 * correct / len(train_dataloader.dataset), epoch)
    writer.add_scalar('Loss/train', classifier_loss, epoch) 
    return classifier_loss
def evaluate(model, loss, val_dataloader, epoch, writer, device_num):
    model.eval()
    val_loss = 0
    correct = 0
    device = torch.device("cuda:"+str(device_num))
    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.squeeze().long()
            if torch.cuda.is_available():
                data = data.to(device)
                target = target.to(device)
            output = features = model(data)
            output = F.log_softmax(output, dim=1)
            val_loss += loss(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    val_loss /= len(val_dataloader)
    fmt = '\nValidation set: loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            val_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset),
        )
    )
    accuracy = 100.0 * correct / len(val_dataloader.dataset)
    writer.add_scalar('Accuracy/validation', accuracy, epoch)
    writer.add_scalar('Loss/validation', val_loss, epoch)
    return val_loss, accuracy
def test(model, test_dataloader):
    model.eval()
    correct = 0
    total_max_values_correct = 0  
    total_correct_samples = 0 
    total_max_values = 0
    total_samples = 0
    total_max_values_wrong = 0  
    total_wrong_samples = 0  
    features_list = []
    labels_list = []
    target_pred = []
    target_real = []
    probabilities_list = []  
    t = 0
    i = 0
    with torch.no_grad():
        for data, target in test_dataloader:
            start_time = time.time()
            target = target.long()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = features = model(data)
            out = F.log_softmax(output, dim=1)
            max_values, max_indices = torch.max(out, dim=1)
            total_max_values += max_values.sum().item()  
            total_samples += data.size(0) 
            pred = output.argmax(dim=1, keepdim=True)  
            correct_mask = pred.eq(target.view_as(pred)) 
            correct += correct_mask.sum().item()
            max_values_correct = max_values[correct_mask.squeeze()] 
            total_max_values_correct += max_values_correct.sum().item() 
            total_correct_samples += correct_mask.sum().item()  
            incorrect_mask = ~correct_mask.squeeze()  
            max_values_wrong = max_values[incorrect_mask]
            total_max_values_wrong += max_values_wrong.sum().item() 
            total_wrong_samples += incorrect_mask.sum().item()  
            end_time = time.time()
            elapsed_time = end_time - start_time
            t += elapsed_time
            i += 1
            features_list.append(features.cpu().numpy())  
            labels_list.append(target.cpu().numpy())
            target_pred.extend(pred.view(-1).tolist())
            target_real.extend(target.view(-1).tolist())
            probabilities_list.append(out.cpu().numpy())
        t /= i
        print("Average processing time per batch:", t)
        target_pred = np.array(target_pred)
        target_real = np.array(target_real)
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    probabilities = np.concatenate(probabilities_list, axis=0)
    accuracy = correct / len(test_dataloader.dataset)
    print("Accuracy:", accuracy)
    return target_pred, target_real, all_features, all_labels, probabilities
def train_and_evaluate(model, loss_function, train_dataloader, val_dataloader, optimizer, epochs, writer, save_path, device_num):
    train_losses = []
    val_losses = []
    current_min_val_loss = 100         
    #early_stopping = EarlyStopping(save_path)
    time_start1 = time.time()
    for epoch in range(1, epochs + 1):
        time_start = time.time()
        train_loss = train(model, train_dataloader, optimizer, epoch, writer, device_num, 0.9)
        train_losses.append(train_loss)
        val_loss, val_accuracy = evaluate(model, loss_function, val_dataloader, epoch, writer, device_num)
        val_losses.append(val_loss)
        if val_loss < current_min_val_loss:
            print("The validation loss is decreased from {} to {}, new model weight is saved.".format(
                current_min_val_loss, val_loss))
            current_min_val_loss = val_loss
            torch.save(model, save_path)
        else:
            print("The validation loss is not decreased.")
        time_end = time.time()
        time_sum = time_end - time_start
        print("time for each epoch is: %s" % time_sum)
        print("------------------------------------------------")
    time_end1 = time.time()
    Ave_epoch_time = (time_end1 - time_start1) / epochs
    print("Avgtime for each epoch is: %s" % Ave_epoch_time)
    return train_losses, val_losses
if __name__ == '__main__':
    conf = Config()                                                   
    writer = SummaryWriter("logs")
    device = torch.device("cuda:"+str(conf.device_num))
    RANDOM_SEED = 300  # any random number
    set_seed(RANDOM_SEED)
    run_for = 'Train'
    #run_for = 'Classification'
    if run_for == 'Train':

        X_train, X_val, Y_train, Y_val = read_train_data()
        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
        train_dataloader = DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True)
        X = torch.Tensor(X_train)
        Y = torch.Tensor(X_val)
        print(X.size())
        print(Y.size())
        val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
        val_dataloader = DataLoader(val_dataset, batch_size=conf.batch_size, shuffle=True)
        model = models.swin_t(weights="DEFAULT")
        print(model)
        X1 = X[:1]
        print(X1.size())
        flops, params = thop_profile(model, inputs=(X1,))
        print('flops:{}'.format(flops))
        print('params:{}'.format(params))
        if torch.cuda.is_available():
            model = model.to(device)
        loss = nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss = loss.to(device)
        summary(model, input_size=(1, 102, 45))
        optim = torch.optim.Adam(model.parameters(), lr=conf.lr, weight_decay=0)
        time_start = time.time()
        train_losses, val_losses = train_and_evaluate(model, loss_function=loss, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                       optimizer=optim, epochs=conf.epochs, writer=writer, save_path=conf.save_path,
                       device_num=conf.device_num)
        time_end = time.time()
        time_sum = time_end - time_start
        print("total training time is: %s" % time_sum)
        print(train_losses)
        print(val_losses)
    elif run_for == 'Classification':
        X_test, Y_test, = read_test_data()
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
        data = torch.tensor(X_test)
        test_dataloader = DataLoader(test_dataset)
        model = torch.load('model_weight/Raw2.pth')
        pred, real, features, labels, probabilities = test(model, test_dataloader)
        confusion(pred, real, range(7))





