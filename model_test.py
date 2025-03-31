from torch.utils.data import DataLoader
from util import netDataset
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.manifold import TSNE
from model import Model
from scipy.io import savemat
import itertools

def eval(model, test_loader):
    eval_loss = 0.0
    total_acc = 0.0
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    predictions = []
    labels = []
    probs = []

    for i, batch in enumerate(test_loader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            logits = model(x)
            batch_loss = loss_function(logits, y)
            eval_loss += batch_loss.item()
            _, preds = logits.max(1)

            num_correct = (preds == y).sum().item()
            total_acc += num_correct

            predictions.extend(preds.tolist())
            labels.extend(y.tolist())
            probs.extend(logits.softmax(dim=1).cpu().numpy())

    loss = eval_loss / len(test_loader)
    acc = round((total_acc / (len(test_loader) * eval_batch_size)) * 100, 2)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    auc = roc_auc_score(labels, np.array(probs), multi_class='ovr')

    return loss, acc, precision, recall, f1, auc, np.array(predictions), np.array(labels)

def extract(model, test_loader):
    model.eval()
    feature = torch.tensor([])
    label = torch.tensor([])
    for i, batch in enumerate(test_loader):
        x, y = batch
        x = x.to(device)
        with torch.no_grad():
            fea = model(x, True)
            fea = fea.detach().cpu()
            feature = torch.cat([feature, fea])
            label = torch.cat([label, y])

    return feature.numpy(), label.numpy()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("load dataset.........................")
    trainpath = 'image/train/'  ####
    valpath = 'image/test/'  ####

    img_size = 64
    eval_batch_size = 8

    train_lines = os.listdir(trainpath)
    val_lines = os.listdir(valpath)

    model = Model(num_classes=10).to(device)

    train_dataset = netDataset(trainpath, train_lines, img_size)
    val_dataset = netDataset(valpath, val_lines, img_size)
    train_loader = DataLoader(train_dataset, shuffle=False, batch_size=eval_batch_size,
                              num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(val_dataset, shuffle=False, batch_size=eval_batch_size,
                             num_workers=0, pin_memory=True, drop_last=False)
    try:
        net_dict = model.state_dict()
        model_para = torch.load('output/best.pt').to(device)

        state_dict = {k: v for k, v in model_para.named_parameters() if k in net_dict.keys()}
        net_dict.update(state_dict)
        model.load_state_dict(net_dict)
    except:
        model = torch.load('output/best.pt', map_location=torch.device('cpu')).to(device)

    print('model test-----------------')
    eval_loss, eval_acc, precision, recall, f1, auc, predictions, labels = eval(model, test_loader)
    print(predictions)
    print(labels)
    acc_str = "{:.2f}%".format(eval_acc)
    print("准确率：", acc_str)
    print("精确率：", precision)
    print("召回率：", recall)
    print("F1-score：", f1)
    print("AUC：", auc)

    print('feature extraction-----------------')
    train_feature, train_label = extract(model, train_loader)
    test_feature, test_label = extract(model, test_loader)
    print('train_feature.shape:', train_feature.shape)
    print('train_label.shape:', train_label.shape)
    print('test_feature.shape:', test_feature.shape)
    print('test_label.shape:', test_label.shape)

    savemat('output/feature.mat', {'train_feature': train_feature,
                                   'train_label': train_label,
                                   'test_feature': test_feature,
                                   'test_label': test_label})


    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    tsne = TSNE(n_components=2,random_state=33)
    tsne_results = tsne.fit_transform(test_feature)

    plt.figure(figsize=(8, 8))
    for i, label in enumerate(np.unique(test_label)):
        plt.scatter(tsne_results[test_label == label, 0], tsne_results[test_label == label, 1], label=int(label))
    plt.legend()
    plt.legend(fontsize=16)
    plt.xlabel('Dimension 1', fontsize=18)
    plt.ylabel('Dimension 2', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig('result/Feature visualization.jpg', dpi=600)
    plt.show()

    con_mat = confusion_matrix(labels.astype(str), predictions.astype(str))
    classes = list(set(labels))
    classes.sort()

    plt.imshow(con_mat, cmap=plt.cm.Blues, alpha=0.8)
    indices = range(len(con_mat))
    plt.xticks(indices, classes, fontsize='16')
    plt.yticks(indices, classes, fontsize='16')
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize='16')
    plt.xlabel('Prediction', fontsize='18')
    plt.ylabel('True', fontsize='18')
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index, second_index, con_mat[second_index][first_index], va='center', ha='center',
                     fontsize='16')

    plt.savefig(r'result/confusion matrix.jpg', dpi=600)
    plt.show()
