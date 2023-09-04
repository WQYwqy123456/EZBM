# coding:utf-8
import os, json, atexit, time, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

#G类：存储日志的一些配置信息
class G:
    output_dir = None
    output_file = None
    first_row = True
    log_headers = []
    log_current_row = {}

#指定输出目录路径
def configure_output_dir(dir=None):
    G.output_dir = dir
    G.output_file = open(os.path.join(G.output_dir, "log.txt"), 'w')
    atexit.register(G.output_file.close)
    print("Logging data to %s" % G.output_file.name)

#保存超参数
def save_hyperparams(params):
    with open(os.path.join(G.output_dir, "hyperparams.json"), 'w') as out:
        out.write(json.dumps(params, separators=(',\n', '\t:\t'), sort_keys=True))

#保存torch模型
def save_pytorch_model(model):
    """
    Saves the entire pytorch Module
    """
    torch.save(model, os.path.join(G.output_dir, "model.pkl"))

#加载torch模型
def load_pytorch_model(model):
    """
    Saves the entire pytorch Module
    """
    temp = torch.load('model.pkl')
    model.resnet.load_state_dict(temp.resnet.state_dict())
    model.classifier.load_state_dict(temp.classifier.state_dict())

#计算分类准确率的函数，接受模型输出output, 真实标签target, 可选的topk参数 用于指定计算topk的准确率  返回一个包含准确率值的列表
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

#记录表格数据的函数
def log_tabular(key, val):
    if G.first_row:
        G.log_headers.append(key)
    else:
        assert key in G.log_headers
    assert key not in G.log_current_row
    G.log_current_row[key] = val

#将表格数据写入日志文件
def dump_tabular():
    vals = []
    for key in G.log_headers:
        val = G.log_current_row.get(key, "")
        vals.append(val)
    if G.output_dir is not None:
        if G.first_row:
            G.output_file.write("\t".join(G.log_headers))
            G.output_file.write("\n")
        G.output_file.write("\t".join(map(str, vals)))
        G.output_file.write("\n")
        G.output_file.flush()
    G.log_current_row.clear()
    G.first_row = False

#根据混淆矩阵内容，绘制出一个直观的可视化图表[真实标签，预测标签，类别列表，是否进行归一化, 图标标题，使用的颜色映射]
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix  计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

#根据论文中的采样策略，target标签列表，cls_num_list是每个类别目前拥有的样本数
def EasySampling(features, targets, cls_num_list):
    cls_num_list = np.array(cls_num_list)   #转成数组
    num_classes = len(cls_num_list)         #总共有多少类
    num_max = np.max(cls_num_list)          #类中最多的样本数
    num4gen = num_max/cls_num_list          
    new_features, new_targets = [], []
    for i in range(num_classes):            #遍历每个类
        idx4target = np.where(targets == i)[0]   #得到当前目标类的索引
        idx4other = np.where(targets != i)[0]    #得到当前其他类的索引
        if num4gen[i] -1 < 1:                    #当前类不需要采样就停止
            continue
        target_samples = features[idx4target]    #得到目标类特征
        other_samples = features[idx4other]      #得到其它类特征
        other_labels = targets[idx4other]        #得到标签列表
        num4add = int(num4gen[i] - 1)            #计算需要生成的新样本量

        # for each sample in target class generate num4add samples
        for j in range(len(idx4target)):    #遍历target_sample中的每一个样本
            temp = target_samples[j]
            temp_dis = np.sum(abs(temp - other_samples), axis=1)     #计算当前样本与其他样本的距离总和
            temp_idx = np.argpartition(temp_dis, num4add)[:num4add]   #根据距离对其他样本的索引进行分区，选择距离最近的num4add个
            temp_others = other_samples[temp_idx]                      #获取其他样本特征
            temp_labels = other_labels[temp_idx]                       #获取其他样本标签
            lam = cls_num_list[i]/(cls_num_list[i] + cls_num_list[temp_labels])    #根据论文中的公式计算生成样本权重
            lam = lam.reshape(num4add, -1)
            temp = (1-lam)*temp
            temp_others = lam*temp_others
            new_sample = temp + temp_others               #得到新生成的混合样本的权重
            new_label = np.array([i]*num4add)             #得到新样本的标签（跟target保持一致）
            new_features.extend(new_sample)
            new_targets.extend(new_label)

    return new_features, new_targets       #返回得到的新的样本特征和样本标签

#重写torch里面的采样器（用于采样不平衡数据）
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = [0] * len(np.unique(dataset.targets))
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_to_count[label] += 1

        beta = 0.9999
        effective_num = 1.0 - np.power(beta, label_to_count)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)

        # weight for each sample, 依概率选择样本
        weights = [per_cls_weights[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.targets[idx]

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples
