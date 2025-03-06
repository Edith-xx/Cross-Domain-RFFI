from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def confusion(y_pred,y_true,labels):
    # y_pred = [] # ['2','2','3','1','4'] # 类似的格式
    # y_true = [] # ['0','1','2','3','4'] # 类似的格式
    # 对上面进行赋值

    C = confusion_matrix(y_true, y_pred, labels=labels) # 可将'1'等替换成自己的类别，如'cat'。

    plt.matshow(C, cmap=plt.cm.Blues) # 根据最下面的图按自己需求更改颜色
    # plt.colorbar()

    for i in range(len(C)):
        for j in range(len(C)):
            plt.annotate(C[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')

    # plt.tick_params(labelsize=15) # 设置左边和上面的label类别如0,1,2,3,4的字体大小。

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.ylabel('True label', fontdict={'family': 'Times New Roman', 'size': 20}) # 设置字体大小。
    # plt.xlabel('Predicted label', fontdict={'family': 'Times New Roman', 'size': 20})
    # plt.xticks(range(0,5), labels=['a','b','c','d','e']) # 将x轴或y轴坐标，刻度 替换为文字/字符
    # plt.yticks(range(0,5), labels=['a','b','c','d','e'])
    plt.show()

