# PaddlePaddle-mnist-lenet5

# 卷积神经网络

卷积神经网络通常包含以下几种层：

    卷积层（Convolutional layer），卷积神经网路中每层卷积层由若干卷积单元组成，每个卷积单元的参数都是通过反向传播算法优化得到的。卷积运算的目的是提取输入的不同特征，第一层卷积层可能只能提取一些低级的特征如边缘、线条和角等层级，更多层的网络能从低级特征中迭代提取更复杂的特征。
    线性整流层（Rectified Linear Units layer, ReLU layer），这一层神经的活性化函数（Activation function）使用线性整流（Rectified Linear Units, ReLU）f(x)=max(0,x)。
    池化层（Pooling layer），通常在卷积层之后会得到维度很大的特征，将特征切成几个区域，取其最大值或平均值，得到新的、维度较小的特征。
    全连接层（ Fully-Connected layer）, 把所有局部特征结合变成全局特征，用来计算最后每一类的得分。

LeNet-5 的结构就如下所示：

    卷积层 – 池化层- 卷积层 – 池化层 – 卷积层 – 全连接层

# 关于PaddlePaddle的使用,飞桨卷积API介绍

飞桨卷积算子对应的API是paddle.fluid.dygraph.Conv2D，我们可以直接调用API进行计算，也可以在此基础上修改。常用的参数如下：

    num_channels (int) - 输入图像的通道数。
    num_fliters (int) - 卷积核的个数，和输出特征图通道数相同，相当于C_out。
    filter_size(int|tuple) - 卷积核大小，可以是整数，比如3，表示卷积核的高和宽均为3 ；或者是两个整数的list，例如[3,2]，表示卷积核的高为3，宽为2。
    stride(int|tuple) - 步幅，可以是整数，默认值为1，表示垂直和水平滑动步幅均为1；或者是两个整数的list，例如[3,2]，表示垂直滑动步幅为3，水平滑动步幅为2。
    padding(int|tuple) - 填充大小，可以是整数，比如1，表示竖直和水平边界填充大小均为1；或者是两个整数的list，例如[2,1]，表示竖直边界填充大小为2，水平边界填充大小为1。
    act（str）- 应用于输出上的激活函数，如Tanh、Softmax、Sigmoid，Relu等，默认值为None。

输入数据维度[N,C_in,H_in,W_in]，输出数据维度[N,num_filters,H_out,W_out]，权重参数www的维度[num_filters,C_in,filter_size_h,filter_size_w]，偏置参数bbb的维度是[num_filters]。

# 代码的解读

第一步：首先是导入需要使用的库，作者环境是python-3.7和PaddlePaddle-1.8

    import os
    import random
    import paddle
    import paddle.fluid as fluid
    from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear
    import numpy as np
    from PIL import Image
    import gzip
    import json

第二步：获取数据集

解压数据集并获取数据

    datafile = './mnist.json.gz'
    data = json.load(gzip.open(datafile))
    train_set, val_set, eval_set = data

数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS

    IMG_ROWS = 28
    IMG_COLS = 28

训练模式下需要打乱训练数据

    if mode == 'train':
            random.shuffle(index_list)
            
按照索引读取数据

    for i in index_list:   
            # 读取图像和标签，转换其尺寸和类型
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img) 
            labels_list.append(label)
            # 如果当前数据缓存达到了batch size，就返回一个批次数据
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                
    # 如果剩余数据的数目小于BATCHSIZE，
    # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
    if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)
            
定义模型结构

    class MNIST(fluid.dygraph.Layer):   
       def __init__(self):
         super(MNIST, self).__init__()
         
         # 定义一个卷积层，使用relu激活函数
         self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一个卷积层，使用relu激活函数
         self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
         # 定义一个池化层，池化核为2，步长为2，使用最大池化方式
         self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
         # 定义一个全连接层，输出节点数为10 
         self.fc = Linear(input_dim=980, output_dim=10, act='softmax')

定义网络的前向计算过程

    def forward(self, inputs, label):
         x = self.conv1(inputs)
         x = self.pool1(x)
         x = self.conv2(x)
         x = self.pool2(x)
         x = fluid.layers.reshape(x, [x.shape[0], 980])
         x = self.fc(x)
         if label is not None:
             acc = fluid.layers.accuracy(input=x, label=label)
             return x, acc
         else:
             return x
           
开始训练

    #调用加载数据的函数
    train_loader = load_data('train')
    
    #在使用GPU机器时，可以将use_gpu变量设置成True
    use_gpu = True
    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()

    with fluid.dygraph.guard(place):
        model = MNIST()
        model.train() 
    
        #四种优化算法的设置方案，可以逐一尝试效果
        optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.01, parameter_list=model.parameters())
        #optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=0.01, momentum=0.9, parameter_list=model.parameters())
        #optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.01, parameter_list=model.parameters())
        #optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.01, parameter_list=model.parameters())
    
        EPOCH_NUM = 50
        for epoch_id in range(EPOCH_NUM):
            for batch_id, data in enumerate(train_loader()):
                #准备数据
                image_data, label_data = data
                image = fluid.dygraph.to_variable(image_data)
                label = fluid.dygraph.to_variable(label_data)

                #前向计算的过程，同时拿到模型输出值和分类准确率
                predict, acc = model(image, label)
            
                #计算损失，取一个批次样本损失的平均值
                loss = fluid.layers.cross_entropy(predict, label)
                avg_loss = fluid.layers.mean(loss)
            
                #每训练了200批次的数据，打印下当前Loss的情况
                if batch_id % 200 == 0:
                    print("epoch: {:0>3d}, batch: {:0>3d}, loss is: {:.5f}, acc is {:.5f}".format(epoch_id, batch_id, avg_loss.numpy()[0], acc.numpy()[0]))
            
                #后向传播，更新参数的过程
                avg_loss.backward()
                optimizer.minimize(avg_loss)
                model.clear_gradients()
                
保存模型

    fluid.save_dygraph(model.state_dict(), 'mnist')
