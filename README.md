# 2D-UNet-Pytorch
使用2D-UNet和2D-UNet++(Nested UNet)对Chaos、Promise12两个数据集进行分割
## 数据集获取
### CHAOS
https://chaos.grand-challenge.org/Combined_Healthy_Abdominal_Organ_Segmentation/
在官网下载好数据后，解压CHAOS_Train_Sets.zip压缩包，将其下的CT文件夹复制到代码目录的data/chaos文件夹中。



    
### PROMISE12
https://promise12.grand-challenge.org/
在官网下载好数据后，训练数据存放在三个压缩包中，将三个压缩包分别解压，并将内容复制到代码目录的data/promise12文件夹中。

具体数据存放格式如下:

    data
    ├── chaos
        ├──CT
            ├──1
            ├──2
            ├──5
            ├──...
    ├── promise12
        ├──Case00.mhd
        ├──Case00.raw
        ├──Case00_segmentation.mhd
        ├──Case00_segmentation.raw
        ├──Case01.mhd
        ├──Case01.raw
        ├──Case01_segmentation.mhd
        ├──Case01_segmentation.raw
        ├──...
## 训练模型
在终端中输入 

    python train.py --model=unet --dataset=promise12
即可使用unet对promise12数据集进行训练，如果要使用unet++，就令参数--model=nestedunet，如果要使用chaos数据集，就令参数--dataset=chaos。
在模型训练开始，会在代码所在目录下生成logs_train文件夹，每次训练都会在该文件夹下生成一个子文件夹，记录当次训练的训练日志。
## 在tensorboard中观察训练曲线
代码在训练过程中会记录每个epoch在训练集上的loss和dice以及验证集上的loss和dice，并保存在tensorboard中。
在终端中输入

    tensorboard --logdir=logs_train
在浏览器中打开对应端口，即可使用tensorboard观察训练记录。
