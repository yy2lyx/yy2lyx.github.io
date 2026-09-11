---
layout: article
title: "pytorch_lightning使用体验"
date: 2023-12-28 00:00:00
tags: [DeepLearning]
excerpt: "记录pytorch_lightning的使用过程心得体会"
cover: assets/images/lightning.jpg
article_header:
  type: cover
  image:
    src: assets/images/lightning.jpg
key: post-2024-1-18-pytorch-lightning
---



> 如果是一些小模型想要快速实验，不想怎么写代码的，可以通过pytorch_lightning快速搭建模型，但是如果涉及到大模型，以及分布式训练预测，咱还是老老实实用pytorch吧。

### 一. 使用体验

就像很多年前写过tensorflow之后看到keras后的欣喜，当我看到pytorch_lightning后瞬间就喜欢上了它！对于pytorch的重度使用者来说，每次都要写很多重复的训练预测代码，总感觉代码复用起来很麻烦，于是pytorch_lightning它来啦！

**pytorch_lightning的优势：**

* 代码可读性、复用性高

* 自由度和pytorch一样高，并没有像使用keras一样感觉封装过死的感觉。

* 能像keras一样快速搭建模型，简化模型训练和预测的过程

* 支持分布式训练

### 二. 安装和使用

官网地址是：https://lightning.ai/ 

pip进行安装：`pip show pytorch_lightning`

下面使用MNIST来展示如何使用pytorch_lightning来简化自己的代码

#### 2.1 数据模块LightningDataModule

通常情况下，我们需要做一些预处理，以及在定义完自己的dataset后，需要定义dataloader，这里可以直接继承LightningDataModule模块，直接重写其中的方法即可。

```python
class MNISTDataModule(LightningDataModule):
    def __init__(self,root_dir,val_size,num_workers,batch_size):
        super(MNISTDataModule, self).__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        """
        download data once
        """
        MNIST(self.hparams.root_dir, train=True, download=True)
        MNIST(self.hparams.root_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        setup dataset for each machine
        """
        dataset = MNIST(self.hparams.root_dir,
                        train=True,
                        download=False,
                        transform=T.ToTensor())
        train_length = len(dataset)
        self.train_dataset, self.val_dataset = \
            random_split(dataset,
                         [train_length - self.hparams.val_size, self.hparams.val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=self.hparams.num_workers,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)
```

#### 2.2 训练和预测模块LightningModule

之前每次训练和预测模型的时候，我都会写一个该过程的一个基类，来封装每个epoch模型训练、验证的过程，其实每次不同的项目、不同的模型继承了上述的基类，但是基本上也就是改变其中的每个batch训练、验证的方法，然后看到了一个别人封装的这么完美的训练预测基类，简直开心的不要不要的。。

```python
class MNISTModel(LightningModule):
    def __init__(self, hidden_dim, num_classes, lr, num_epochs):
        super().__init__()
        self.save_hyperparameters()
        self.net = LinearModel(self.hparams.hidden_dim)
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.hparams.num_classes
        )
        self.f1_score = torchmetrics.F1Score(task="multiclass", num_classes=self.hparams.num_classes)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        self.optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        scheduler = CosineAnnealingLR(self.optimizer,
                                      T_max=self.hparams.num_epochs,
                                      eta_min=self.hparams.lr / 1e2)

        return [self.optimizer], [scheduler]

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()

    def _common_step(self, batch, batch_idx):
        images, labels = batch
        logits_predicted = self(images)

        loss = self.loss_fn(logits_predicted, labels)
        acc = self.accuracy(logits_predicted, labels)
        # acc = torch.sum(torch.eq(torch.argmax(logits_predicted, -1), labels).to(torch.float32)) / len(labels)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch,batch_idx)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train_step_loss', loss)

        train_rs = {'train_loss': loss,
               'train_acc': acc}
        self.training_step_outputs.append(train_rs)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        rs = self(batch[0])
        rs = torch.argmax(rs, -1).numpy().tolist()
        return rs

    def validation_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)

        log = {'val_loss': loss,
               'val_acc': acc}
        self.validation_step_outputs.append(log)
        return log
```

#### 2.3 callbacks

这里如果我们觉得上面这些无法满足我们的日常训练、预测的需求，那么完全可以再增加一些其他需要的第三方和自己定义的callbacks，当然pytorch_lightning其实已经封装了很多常用的callbacks了，比如下面的几个常用的：

* 模型定义怎么保存ckpt：`ModelCheckpoint`

* 如何定义训练及早停止：`MINISTCallBack`

* 定义进度条：`TQDMProgressBar`

当然了，我们想定义属于自己的callback怎么弄呢：

```python
class MINISTCallBack(Callback):
    def __init__(self):
        super(MINISTCallBack, self).__init__()


    def on_predict_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        print("Predict is ending")

    def on_train_epoch_end(self, trainer : "pl.Trainer", pl_module: "pl.LightningModule"):
        epoch_mean_loss = torch.stack([x['train_loss'] for x in pl_module.training_step_outputs]).mean()
        epoch_mean_acc = torch.stack([x['train_acc'] for x in pl_module.training_step_outputs]).mean()
        pl_module.log("train/loss", epoch_mean_loss, prog_bar=True)
        pl_module.log("train/acc", epoch_mean_acc, prog_bar=True)

        pl_module.training_step_outputs.clear()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        epoch_mean_loss = torch.stack([x['val_loss'] for x in pl_module.validation_step_outputs]).mean()
        epoch_mean_acc = torch.stack([x['val_acc'] for x in pl_module.validation_step_outputs]).mean()

        pl_module.log('val/loss', epoch_mean_loss, prog_bar=True)
        pl_module.log('val/acc', epoch_mean_acc, prog_bar=True)

        pl_module.validation_step_outputs.clear()
```

#### 2.4 调用

当我们都写完了上述我们定义好的数据模块，训练预测模块，那么如何使用呢？pytorch_lightning这里用了一个专门的类Trainer来调用。

**训练调用**：

```python
trainer = Trainer(max_epochs=config.num_epochs,
                      # resume_from_checkpoint = 'ckpts/exp3/epoch=7.ckpt', # 断点续训
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=True,  # 显示模型构造
                      accelerator='auto',
                      devices=1,  # 多少个设备
                      deterministic=True,
                      num_sanity_val_steps=1,  # 正式训练之前跑一次validation 测试程序是否出错
                      benchmark=True,  # cudnn加速训练（要确保每个batch同一个大小）
                      )
    # mnist_model.load_from_checkpoint('ckpts/exp3/epoch=7.ckpt')
    trainer.fit(mnist_model,mnist_data)
```

**预测调用**，可以定义一个dataloader，也可以定义测试的数据模块，同时也能直接对单一一个tensor作为输入，进行预测：

```python
rs = trainer.predict(mnist_model, dataloaders=test_loader)
rs = trainer.predict(mnist_model, datamodule=test_datamodule)
```

### 三. 分布式训练

pytorch_lightning也支持分布式，但是它只支持pytorch原生的DDP，作为被HuggingFace的accelerate圈粉的我。。。只能退坑了，拜拜👋🏻
