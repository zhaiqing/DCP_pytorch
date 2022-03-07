# DCP_pytorch
Deep Closest Point</br>
新增vis文件，可以使配准前后的点云可视化</br>
设备：英伟达1050ti,4GB,win10</br>
# 训练记录：
训练的时候batch_size 只能设置为2，一个epoch训练的时间为50分钟</br>
由于又transformer的存在，网络结构较大</br>
有tensorboard，可以查看实验数据</br>
训练时间过长</br>
通过比较test_loss，让模型文件中有一个model.best.t7,用于记录最好的一次训练的参数</br>
训练不仅求出从a->b的旋转R和平移t，还求出了从b->a的旋转R和平移t（可借鉴）</br>
通过SummaryWriter记录日志信息，可以使用tensorboard访问得到实验结果图</br>
通过IOStream 来控制log文件输出与控制台信息输出</br>
保存模型文件:</br>
```
torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
```

