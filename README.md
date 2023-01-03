## Pruning[NISP'15] paper code reproduction

- Paper：[[1506.02626v3] Learning both Weights and Connections for Efficient Neural Networks (arxiv.org)](https://arxiv.org/abs/1506.02626v3)
- Experimental Reproduction ideas: [Pruning[NISP'15] paper code reproduction | luser's study site (luserli.github.io)](https://luserli.github.io/post/pruningnisp15lun-wen-fu-xian/)

### File Directory

- `datasets`:
  - `carvnocar.h5`: 用于网络训练参数的数据集
  - `parameters.npy`: 包含有网络训练好之后保存的参数字典数据文件
  - `data.npy`: 包含网络训练时间，训练过程中的cost值等
  - `degree_costs.npy`: 包含不同pruning度下迭代到收敛时的cost值
  - `prun_parameter`: 包含不同pruning度下训练到cost收敛时的参数
- `figure`: 一些实验中的效果图
- `photos_demo`: 用于测试网络判断效果的测试图片
- `nn_main.py`: 网络训练时的主文件
- `nn_functions.py`: 实现神经网络的主要函数
- `nn_test.py`: 用于测试训练好的神经网络
- `prun_mask.py`: 生成mask矩阵的实验
- `pruning.py`: 对原始参数进行剪枝和再训练，也是这里的主要文件
- `prun_parameters.py`: 对裁剪后的参数进行测试

## Experimental record

不同剪枝度得到的训练至收敛后的cost：

- ![](https://s2.loli.net/2023/01/03/tgQWHFR1n8IKXZM.png)




使用第**三十九次**剪枝后重训练得到的参数与原参数进行比对、判断：

```
Ori parameter:
Parameter pruning degree:  0.0 %
Training set accuracy： 98.89502762430939 %
Test set accuracy： 80.0 %
Pruned parameter:
Parameter pruning degree:  86.406 %
Training set accuracy： 99.4475138121547 %
Test set accuracy： 80.0 %
```

No pruning 的原`W1`参数：

```
[[ 0.17652197 -0.06106679 -0.07412075 ... -0.14664018 -0.00927934
   0.06886687]
 [ 0.11185143 -0.06578853 -0.0011025  ...  0.08918518  0.07352842
  -0.00663041]
 [-0.10772029  0.03944582 -0.24708339 ...  0.0550951  -0.03051575
  -0.06339629]
 [-0.05884084  0.20572945  0.03835234 ...  0.16899935  0.02967805
   0.07047436]
 [ 0.04006457 -0.03186718  0.00984735 ...  0.01321126 -0.09708557
   0.21907507]]
```

`86.406%`pruning后的`W1`参数：

```
[[ 0.18168446  0.          0.         ...  0.          0.
   0.        ]
 [ 0.          0.          0.         ...  0.          0.
   0.        ]
 [-0.         -0.         -0.25039574 ... -0.         -0.
  -0.        ]
 [ 0.          0.20658655  0.         ...  0.16506836  0.
   0.        ]
 [-0.         -0.         -0.         ... -0.         -0.
   0.23942507]]
```

原参数对测试图片进行预测：

![](https://s2.loli.net/2023/01/03/4YToAjPKh5wO9kH.png)

​																													**26/27**

剪枝后[`86.406%`]：

![](https://s2.loli.net/2023/01/03/UQH7gkdM3Ronm5X.png)

​																													**25/27**
