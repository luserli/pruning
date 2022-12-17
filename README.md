# Pruning[NISP'15]论文复现

- 论文地址：[[1506.02626v3\] Learning both Weights and Connections for Efficient Neural Networks (arxiv.org)](https://arxiv.org/abs/1506.02626v3)



## Pseudo-code

| $Pruning\ Deep\ Neural\ Networks$                            |
| :----------------------------------------------------------- |
| $Initialization:W^{(0)}\ with\ W^{(0)}\sim N(0, sum),iter=0.$                       <br />$Hyper-parameter:\ threshold, \delta$                                                              <br />$Output:W(t).$ |
| $Train\ Connectivity$                                        |
| ***while***$\ not\ converged\ $***do***                                                                                    <br />        $W^{(t)}=W^{(t-1)}-\eta^{t}\bigtriangledown f(W^{(t-1)};x^{(t-1)});$                                             <br />        $t=t+1;$                                                                                                     <br />***end*** |
| $Prune\ Connections$                                         |
| $//\ initialize\ the\ mask\ by\ thresholding\ the\ weights.$                            <br />$Mask=1(|W|>threshold);$                                                                       <br />$W=W\cdot Mask;$ |
| $Retrain\ Weights$                                           |
| ***while*** $not\ converged$ ***do***                                                                                    <br />        $W^{(t)}=W^{(t-1)}-\eta^{t}\bigtriangledown f(W^{(t-1)};x^{(t-1)});$                                            <br />        $W^{(t)}=W^{(t)}\cdot Mask;$                                                                                <br />        $t=t+1;$                                                                                                      <br />***end*** |
| $Iterative\ Pruning$                                         |
| $threshold=threshold+\delta [iter++];$                                                          <br />***goto*** $Prune\ Connections;$ |

```
Parameter pruning degree:  33.0 %

Pruning parameters:
Training set accuracy： 59.11602209944751 %
Test set accuracy： 61.666666666666664 %
Retrain 199: 100%|█████████████████████████████████████████| 200/200 [00:06<00:00, 31.85it/s, cost=0.13132939097585444]

Pruning and retrain parameters:
Training set accuracy： 96.68508287292818 %
Test set accuracy： 80.0 %

Parameter pruning degree:  34.5 %

Pruning parameters:
Training set accuracy： 59.11602209944751 %
Test set accuracy： 61.666666666666664 %
Retrain 199: 100%|██████████████████████████████████████████| 200/200 [00:05<00:00, 33.62it/s, cost=0.7660788360590391]

Pruning and retrain parameters:
Training set accuracy： 42.5414364640884 %
Test set accuracy： 44.99999999999999 %

Parameter pruning degree:  36.0 %

Pruning parameters:
Training set accuracy： 59.11602209944751 %
Test set accuracy： 61.666666666666664 %
Retrain 199: 100%|██████████████████████████████████████████| 200/200 [00:05<00:00, 33.54it/s, cost=0.6739200190350145]

Pruning and retrain parameters:
Training set accuracy： 59.11602209944751 %
Test set accuracy： 61.666666666666664 %
```

```
Parameter pruning degree:  58 %

Pruning parameters:
Training set accuracy： 76.24309392265194 %
Test set accuracy： 71.66666666666667 %
Retrain 599: 100%|████████████████████████████████████████| 600/600 [00:19<00:00, 31.12it/s, cost=0.030149974760246834]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 81.66666666666667 %

Parameter pruning degree:  59 %

Pruning parameters:
Training set accuracy： 67.95580110497238 %
Test set accuracy： 68.33333333333334 %
Retrain 599: 100%|████████████████████████████████████████| 600/600 [00:18<00:00, 32.14it/s, cost=0.030070526749287087]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 81.66666666666667 %

Parameter pruning degree:  60 %

Pruning parameters:
Training set accuracy： 67.40331491712706 %
Test set accuracy： 68.33333333333334 %
Retrain 599: 100%|█████████████████████████████████████████| 600/600 [00:18<00:00, 32.82it/s, cost=0.01819574258059817]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 81.66666666666667 %
```

```
Parameter pruning degree:  98 %

Pruning parameters:
Training set accuracy： 67.95580110497238 %
Test set accuracy： 68.33333333333334 %
Retrain 199: 100%|█████████████████████████████████████████| 200/200 [00:06<00:00, 32.44it/s, cost=0.04642880078345218]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 80.0 %

Parameter pruning degree:  99 %

Pruning parameters:
Training set accuracy： 70.1657458563536 %
Test set accuracy： 68.33333333333334 %
Retrain 199: 100%|████████████████████████████████████████| 200/200 [00:06<00:00, 32.74it/s, cost=0.036446431663070065]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 81.66666666666667 %
```

```
Parameter pruning degree:  1.55 %

Pruning parameters:
Training set accuracy： 59.668508287292816 %
Test set accuracy： 61.666666666666664 %
Retrain 499: 100%|████████████████████████████████████████| 500/500 [00:20<00:00, 24.74it/s, cost=0.026854588414961026]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 78.33333333333333 %

Parameter pruning degree:  1.91 %

Pruning parameters:
Training set accuracy： 59.11602209944751 %
Test set accuracy： 61.666666666666664 %
Retrain 499: 100%|████████████████████████████████████████| 500/500 [00:24<00:00, 20.70it/s, cost=0.008733250846651884]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 78.33333333333333 %

Parameter pruning degree:  2.34 %

Pruning parameters:
Training set accuracy： 71.8232044198895 %
Test set accuracy： 75.0 %
Retrain 499: 100%|█████████████████████████████████████████| 500/500 [00:24<00:00, 20.72it/s, cost=0.01599788979202738]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 78.33333333333333 %

Parameter pruning degree:  2.85 %

Pruning parameters:
Training set accuracy： 97.79005524861878 %
Test set accuracy： 78.33333333333333 %
Retrain 499: 100%|██████████████████████████████████████████| 500/500 [00:21<00:00, 23.62it/s, cost=0.6614138746422282]

Pruning and retrain parameters:
Training set accuracy： 60.773480662983424 %
Test set accuracy： 61.666666666666664 %

Parameter pruning degree:  3.45 %

Pruning parameters:
Training set accuracy： 60.773480662983424 %
Test set accuracy： 61.666666666666664 %
Retrain 499: 100%|████████████████████████████████████████| 500/500 [00:17<00:00, 28.30it/s, cost=0.020095431141667525]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 78.33333333333333 %

Parameter pruning degree:  4.15 %

Pruning parameters:
Training set accuracy： 98.34254143646409 %
Test set accuracy： 78.33333333333333 %
Retrain 499: 100%|████████████████████████████████████████| 500/500 [00:19<00:00, 25.99it/s, cost=0.031202759521449203]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 78.33333333333333 %

Parameter pruning degree:  4.96 %

Pruning parameters:
Training set accuracy： 86.74033149171271 %
Test set accuracy： 70.0 %
Retrain 499: 100%|████████████████████████████████████████| 500/500 [00:23<00:00, 21.01it/s, cost=0.010311013676648634]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 76.66666666666667 %

Parameter pruning degree:  5.89 %

Pruning parameters:
Training set accuracy： 89.50276243093923 %
Test set accuracy： 71.66666666666667 %
Retrain 499: 100%|████████████████████████████████████████| 500/500 [00:24<00:00, 20.71it/s, cost=0.019787374051867593]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 80.0 %

Parameter pruning degree:  6.95 %

Pruning parameters:
Training set accuracy： 95.02762430939227 %
Test set accuracy： 71.66666666666667 %
Retrain 499: 100%|██████████████████████████████████████████| 500/500 [00:22<00:00, 22.70it/s, cost=0.6764234833695614]

Pruning and retrain parameters:
Training set accuracy： 59.11602209944751 %
Test set accuracy： 61.666666666666664 %

Parameter pruning degree:  8.15 %

Pruning parameters:
Training set accuracy： 59.11602209944751 %
Test set accuracy： 61.666666666666664 %
Retrain 499: 100%|██████████████████████████████████████████| 500/500 [00:17<00:00, 28.37it/s, cost=0.6614599561567005]

Pruning and retrain parameters:
Training set accuracy： 60.773480662983424 %
Test set accuracy： 63.333333333333336 %

Parameter pruning degree:  9.5 %

Pruning parameters:
Training set accuracy： 62.430939226519335 %
Test set accuracy： 61.666666666666664 %
Retrain 499: 100%|████████████████████████████████████████| 500/500 [00:18<00:00, 27.05it/s, cost=0.016002192579504432]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 78.33333333333333 %

Parameter pruning degree:  11.01 %

Pruning parameters:
Training set accuracy： 65.19337016574585 %
Test set accuracy： 68.33333333333334 %
Retrain 499: 100%|██████████████████████████████████████████| 500/500 [00:23<00:00, 21.61it/s, cost=0.6664570819825356]

Pruning and retrain parameters:
Training set accuracy： 60.22099447513812 %
Test set accuracy： 61.666666666666664 %

Parameter pruning degree:  12.69 %

Pruning parameters:
Training set accuracy： 60.773480662983424 %
Test set accuracy： 63.333333333333336 %
Retrain 499: 100%|█████████████████████████████████████████| 500/500 [00:24<00:00, 20.60it/s, cost=0.03480579345056958]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 76.66666666666667 %
```

```
Parameter pruning degree:  0.0 %

Parameter pruning degree:  1.74 %
```

```
Parameter pruning degree:  0.35 %

Pruning parameters:
Training set accuracy： 96.68508287292818 %
Test set accuracy： 75.0 %
Retrain 499: 100%|█████████████████████████████████████████| 500/500 [00:16<00:00, 30.50it/s, cost=0.07457813256685683]

Pruning and retrain parameters:
Training set accuracy： 96.68508287292818 %
Test set accuracy： 78.33333333333333 %

Parameter pruning degree:  0.5 %

Pruning parameters:
Training set accuracy： 40.88397790055248 %
Test set accuracy： 38.33333333333333 %
Retrain 499: 100%|██████████████████████████████████████████| 500/500 [00:15<00:00, 31.85it/s, cost=0.6764234833695615]

Pruning and retrain parameters:
Training set accuracy： 59.11602209944751 %
Test set accuracy： 61.666666666666664 %
Traceback (most recent call last):
  File "D:\Code\Python\test\neural_network\NN\pruning\pruning.py", line 101, in <module>
    threshold=np.percentile(b, h_threshold)
  File "<__array_function__ internals>", line 180, in percentile
  File "D:\Python\Program\lib\site-packages\numpy\lib\function_base.py", line 4134, in percentile
    return _quantile_unchecked(
  File "D:\Python\Program\lib\site-packages\numpy\lib\function_base.py", line 4383, in _quantile_unchecked
    r, k = _ureduce(a,
  File "D:\Python\Program\lib\site-packages\numpy\lib\function_base.py", line 3702, in _ureduce
    r = func(a, **kwargs)
  File "D:\Python\Program\lib\site-packages\numpy\lib\function_base.py", line 4552, in _quantile_ureduce_func
    result = _quantile(arr,
  File "D:\Python\Program\lib\site-packages\numpy\lib\function_base.py", line 4658, in _quantile
    take(arr, indices=-1, axis=DATA_AXIS)
  File "<__array_function__ internals>", line 180, in take
  File "D:\Python\Program\lib\site-packages\numpy\core\fromnumeric.py", line 190, in take
    return _wrapfunc(a, 'take', indices, axis=axis, out=out, mode=mode)
  File "D:\Python\Program\lib\site-packages\numpy\core\fromnumeric.py", line 57, in _wrapfunc
    return bound(*args, **kwds)
IndexError: cannot do a non-empty take from an empty axes.
```

```
Parameter pruning degree:  38.04 %

Pruning parameters:
Training set accuracy： 59.11602209944751 %
Test set accuracy： 61.666666666666664 %
Retrain 499: 100%|████████████████████████████████████████| 500/500 [00:17<00:00, 29.01it/s, cost=0.015577048681101699]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 76.66666666666667 %

Parameter pruning degree:  42.15 %

Pruning parameters:
Training set accuracy： 59.11602209944751 %
Test set accuracy： 61.666666666666664 %
Retrain 1999: 100%|█████████████████████████████████████| 2000/2000 [01:02<00:00, 31.83it/s, cost=0.012949745637572683]

Pruning and retrain parameters:
Training set accuracy： 99.4475138121547 %
Test set accuracy： 80.0 %
```

