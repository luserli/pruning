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
