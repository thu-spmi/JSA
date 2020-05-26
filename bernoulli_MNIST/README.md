Codes for reproducing experiments of  Table 1.

```
sh experiments/run.sh [method] [net] [seed]
```

method is one of ["jsa","rws","vimco"]

net is one of ["linear1","linear2","nonlinear"]

seed is the random seed for independent trials, int.

For example: 

sh  experiments/run.sh jsa linear1 1

will get result of one trial of JSA with "Linear" architecture in Table 1. 



 