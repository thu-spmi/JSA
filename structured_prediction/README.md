Codes for reproducing experiments of  Table 3.

```
sh experiments/run.sh [method] [n] [seed]
```

method is one of ["jsa","rws","vimco"]

n is the particles number, int

seed is the random seed for independent trials, int.

For example: 

sh  experiments/run.sh jsa 5 1

will get result of one trial of JSA with "n=5" in Table 3. 



 