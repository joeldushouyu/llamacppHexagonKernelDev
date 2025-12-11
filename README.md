## This is a test module for running GGML operations on Hexagon DSP


## Setup Instructions

1. Clone the repo into the same directory with the llama.cpp folder
The directory structure should look like this:
```
.
├── llama.cpp
└── ggmlHexagonModule
```

2. Build the module
First, navigate to the hexagonSDK docker container as [instructed here](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/hexagon/README.md) and then run: 
```bash
ubuntu@9fe9ecd0cac1:/workspace/hexagonDevep/ggmlHexagonModule$ ./build_test.sh 
```
3. Deply and run the module

```bash
## Run in host machine terminal, not in the docker container
./deploy_and_test.sh 
```



### Some helpful utilities

1. Plot GELU/SILU functions
   * after saving the gelu output to csv file, you can run the following script to plot the result
   ```bash
    ~/hexagonExplore/hexagonDevep/ggmlHexagonModule/python_util/plot_csv_data$ ./plotGraph.sh gelu_approx.csv 
   ```