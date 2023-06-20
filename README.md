# Complex Eventuality Query Answering (CEQA)
Official Implementation of paper: [Complex Query Answering on Eventuality Knowledge Graph with Implicit Logical Constraints](https://arxiv.org/abs/2305.19068). If you would like to build your own query data, you can use the following instructions for data sampling. Note that you may need to adjust the directories in the Python scripts.


## Data Sampling
The data we used in this paper is subsampled from [ASER2.1](https://hkust-knowcomp.github.io/ASER/html/index.html), and it is called ASER-50k. The ASER-50K graph is available in the data file. All sampled & filtered data can be downloaded [here](https://drive.google.com/file/d/11UJCcLeGwS6vfnnnD8zb5hm1aNlmS3jP/view?usp=sharing), and you do not have to run the sampling code by yourself. 

### Query Sampling (without informational atomics)
In the first step, we sample the complex queries (without informational atomics) from the ASER-50K by running the file:

```
python sample_all_types_aser_train.py
python sample_all_types_aser_validation.py
python sample_all_types_aser_test.py
```
After this step, you will get many query files in the directory ``` ./query_data ``` (we provide a small data in the same format in ```./query_data_dev``` for debugging and unit testing)

### Sample Informational Atomics & Theorem Proving

These two steps are combined together, we sample some informational atomics for each query and then use the theorem provers to filter out the answers that are contradictory. 

```
python filter_explanation.py
```

After this step, you will get the data files used for training and evaluation in the directory ``` ./query_data_filtered ``` (Similarly, we provide a small data in the same format in ```./query_data_dev_filtered``` for debugging and unit testing)


### Query Encoding with Memory Constraints

We implement all the models in ```./model```. In each model file, we have their original implementation and the inherited memory-enhanced version. For running the experiments, simply by using:

```
./run_gqe.sh
./run_gqe_con.sh
```
The ```_con.sh``` are constraint memory-enhanced models. You can replace ```gqe``` with ```q2p```, ```mlp```, and ```fuzzqe``` respectively. 
To monitor the training and evaluation process, you can use the tensorboard at log dir ```../logs/gradient_tape```


### Citation
If you find the paper/data/code interesting, please cite our work:
```
@article{bai2023complex,
  title={Complex Query Answering on Eventuality Knowledge Graph with Implicit Logical Constraints},
  author={Bai, Jiaxin and Liu, Xin and Wang, Weiqi and Luo, Chen and Song, Yangqiu},
  journal={arXiv preprint arXiv:2305.19068},
  year={2023}
}
```


