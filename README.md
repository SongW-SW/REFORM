# REFORM

This is the code for the paper REFORM: Error-Aware Few-Shot Knowledge Graph Completion, CIKM 2021. [PDF file](https://songw-sw.github.io/REFORM.pdf)  
The citation is currently unavailable and we will update this page when the paper is finally published.
![Alt text](./Framework.png)

These are the statistics of the datasets:

<img src="./Dataset.png" alt="Editor" width="500">



To run the code, type the following commands:  

tar zxvf FB_data.tar.gz  
python train.py  

For other datasets (NELL and Wiki), I put the dataset in the following link:
https://drive.google.com/file/d/1GrOiNybt9q_pFrQo7r0XBsbTNo4hVMT_/view?usp=sharing

NELL datasets have been processed, while Wiki dataset somehow only has original data. 

To process your own dataset or change the noise rate, use the python file named data.py to produce the dataset and train_transe.py to train the TransE embedding. Then use train.py to train the model on new datasets.
