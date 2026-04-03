
---
## Usage

- **CIFAR Experiments (`exp/`)**: Pruning redundant samples on CIFAR-10 and CIFAR-100  
- **Tiny-ImageNet Experiments (`exp/`)**: Pruning noisy samples with 20% label noise  

---

## Pruning Redundant Samples on CIFAR

### Step 1: Collect Training Dynamics

Train the model on the full dataset to collect per-sample outputs for scoring.  
ESPS only requires early-stage training dynamics (first 10 epochs), making it computationally efficient.

```
python train.py \
    --data_path ./data \
    --dataset cifar10 \
    --arch resnet18 \
    --epochs 200 \
    --learning-rate 0.1 \
    --batch-size 128 \
    --manualSeed 42 \
    --dynamics \
    --save_path ./checkpoint
```

The collected training dynamics will be saved to the path specified by `--save_path`.

---

## Step 2: Evaluate Sample Importance (Score Computation)

After collecting training dynamics, an importance score is computed for each sample using various data pruning methods, including EL2N, Dyn-Unc, TDDS, and DUAL.

At this stage, ESPS is recommended, as it more efficiently removes redundant samples.

Run the following command, specifying the path to your saved dynamics and where you want to store the results.

```
python importance_evaluation.py \
    --dataset cifar10 \
    --dynamics_path ./checkpoint/cifar10/42/npy/ \
    --save_path ./checkpoint/cifar10/42/generated_mask/
```
This command generates two .npy files for each method:

- `XXX_score.npy`: Contains the importance score for each data point, ordered by original sample index.

- `XXX_mask.npy`: Contains the sorted sample indexes based on their importance scores.


---

## Step3: Train Classifiers on the Pruned Dataset

Now you can train a model using the pruned dataset that you created in the previous step. The `--subset_rate` parameter determines the percentage of data to keep. For example, a value of 0.1 keeps 10% of the dataset.
The `--sample balance` option enables the Balanced Sampling Strategy.

Use the following command, making sure to update the file paths (`--score-path`, `--mask-path`, and `--target-probs-path`) to the files generated in Step 2.

```
python train_subset.py \
    --data_path ./data \
    --dataset cifar10 \
    --arch resnet18 \
    --epochs 200 \
    --learning_rate 0.1 \
    --batch-size 32 \
    --save_path ./checkpoint/pruned-dataset/cifar10/42 \
    --subset_rate 0.1  \
    --target-probs-path ./checkpoint/cifar10/42/generated_mask/target_probs.npy \
    --score-path ./checkpoint/cifar10/42/generated_mask/esps_10_mask.npy \
    --mask-path ./checkpoint/cifar10/42/generated_mask/esps_10_mask.npy \
    --keep high \
    --sample balance
```

For high pruning ratios, using a smaller batch size generally leads to better performance.  
In our experiments, we use a batch size of 32 when the pruning ratio exceeds 70%, and 128 otherwise.  
Please refer to the experimental settings section in our paper for more details.


---
## Pruning Noise on Tiny-ImageNet

## Step 1: Collect Training Dynamics

Train the model on the full dataset to collect per-sample outputs for scoring.  
ESPS-N only uses early-stage training dynamics (first 30 epochs).


```
python train.py \
    --data_path ./data \
    --dataset tiny-imagenet \
    --arch resnet34 \
    --epochs 90 \
    --learning-rate 0.1 \
    --batch-size 256 \
    --manualSeed 42 \
    --dynamics \
    --save_path ./checkpoint
```

The collected training dynamics will be saved to the path specified by `--save_path`.

---

## Step 2: Evaluate Sample Importance (Score Computation)

After collecting training dynamics, you can compute an importance score for each sample using various data pruning methods, including EL2N, Dyn-Unc, TDDS, and DUAL.  

At this stage, ESPS-N is recommended, as it more effectively removes noisy samples.

Run the following command, specifying the path to your saved dynamics and where you want to store the results.

```
python importance_evaluation.py \
    --dataset tiny-imagenet \
    --dynamics_path ./checkpoint/tiny-imagenet/42/npy/ \
    --save_path ./checkpoint/tiny-imagenet/42/generated_mask/
```
This command generates two .npy files for each method:

- `XXX_score.npy`: Contains the importance score for each data point, ordered by original sample index.

- `XXX_mask.npy`: Contains the sorted sample indexes based on their importance scores.


---

## Step3: Train Classifiers on the Pruned Dataset

Now you can train a model using the pruned dataset that you created in the previous step. The `--subset_rate` parameter determines the percentage of data to keep. For example, a value of 0.3 keeps 30% of the dataset.

Use the following command, making sure to update the file paths (`--score-path`, `--mask-path`, and `--target-probs-path`) to the files generated in Step 2.


```
python train_subset.py \
    --data_path ./data \
    --dataset tiny-imagenet \
    --arch resnet34 \
    --epochs 90 \
    --learning_rate 0.1 \
    --batch-size 256 \
    --save_path ./checkpoint/pruned-dataset/tiny-imagenet/42 \
    --subset_rate 0.3  \
    --target-probs-path ./checkpoint/tiny-imagenet/42/generated_mask/target_probs.npy \
    --score-path ./checkpoint/tiny-imagenet/42/generated_mask/esps_n_30_mask.npy \
    --mask-path ./checkpoint/tiny-imagenet/42/generated_mask/esps_n_30_mask.npy \
    --keep low \
```

For noise dataset we add Tiny_imageNet for 20% label noise on train data.  
Please refer to the experimental settings section in our paper for more details.


---
### Attribution

This code is mostly build upon 
```bibtex
@misc{cho2025lightweightdatasetpruningtraining,
      title={Lightweight Dataset Pruning without Full Training via Example Difficulty and Prediction Uncertainty}, 
      author={Yeseul Cho and Baekrok Shin and Changmin Kang and Chulhee Yun},
      year={2025},
      eprint={2502.06905},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.06905}, 
}
```
