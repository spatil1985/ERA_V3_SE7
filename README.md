# Model 1
### Targets: 
 - Start with the model given in colab from the session 7 class.
### Results: 
- Parameters: 13.8k
- Best Train Accuracy: 98.95
- Best Test Accuracy: 99.41 (15th Epoch)
### Analysis: 
 - The model is a simple CNN with 7 layers. It uses dropout to prevent overfitting. The model is able to achieve high accuracy on the test set, but it is not able to achieve the accuracy of 99.41% with less than 8K parameters.
 - The model is able to achieve 99.41% accuracy on the test set with 13.8k parameters.
 - The model is able to achieve 98.95% accuracy on the train set with 13.8k parameters.


## Code for Model 1

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

dropout_value = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value)
        )

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
```

## Console Log for Model 1

```
PS E:\AI\github\era-v3-s7-cnn-cloud> python src/train.py
cpu
----------------------------------------------------------------
        Layer (type)               Output Shape         Param # 
================================================================
            Conv2d-1           [-1, 16, 26, 26]             144 
              ReLU-2           [-1, 16, 26, 26]               0 
       BatchNorm2d-3           [-1, 16, 26, 26]              32 
           Dropout-4           [-1, 16, 26, 26]               0 
            Conv2d-5           [-1, 32, 24, 24]           4,608 
              ReLU-6           [-1, 32, 24, 24]               0 
       BatchNorm2d-7           [-1, 32, 24, 24]              64 
           Dropout-8           [-1, 32, 24, 24]               0 
            Conv2d-9           [-1, 10, 24, 24]             320 
        MaxPool2d-10           [-1, 10, 12, 12]               0 
           Conv2d-11           [-1, 16, 10, 10]           1,440 
             ReLU-12           [-1, 16, 10, 10]               0
      BatchNorm2d-13           [-1, 16, 10, 10]              32
          Dropout-14           [-1, 16, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           2,304
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
          Dropout-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
          Dropout-22             [-1, 16, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           2,304
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 13,808
Trainable params: 13,808
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.06
Params size (MB): 0.05
Estimated Total Size (MB): 1.12
----------------------------------------------------------------
CUDA Available? False
EPOCH: 0
Loss=0.01941436156630516 Batch_id=937 Accuracy=91.28: 100%|█████████████████████████████████████████████████████████████████████████████| 938/938 [02:13<00:00,  7.01it/s] 

Test set: Average loss: 0.0572, Accuracy: 9821/10000 (98.21%)

EPOCH: 1
Loss=0.03962118551135063 Batch_id=937 Accuracy=97.69: 100%|█████████████████████████████████████████████████████████████████████████████| 938/938 [02:07<00:00,  7.38it/s] 

Test set: Average loss: 0.0518, Accuracy: 9824/10000 (98.24%)

EPOCH: 2
Loss=0.005107395816594362 Batch_id=937 Accuracy=98.19: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:04<00:00,  7.52it/s] 

Test set: Average loss: 0.0303, Accuracy: 9902/10000 (99.02%)

EPOCH: 3
Loss=0.048854272812604904 Batch_id=937 Accuracy=98.31: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:00<00:00,  7.81it/s] 

Test set: Average loss: 0.0272, Accuracy: 9913/10000 (99.13%)

EPOCH: 4
Loss=0.002012245124205947 Batch_id=937 Accuracy=98.56: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:00<00:00,  7.81it/s] 

Test set: Average loss: 0.0279, Accuracy: 9912/10000 (99.12%)

EPOCH: 5
Loss=0.005592297296971083 Batch_id=937 Accuracy=98.64: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:01<00:00,  7.70it/s] 

Test set: Average loss: 0.0233, Accuracy: 9920/10000 (99.20%)

EPOCH: 6
Loss=0.021483028307557106 Batch_id=937 Accuracy=98.64: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:01<00:00,  7.73it/s] 

Test set: Average loss: 0.0233, Accuracy: 9924/10000 (99.24%)

EPOCH: 7
Loss=0.026516681537032127 Batch_id=937 Accuracy=98.74: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:03<00:00,  7.59it/s] 

Test set: Average loss: 0.0229, Accuracy: 9924/10000 (99.24%)

EPOCH: 8
Loss=0.022202573716640472 Batch_id=937 Accuracy=98.75: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:07<00:00,  7.33it/s] 

Test set: Average loss: 0.0219, Accuracy: 9934/10000 (99.34%)

EPOCH: 9
Loss=0.02282341569662094 Batch_id=937 Accuracy=98.89: 100%|█████████████████████████████████████████████████████████████████████████████| 938/938 [02:03<00:00,  7.57it/s] 

Test set: Average loss: 0.0238, Accuracy: 9923/10000 (99.23%)

EPOCH: 10
Loss=0.014295128174126148 Batch_id=937 Accuracy=98.89: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:02<00:00,  7.63it/s] 

Test set: Average loss: 0.0220, Accuracy: 9926/10000 (99.26%)

EPOCH: 11
Loss=0.0047846161760389805 Batch_id=937 Accuracy=98.88: 100%|███████████████████████████████████████████████████████████████████████████| 938/938 [02:01<00:00,  7.73it/s] 

Test set: Average loss: 0.0221, Accuracy: 9935/10000 (99.35%)

EPOCH: 12
Loss=0.009756717830896378 Batch_id=937 Accuracy=98.88: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:10<00:00,  7.17it/s] 

Test set: Average loss: 0.0207, Accuracy: 9937/10000 (99.37%)

EPOCH: 13
Loss=0.19465979933738708 Batch_id=937 Accuracy=98.95: 100%|█████████████████████████████████████████████████████████████████████████████| 938/938 [02:01<00:00,  7.71it/s] 
Test set: Average loss: 0.0197, Accuracy: 9938/10000 (99.38%)

EPOCH: 14
EPOCH: 14
Loss=0.052712198346853256 Batch_id=937 Accuracy=98.94: 100%|████████████████████████████████████████████████████████████████████████████| 938/938 [02:04<00:00,  7.54it/s] ████████| 938/938 [02:04<00:00,  7.54it/s]

Test set: Average loss: 0.0206, Accuracy: 9941/10000 (99.41%)


===================
Results:
Parameters: 13.8k
Best Train Accuracy: 98.95
Best Test Accuracy: 99.41 (15th Epoch)
===================
PS E:\AI\github\era-v3-s7-cnn-cloud>
```


# Model 2
### Targets: 
 - Improve the model to reduce the number of parameters less than 8K.
The new model introduces several changes to enhance efficiency, reduce complexity, and improve generalization. Here's a detailed analysis of the changes, their purposes, and benefits:

---

### **1. Number of Filters**
- **Old Model**:
  - Initial block: 16 filters.
  - Second block: 32 filters.
  - Transition block: Down to 10 filters.
  - Later layers: Consistently higher number of filters (16).
- **New Model**:
  - Initial block: Reduced to 8 filters.
  - Second block: Reduced to 16 filters.
  - Transition block: Reduced to 8 filters, then further processing with smaller numbers of filters (e.g., 12 filters in later blocks).
- **Reasoning**: The new model uses fewer filters in each layer, making it computationally lighter.
- **Benefits**:
  - **Reduced computational cost**: Smaller filters reduce memory and processing requirements.
  - **Simpler model**: Avoids overfitting by reducing overparameterization, especially important for small datasets.

---

### **2. Transition Block**
- **Old Model**:
  - Transition block reduced filters from 32 to 10 using a `1x1` convolution, followed by pooling.
- **New Model**:
  - Transition block reduces filters from 16 to 8, followed by pooling.
- **Reasoning**: The transition block now reduces to a smaller feature map size, ensuring a more gradual dimensionality reduction.
- **Benefits**:
  - Maintains a balance between feature richness and computational efficiency.
  - Prevents excessive information loss due to abrupt reductions.

---

### **3. Depth of Convolution Blocks**
- **Old Model**:
  - Four convolution blocks in the second stage, all with higher numbers of filters.
- **New Model**:
  - Three convolution blocks in the second stage, with fewer filters (12 filters in blocks 4, 5, and 6).
- **Reasoning**: The new model reduces the depth and number of filters, opting for efficiency.
- **Benefits**:
  - Reduces overfitting risk, especially for small datasets.
  - Computationally less expensive.

---

### **4. Kernel Sizes and GAP**
- **Old Model**:
  - Global Average Pooling (GAP) with a kernel size of `6` and a spatial dimension consistent with the larger feature maps.
- **New Model**:
  - GAP with a kernel size of `8`, matched to the smaller feature maps produced by the new architecture.
- **Reasoning**: Kernel size is adjusted to fit the reduced feature map dimensions.
- **Benefits**:
  - Ensures the GAP operation appropriately aggregates spatial features.
  - Aligns model architecture with computational resources.

---

### **5. Removal of the Final `Dropout` Layer**
- **Old Model**: Included a standalone `dropout` layer at the end.
- **New Model**: Relies only on dropout within convolution blocks.
- **Reasoning**: Standalone dropout at the end might not significantly impact regularization since the GAP already reduces dimensionality.
- **Benefits**:
  - Avoids redundancy and simplifies the architecture.
  - Focuses regularization within intermediate layers where overfitting risk is higher.

---

### **6. Adjustment of Feature Map Sizes**
- **Old Model**: Larger feature maps due to the higher number of filters in all stages.
- **New Model**: Reduced feature map sizes by limiting filters and using fewer blocks in later stages.
- **Reasoning**: Smaller feature maps conserve memory and processing time.
- **Benefits**:
  - Makes the model lighter and faster without sacrificing performance on datasets where simpler architectures suffice.
  - Helps focus on essential features while avoiding noise.

---

### **7. Overall Simplification**
- **Old Model**: Relatively complex with higher filters, deeper layers, and standalone dropout.
- **New Model**: Streamlined, with fewer filters, fewer blocks, and integrated regularization.
- **Reasoning**: Aligns with the principle of Occam’s Razor, ensuring the model is not overly complex for the given task.
- **Benefits**:
  - Easier to train, especially on resource-constrained systems.
  - Lower risk of overfitting on small datasets.
  - Reduced inference time and power consumption.

---

### **Summary of Benefits**
1. **Efficiency**: Fewer filters and blocks reduce computation time and resource requirements.
2. **Generalization**: Simpler architecture avoids overfitting, especially useful for small or moderately complex datasets.
3. **Alignment with Input Size**: Smaller feature maps and adjusted GAP improve processing consistency.
4. **Robust Regularization**: Dropout within intermediate layers ensures effective regularization without unnecessary complexity.

These changes make the new model more compact and efficient while maintaining sufficient capacity to handle moderately complex tasks. It is particularly suited for environments where computational resources or data are limited.

### Results: 
 - Parameters: 5.0k
 - Best Train Accuracy: 98.36   
 - Best Test Accuracy: 99.08 (14th Epoch)
### Analysis: 
 - The model is a simple CNN with 7 layers. It uses dropout to prevent overfitting. The model is able to achieve high accuracy on the test set, but it is not able to achieve the accuracy of 99.41% with less than 8K parameters.
 - The model is able to achieve 99.08% accuracy on the test set with 5.0k parameters.
 - The model is able to achieve 98.36% accuracy on the train set with 5.0k parameters.

## Code for Model 2

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

dropout_value = 0.1

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dropout_value)
        )

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(1, 1), padding=0, bias=False),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        )
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        )

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        )
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.gap(x)        
        x = self.convblock7(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

```

## Console Log for Model 2

```
PS E:\AI\github\era-v3-s7-cnn-cloud> python src/train.py
cpu
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 26, 26]              72
              ReLU-2            [-1, 8, 26, 26]               0
       BatchNorm2d-3            [-1, 8, 26, 26]              16
           Dropout-4            [-1, 8, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,152
              ReLU-6           [-1, 16, 24, 24]               0
       BatchNorm2d-7           [-1, 16, 24, 24]              32
           Dropout-8           [-1, 16, 24, 24]               0
            Conv2d-9            [-1, 8, 24, 24]             128
        MaxPool2d-10            [-1, 8, 12, 12]               0
           Conv2d-11           [-1, 12, 10, 10]             864
             ReLU-12           [-1, 12, 10, 10]               0
      BatchNorm2d-13           [-1, 12, 10, 10]              24
          Dropout-14           [-1, 12, 10, 10]               0
           Conv2d-15             [-1, 12, 8, 8]           1,296
             ReLU-16             [-1, 12, 8, 8]               0
      BatchNorm2d-17             [-1, 12, 8, 8]              24
          Dropout-18             [-1, 12, 8, 8]               0
           Conv2d-19             [-1, 12, 8, 8]           1,296
             ReLU-20             [-1, 12, 8, 8]               0
      BatchNorm2d-21             [-1, 12, 8, 8]              24
          Dropout-22             [-1, 12, 8, 8]               0
        AvgPool2d-23             [-1, 12, 1, 1]               0
           Conv2d-24             [-1, 10, 1, 1]             120
================================================================
Total params: 5,048
Trainable params: 5,048
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.57
Params size (MB): 0.02
Estimated Total Size (MB): 0.60
----------------------------------------------------------------
CUDA Available? False
EPOCH: 0
Loss=0.03700254112482071 Batch_id=937 Accuracy=84.31: 100%|███| 938/938 [01:24<00:00, 11.13it/s]        

Test set: Average loss: 0.0873, Accuracy: 9772/10000 (97.72%)

EPOCH: 1
Loss=0.03686046972870827 Batch_id=937 Accuracy=96.06: 100%|██████████| 938/938 [01:09<00:00, 13.43it/s]

Test set: Average loss: 0.0762, Accuracy: 9770/10000 (97.70%)

EPOCH: 2
Loss=0.11288551241159439 Batch_id=937 Accuracy=96.95: 100%|██████████| 938/938 [01:27<00:00, 10.72it/s] 

Test set: Average loss: 0.0499, Accuracy: 9843/10000 (98.43%)

EPOCH: 3
Loss=0.163455992937088 Batch_id=937 Accuracy=97.34: 100%|████████████| 938/938 [01:22<00:00, 11.43it/s] 

Test set: Average loss: 0.0430, Accuracy: 9865/10000 (98.65%)

EPOCH: 4
Loss=0.024510761722922325 Batch_id=937 Accuracy=97.69: 100%|█████████| 938/938 [01:24<00:00, 11.09it/s] 

Test set: Average loss: 0.0401, Accuracy: 9860/10000 (98.60%)

EPOCH: 5
Loss=0.052453331649303436 Batch_id=937 Accuracy=97.78: 100%|█████████| 938/938 [01:22<00:00, 11.33it/s] 

Test set: Average loss: 0.0388, Accuracy: 9865/10000 (98.65%)

EPOCH: 6
Loss=0.18465520441532135 Batch_id=937 Accuracy=97.91: 100%|██████████| 938/938 [01:23<00:00, 11.23it/s] 

Test set: Average loss: 0.0334, Accuracy: 9896/10000 (98.96%)

EPOCH: 7
Loss=0.36025211215019226 Batch_id=937 Accuracy=97.92: 100%|██████████| 938/938 [01:22<00:00, 11.42it/s] 

Test set: Average loss: 0.0317, Accuracy: 9903/10000 (99.03%)

EPOCH: 8
Loss=0.007759375497698784 Batch_id=937 Accuracy=98.13: 100%|█████████| 938/938 [01:21<00:00, 11.54it/s] 

Test set: Average loss: 0.0361, Accuracy: 9882/10000 (98.82%)

EPOCH: 9
Loss=0.011883535422384739 Batch_id=937 Accuracy=98.11: 100%|█████████| 938/938 [01:21<00:00, 11.50it/s] 

Test set: Average loss: 0.0336, Accuracy: 9881/10000 (98.81%)

EPOCH: 10
Loss=0.2088194191455841 Batch_id=937 Accuracy=98.21: 100%|███████████| 938/938 [01:22<00:00, 11.37it/s] 

Test set: Average loss: 0.0306, Accuracy: 9896/10000 (98.96%)

EPOCH: 11
Loss=0.01457920391112566 Batch_id=937 Accuracy=98.19: 100%|██████████| 938/938 [01:21<00:00, 11.52it/s] 

Test set: Average loss: 0.0288, Accuracy: 9907/10000 (99.07%)

EPOCH: 12
Loss=0.07502304762601852 Batch_id=937 Accuracy=98.20: 100%|██████████| 938/938 [01:26<00:00, 10.84it/s] 

Test set: Average loss: 0.0326, Accuracy: 9897/10000 (98.97%)

EPOCH: 13
Loss=0.007396047003567219 Batch_id=937 Accuracy=98.32: 100%|█████████| 938/938 [01:23<00:00, 11.19it/s] 

Test set: Average loss: 0.0274, Accuracy: 9908/10000 (99.08%)

EPOCH: 14
Loss=0.05654190108180046 Batch_id=937 Accuracy=98.36: 100%|██████████| 938/938 [01:24<00:00, 11.16it/s] 

Test set: Average loss: 0.0311, Accuracy: 9898/10000 (98.98%)


===================
Results:
Parameters: 5.0k
Best Train Accuracy: 98.36
Best Test Accuracy: 99.08 (14th Epoch)
===================
PS E:\AI\github\era-v3-s7-cnn-cloud> 
```


# Model 3
### Targets: 
 - The number of parameters less than 8K, but the accuracy is not should be higher than 99.41% in 15 epochs. The next goal is to achieve the target of 99.41% accuracy in less than 15 epochs.
The new model introduces several changes that enhance its performance, robustness, and efficiency. Here's a detailed breakdown of the changes and their purposes:

---

### **1. Dropout Value**
- **Old Model**: `dropout_value = 0.1`
- **New Model**: `dropout_value = 0.03`
- **Reasoning**: A lower dropout value reduces the amount of dropped neurons during training, allowing more information to flow through the network in each forward pass.
- **Benefit**: Reduces the risk of underfitting and may help in faster convergence while still providing regularization to prevent overfitting.

---

### **2. Activation Function**
- **Old Model**: `ReLU` (Rectified Linear Unit)
- **New Model**: `GELU` (Gaussian Error Linear Unit)
- **Reasoning**: GELU is smoother than ReLU and incorporates a probabilistic element, activating neurons in a range rather than a hard cutoff at 0.
- **Benefit**: Improved gradient flow and convergence, especially for deeper networks. It also helps in reducing sharp saturation regions that could hinder learning.

---

### **3. Number of Filters**
- **Changes**: The number of filters in most layers has been increased slightly (e.g., from 8→10, 16→14).
- **Reasoning**: Increasing the number of filters allows the network to learn more features at each layer.
- **Benefit**: Enhanced capacity to capture more complex features, improving accuracy for complex datasets.

---

### **4. New Skip Connections**
- **New Additions**: 
  - `self.skip1` and `self.skip2` layers introduce **skip connections**.
  - Skip connections add outputs from earlier layers (`x`) to later layers (`x4` and `x6`) after resizing them using interpolation.
- **Reasoning**: Skip connections (like in ResNets) allow gradients to flow back more effectively and mitigate vanishing gradient issues.
- **Benefit**: 
  - Encourages feature reuse, leading to more efficient training.
  - Improves performance in deeper networks.
  - Provides robustness against overfitting by stabilizing the optimization process.

---

### **5. Additional Convolution Blocks**
- **Changes**:
  - `convblock6` is a new block added in the second convolution stage.
- **Reasoning**: Adding more layers increases the model depth, allowing it to learn more hierarchical and fine-grained features.
- **Benefit**: Improves the ability to model complex patterns in the data.

---

### **6. Attention Mechanism**
- **New Feature**: Adaptive attention mechanism.
  - **Details**: `F.adaptive_avg_pool2d` computes a spatially global average and applies a `torch.sigmoid` to scale the features in `x7`.
  - **Operation**: The output `x7` is multiplied element-wise with the attention weights.
- **Reasoning**: Attention mechanisms allow the model to focus on the most important spatial regions of the feature map.
- **Benefit**:
  - Reduces noise and irrelevant information.
  - Improves performance by emphasizing key regions in the input data.

---

### **7. GAP and Kernel Size Reduction**
- **Old Model**: GAP kernel size = `8`, Conv filter input size = `12 channels`.
- **New Model**: GAP kernel size = `6`, Conv filter input size = `12 channels`.
- **Reasoning**: Smaller kernel sizes in GAP lead to reduced computational overhead and better adaptation to the smaller feature maps in the new model.
- **Benefit**: Optimizes computational efficiency without sacrificing the ability to generalize well.

---

### **8. Interpolation for Size Adjustment**
- **New Addition**: `F.interpolate` is used to resize earlier layer outputs to match the spatial dimensions of deeper layers before adding them (in skip connections).
- **Reasoning**: Ensures dimensional consistency when adding features from different layers.
- **Benefit**: Improves feature alignment and allows skip connections to work seamlessly in non-identical spatial dimensions.

---

### **9. Final Convolution Layer Changes**
- **Old Model**: `convblock7` for output.
- **New Model**: `convblock8` replaces it and is paired with attention.
- **Reasoning**: Splitting attention and output prediction into separate layers enhances modularity and specialization of layers.
- **Benefit**:
  - The attention mechanism selectively enhances important features.
  - Improves interpretability and accuracy.

---

### **Summary of Benefits**
1. **Improved Feature Learning**: Deeper layers, skip connections, and attention mechanisms increase the ability to learn complex features.
2. **Better Gradient Flow**: Skip connections stabilize training and prevent vanishing gradients.
3. **Robustness and Efficiency**: Lower dropout values and smoother activations (GELU) reduce the risk of overfitting while improving computational efficiency.
4. **Enhanced Interpretability**: Attention mechanisms allow the model to focus on significant features, making its decisions easier to interpret.
5. **Adaptability to Input Variability**: Interpolation ensures feature alignment across different scales, improving the network's flexibility.

These changes make the new model more powerful and resilient while maintaining efficiency, particularly for datasets with complex patterns.

### Results: 
- Parameters: 7.9k
- Best Train Accuracy: 98.68
- Best Test Accuracy: 99.44 (12th Epoch)
### Analysis: 
- The model is able to achieve 99.44% accuracy on the test set with 7.9k parameters.
- The model is able to achieve 98.68% accuracy on the train set with 7.9k parameters.


## Code for Model 3

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

dropout_value = 0.03

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value)
        )

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        )

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.GELU(),            
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        )
        
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.GELU(),            
            nn.BatchNorm2d(14),
            nn.Dropout(dropout_value)
        )

        # NEW CONVOLUTION BLOCK
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.GELU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        )

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1, bias=False),
            nn.GELU(),            
            nn.BatchNorm2d(12),
            nn.Dropout(dropout_value)
        )
        
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        )

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 

        self.skip1 = nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(1, 1), padding=0, bias=False)
        self.skip2 = nn.Conv2d(in_channels=14, out_channels=12, kernel_size=(1, 1), padding=0, bias=False)

    def forward(self, x):
        x1 = self.convblock1(x)
        x2 = self.convblock2(x1)
        x3 = self.convblock3(x2)
        x = self.pool1(x3)
        
        x4 = self.convblock4(x)
        x4 = x4 + self.skip1(F.interpolate(x, size=(x4.shape[2], x4.shape[3])))
        
        x5 = self.convblock5(x4)
        x6 = self.convblock6(x5)
        x6 = x6 + self.skip2(F.interpolate(x4, size=x6.shape[2:]))
        
        x7 = self.convblock7(x6)
        
        b, c, h, w = x7.shape
        att = F.adaptive_avg_pool2d(x7, 1)
        att = torch.sigmoid(att)
        x7 = x7 * att
        
        x = self.gap(x7)        
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


```

## Console Log for Model 3

```
PS E:\AI\github\era-v3-s7-cnn-cloud> python src/train.py
cpu
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
              GELU-2           [-1, 10, 26, 26]               0
       BatchNorm2d-3           [-1, 10, 26, 26]              20
           Dropout-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 14, 24, 24]           1,260
              GELU-6           [-1, 14, 24, 24]               0
       BatchNorm2d-7           [-1, 14, 24, 24]              28
           Dropout-8           [-1, 14, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             140
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 14, 10, 10]           1,260
             GELU-12           [-1, 14, 10, 10]               0
      BatchNorm2d-13           [-1, 14, 10, 10]              28
          Dropout-14           [-1, 14, 10, 10]               0
           Conv2d-15           [-1, 14, 10, 10]             140
           Conv2d-16             [-1, 14, 8, 8]           1,764
             GELU-17             [-1, 14, 8, 8]               0
      BatchNorm2d-18             [-1, 14, 8, 8]              28
          Dropout-19             [-1, 14, 8, 8]               0
           Conv2d-20             [-1, 12, 6, 6]           1,512
             GELU-21             [-1, 12, 6, 6]               0
      BatchNorm2d-22             [-1, 12, 6, 6]              24
          Dropout-23             [-1, 12, 6, 6]               0
           Conv2d-24             [-1, 12, 6, 6]             168
           Conv2d-25             [-1, 12, 6, 6]           1,296
             GELU-26             [-1, 12, 6, 6]               0
      BatchNorm2d-27             [-1, 12, 6, 6]              24
          Dropout-28             [-1, 12, 6, 6]               0
        AvgPool2d-29             [-1, 12, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             120
================================================================
Total params: 7,902
Trainable params: 7,902
Non-trainable params: 0
----------------------------------------------------------------
        AvgPool2d-29             [-1, 12, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             120
================================================================
Total params: 7,902
Trainable params: 7,902
Non-trainable params: 0
----------------------------------------------------------------
Trainable params: 7,902
Non-trainable params: 0
----------------------------------------------------------------
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.62
Params size (MB): 0.03
Estimated Total Size (MB): 0.65
----------------------------------------------------------------
CUDA Available? False
EPOCH: 0
Loss=0.06323619931936264 Batch_id=937 Accuracy=82.70: 100%|███████████████████████| 938/938 [02:03<00:00,  7.60it/s] 

Test set: Average loss: 0.0663, Accuracy: 9796/10000 (97.96%)

EPOCH: 1
Loss=0.2595078647136688 Batch_id=937 Accuracy=96.57: 100%|████████████████████████| 938/938 [01:53<00:00,  8.27it/s] 

Test set: Average loss: 0.0489, Accuracy: 9843/10000 (98.43%)

EPOCH: 2
Loss=0.06437662243843079 Batch_id=937 Accuracy=97.30: 100%|███████████████████████| 938/938 [02:54<00:00,  5.37it/s] 

Test set: Average loss: 0.0387, Accuracy: 9881/10000 (98.81%)

EPOCH: 3
Loss=0.24132344126701355 Batch_id=937 Accuracy=97.75: 100%|███████████████████████| 938/938 [02:18<00:00,  6.79it/s] 

Test set: Average loss: 0.0305, Accuracy: 9905/10000 (99.05%)

EPOCH: 4
Loss=0.05336469039320946 Batch_id=937 Accuracy=97.99: 100%|███████████████████████| 938/938 [02:19<00:00,  6.72it/s] 

Test set: Average loss: 0.0296, Accuracy: 9911/10000 (99.11%)

EPOCH: 5
Loss=0.045321397483348846 Batch_id=937 Accuracy=98.16: 100%|██████████████████████| 938/938 [02:02<00:00,  7.64it/s] 

Test set: Average loss: 0.0333, Accuracy: 9899/10000 (98.99%)

EPOCH: 6
Loss=0.004230411723256111 Batch_id=937 Accuracy=98.27: 100%|██████████████████████| 938/938 [02:05<00:00,  7.47it/s] 

Test set: Average loss: 0.0246, Accuracy: 9928/10000 (99.28%)

EPOCH: 7
Loss=0.006700605619698763 Batch_id=937 Accuracy=98.36: 100%|██████████████████████| 938/938 [02:04<00:00,  7.55it/s] 

Test set: Average loss: 0.0274, Accuracy: 9914/10000 (99.14%)

EPOCH: 8
Loss=0.051461756229400635 Batch_id=937 Accuracy=98.35: 100%|██████████████████████| 938/938 [01:57<00:00,  8.02it/s] 

Test set: Average loss: 0.0267, Accuracy: 9915/10000 (99.15%)

EPOCH: 9
Loss=0.00195669406093657 Batch_id=937 Accuracy=98.40: 100%|███████████████████████| 938/938 [01:58<00:00,  7.92it/s] 

Test set: Average loss: 0.0251, Accuracy: 9918/10000 (99.18%)

EPOCH: 10
Loss=0.13488353788852692 Batch_id=937 Accuracy=98.53: 100%|███████████████████████| 938/938 [01:58<00:00,  7.93it/s] 

Test set: Average loss: 0.0231, Accuracy: 9921/10000 (99.21%)

EPOCH: 11
Loss=0.010298049077391624 Batch_id=937 Accuracy=98.61: 100%|██████████████████████| 938/938 [01:58<00:00,  7.91it/s] 

Test set: Average loss: 0.0182, Accuracy: 9944/10000 (99.44%)

EPOCH: 12
Loss=0.19654542207717896 Batch_id=937 Accuracy=98.57: 100%|███████████████████████| 938/938 [01:50<00:00,  8.51it/s] 

Test set: Average loss: 0.0212, Accuracy: 9926/10000 (99.26%)

EPOCH: 13
Loss=0.04861040413379669 Batch_id=937 Accuracy=98.63: 100%|███████████████████████| 938/938 [01:48<00:00,  8.61it/s]

Test set: Average loss: 0.0186, Accuracy: 9939/10000 (99.39%)

EPOCH: 14
Loss=0.28460389375686646 Batch_id=937 Accuracy=98.68: 100%|███████████████████████| 938/938 [01:49<00:00,  8.60it/s]

Test set: Average loss: 0.0231, Accuracy: 9923/10000 (99.23%)


===================
Results:
Parameters: 7.9k
Best Train Accuracy: 98.68
Best Test Accuracy: 99.44 (12th Epoch)
===================
PS E:\AI\github\era-v3-s7-cnn-cloud>
```


## AWS EC2 Instance Console Log for Model 3
```
(pytorch) ubuntu@ip-172-31-6-164:~/session7$ 
(pytorch) ubuntu@ip-172-31-6-164:~/session7$ python train.py 
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
              GELU-2           [-1, 10, 26, 26]               0
       BatchNorm2d-3           [-1, 10, 26, 26]              20
           Dropout-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 14, 24, 24]           1,260
              GELU-6           [-1, 14, 24, 24]               0
       BatchNorm2d-7           [-1, 14, 24, 24]              28
           Dropout-8           [-1, 14, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             140
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 14, 10, 10]           1,260
             GELU-12           [-1, 14, 10, 10]               0
      BatchNorm2d-13           [-1, 14, 10, 10]              28
          Dropout-14           [-1, 14, 10, 10]               0
           Conv2d-15           [-1, 14, 10, 10]             140
           Conv2d-16             [-1, 14, 8, 8]           1,764
             GELU-17             [-1, 14, 8, 8]               0
      BatchNorm2d-18             [-1, 14, 8, 8]              28
          Dropout-19             [-1, 14, 8, 8]               0
           Conv2d-20             [-1, 12, 6, 6]           1,512
             GELU-21             [-1, 12, 6, 6]               0
      BatchNorm2d-22             [-1, 12, 6, 6]              24
          Dropout-23             [-1, 12, 6, 6]               0
           Conv2d-24             [-1, 12, 6, 6]             168
           Conv2d-25             [-1, 12, 6, 6]           1,296
             GELU-26             [-1, 12, 6, 6]               0
      BatchNorm2d-27             [-1, 12, 6, 6]              24
          Dropout-28             [-1, 12, 6, 6]               0
        AvgPool2d-29             [-1, 12, 1, 1]               0
           Conv2d-30             [-1, 10, 1, 1]             120
================================================================
Total params: 7,902
Trainable params: 7,902
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.62
Params size (MB): 0.03
Estimated Total Size (MB): 0.65
----------------------------------------------------------------
CUDA Available? True
EPOCH: 0
Loss=0.15198493003845215 Batch_id=468 Accuracy=77.44: 100%|█████████████████████████████████████| 469/469 [00:09<00:00, 51.04it/s]

Test set: Average loss: 0.0948, Accuracy: 9719/10000 (97.19%)

EPOCH: 1
Loss=0.09935759752988815 Batch_id=468 Accuracy=96.50: 100%|█████████████████████████████████████| 469/469 [00:09<00:00, 51.17it/s]

Test set: Average loss: 0.0477, Accuracy: 9858/10000 (98.58%)

EPOCH: 2
Loss=0.10241536051034927 Batch_id=468 Accuracy=97.38: 100%|█████████████████████████████████████| 469/469 [00:09<00:00, 51.02it/s]

Test set: Average loss: 0.0414, Accuracy: 9876/10000 (98.76%)

EPOCH: 3
Loss=0.049096252769231796 Batch_id=468 Accuracy=97.77: 100%|████████████████████████████████████| 469/469 [00:09<00:00, 50.99it/s]

Test set: Average loss: 0.0450, Accuracy: 9869/10000 (98.69%)

EPOCH: 4
Loss=0.16335968673229218 Batch_id=468 Accuracy=98.00: 100%|█████████████████████████████████████| 469/469 [00:09<00:00, 51.13it/s]

Test set: Average loss: 0.0397, Accuracy: 9868/10000 (98.68%)

EPOCH: 5
Loss=0.08642027527093887 Batch_id=468 Accuracy=98.12: 100%|█████████████████████████████████████| 469/469 [00:09<00:00, 51.51it/s]

Test set: Average loss: 0.0415, Accuracy: 9869/10000 (98.69%)

EPOCH: 6
Loss=0.07729555666446686 Batch_id=468 Accuracy=98.17: 100%|█████████████████████████████████████| 469/469 [00:09<00:00, 51.28it/s]

Test set: Average loss: 0.0283, Accuracy: 9910/10000 (99.10%)

EPOCH: 7
Loss=0.04004257544875145 Batch_id=468 Accuracy=98.37: 100%|█████████████████████████████████████| 469/469 [00:09<00:00, 51.47it/s]

Test set: Average loss: 0.0226, Accuracy: 9932/10000 (99.32%)

EPOCH: 8
Loss=0.0827614963054657 Batch_id=468 Accuracy=98.47: 100%|██████████████████████████████████████| 469/469 [00:09<00:00, 51.36it/s]

Test set: Average loss: 0.0229, Accuracy: 9924/10000 (99.24%)

EPOCH: 9
Loss=0.05550988391041756 Batch_id=468 Accuracy=98.52: 100%|█████████████████████████████████████| 469/469 [00:09<00:00, 49.52it/s]

Test set: Average loss: 0.0275, Accuracy: 9917/10000 (99.17%)

EPOCH: 10
Loss=0.05013025179505348 Batch_id=468 Accuracy=98.55: 100%|█████████████████████████████████████| 469/469 [00:09<00:00, 51.25it/s]

Test set: Average loss: 0.0288, Accuracy: 9908/10000 (99.08%)

EPOCH: 11
Loss=0.08146730810403824 Batch_id=468 Accuracy=98.53: 100%|█████████████████████████████████████| 469/469 [00:09<00:00, 51.45it/s]

Test set: Average loss: 0.0213, Accuracy: 9932/10000 (99.32%)

EPOCH: 12
Loss=0.010671068914234638 Batch_id=468 Accuracy=98.56: 100%|████████████████████████████████████| 469/469 [00:09<00:00, 51.16it/s]

Test set: Average loss: 0.0219, Accuracy: 9925/10000 (99.25%)

EPOCH: 13
Loss=0.025044655427336693 Batch_id=468 Accuracy=98.65: 100%|████████████████████████████████████| 469/469 [00:09<00:00, 50.38it/s]

Test set: Average loss: 0.0181, Accuracy: 9944/10000 (99.44%)

EPOCH: 14
Loss=0.0764358714222908 Batch_id=468 Accuracy=98.66: 100%|██████████████████████████████████████| 469/469 [00:09<00:00, 50.18it/s]

Test set: Average loss: 0.0176, Accuracy: 9943/10000 (99.43%)


===================
Results:
Parameters: 7.9k
Best Train Accuracy: 98.66
Best Test Accuracy: 99.44 (14th Epoch)
===================
(pytorch) ubuntu@ip-172-31-6-164:~/session7$ 
```