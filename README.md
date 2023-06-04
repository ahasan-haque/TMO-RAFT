# TMO

This is an easy-to-use video object segmentation code, based on the paper: 

> **Treating Motion as Option to Reduce Motion Dependency in Unsupervised Video Object Segmentation**, *WACV'23*\
> [Suhwan Cho](https://github.com/suhwan-cho), [Minhyeok Lee](https://github.com/Hydragon516), [Seunghoon Lee](https://github.com/iseunghoon), [Chaewon Park](https://github.com/codnjsqkr), [Donghyeong Kim](https://github.com/donghyung87), Sangyoun Lee

URL: [[Official]](https://openaccess.thecvf.com/content/WACV2023/html/Cho_Treating_Motion_as_Option_To_Reduce_Motion_Dependency_in_Unsupervised_WACV_2023_paper.html) [[arXiv]](https://arxiv.org/abs/2209.03138)\
PDF: [[Official]](https://openaccess.thecvf.com/content/WACV2023/papers/Cho_Treating_Motion_as_Option_To_Reduce_Motion_Dependency_in_Unsupervised_WACV_2023_paper.pdf) [[arXiv]](https://arxiv.org/pdf/2209.03138.pdf)

<img src="https://user-images.githubusercontent.com/54178929/208474605-7586894f-11cf-4e38-ac21-75a78216c22d.png" width=800>


## Abstract
In unsupervised VOS, most state-of-the-art methods leverage motion cues obtained from optical flow maps in addition to appearance cues. However, as they are overly dependent on motion cues, which may be unreliable in some cases, they cannot achieve stable prediction. To overcome this limitation, we design a novel network that operates regardless of motion availability, termed as a **motion-as-option network**. Additionally, to fully exploit the property of the proposed network that motion is not always required, we introduce a **collaborative network learning strategy**. As motion is treated as option, fine and accurate segmentation masks can be consistently generated even when the quality of the flow maps is low.

## Preparation
Create a **dataset** directory, and under the directory, store your image sequences under different directory name. Like below:

```
TMO-RAFT
|
-- datasets
  | 
   -- car
      |
        -- img1.jpg
        -- img2.jpg
   -- street
      |
        -- img1.png
        -- img2.png  
```

For evaluation, make **gt** directory (ground truth masks) with same nested structure.
Also, create a **output** directory (mandatory if you run inside a docker container, otherwise optional)

## Testing
1\. Make sure the pre-trained models are in your *"trained_model"* folder.
2\. Install the dependencies.

```
pip install -r requirements.txt
```
2\. Run the following line:
```
python3.9 run.py --test --dataset-path=$DATASET_PATH --output-path=$OUTPUT_PATH
```
In our setup:

```
$DATASET_PATH = "/datasets"
$OUTPUT_PATH = "/output"
```

Feel free to change according to your setup

4\. You can also directly download [pre-trained model](https://drive.google.com/file/d/12k0iZhcP6Z8RdGKCKHvlZq5g9kNtj8wA/view?usp=share_link) and [pre-computed results](https://drive.google.com/file/d/1bWrxXiE5_0Kz-i63xoRk68r8cJL8kMgY/view?usp=sharing).

## Running in docker

You can run a docker container with the following command:

```
docker build -t tmo-raft . && \
docker run -e DATASET_PATH=/datasets \
           -e OUTPUT_PATH=/output \
           -e GT_PATH =/gt \
           -v HOST_DATASET_PATH:/datasets \
           -v HOST_GT_PATH:/gt \
           -v HOST_OUTPUT_PATH:/output \
           --gpus all
           tmo-raft
```

For example, this is the exact command is run locally:

```
docker build -t tmo-raft . && \
docker run -e DATASET_PATH=/datasets \
           -e OUTPUT_PATH=/output \
           -e GT_PATH =/gt \
           -v c:\Users\ge79pih\tmo_data\tmo\tmo_dataset:/datasets \
           -v c:\Users\ge79pih\tmo_data\tmo\tmo_gt:/gt \
           -v c:\Users\ge79pih\tmo_data\tmo\tmo_output:/output \
           --gpus all
           tmo-raft
```

## License
The skeleton of this project is taken from [official TMO Implementation][https://github.com/suhwan-cho/TMO]. Special thanks to @suhwan-cho for providing guideline.
