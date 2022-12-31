# Transformers in Pytorch
## _what are provided in this repo?_
- **Model**: Several transformer-based milestone models are **reimplemented from scratch via pytorch**
  - _Vision Transformer (CV)_ : [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)
  > <img src="/readme_supply/ViT.png" width=40% height=40%></img>
  - _Vanilla Transformer (NLP)_ : [Attention Is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
  > <img src="/readme_supply/vanilla_transformer.png" width=20% height=20%></img>
  >
- **Experiments**: Conduct experiments on **CV/NLP benchmark**, respectively
  - _Image Classification_ : 
    - [ImageNet1k](https://www.image-net.org)
    > <img src="/readme_supply/imagenet1k_dataset.png" width=30% height=30%></img>
    - [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
    > <img src="/readme_supply/cifar10_dataset.png" width=20% height=20%></img>
  - _Neural Machine Translation_ :
    - [Multi30k](https://arxiv.org/pdf/1605.00459.pdf)
    > <img src="/readme_supply/multi30k_dataset.png" width=30% height=30%></img>
- **Pipeline**: End-to-end pipeline
  - _Conveniently Playing_ : integrate data processing and model training/validation into **one-stop shop** pipeline
  - _Efficiently Training_ : accelerate training and evaluating via **DistributeDataParallel(DDP)** and **Mixed Precision(fp16)**
  - _Neatly Reading_ : neat file structure, **easy for reading** but non-trivial 
    - ./script → run train/eval 
    - ./model → model implementation
    - ./data → data processing
# Usage
## 1. Env Requirements
```
# Conda Env
python 3.6.10
torch 1.4.0+cu100
torchvision 0.5.0+cu100
torchtext 0.5.0
spacy 3.4.1
tqdm 4.63.0

# Apex (For mix precision training)
## run `gcc --version` 
gcc (GCC) 5.4.0
## apex installation
git clone https://github.com/NVIDIA/apex
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
rm -rf ./build
python setup.py install --cuda_ext --cpp_ext

# System Env 
## run `nvcc --version`
Cuda compilation tools, release 10.0, V10.0.130
# run `nvidia-smi`
Check your own gpu device status
```
## 2. Data Requirements
- **multi30k, cifar10** could be automatically downloaded in pipeline
- **imagenet1k(ILSVRC2012)** need manual download (_Guide for download imagenet1k_)
  - Wait until three files download.
    - ILSVRC2012_devkit_t12.tar.gz (2.5M)
    - ILSVRC2012_img_train.tar (138G)
    - ILSVRC2012_img_val.tar (6.3G)
  - Run imagenet1k pipeline, ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar will be automatically unzipped and arranged in two directories 'data/ILSVRC2012/train' and 'data/ILSVRC2012/val'.
  - But the unzip process costs more than a few hours or you can do it faster by shell anyway.
```
# Guide for download imagenet1k
mkdir -p data/ILSVRC2012
cd data/ILSVRC2012
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar --no-check-certificate
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```
## 3. Run Experiments
### 3.1 Fine-tuning ViT on imagenet1k, cifar10
- Download pretrained ViT_B_16 model parameters from official storage.
```
cd data
curl -o ViT_B_16.npz https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz
curl -o ViT_B_16_384.npz https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz
```
- Before run experiments
  - Set CUDA env in script/run_img_cls_task.py/\_\_main__ according to your GPU device 
  - Adjust train/eval settings in script/run_img_cls_task.py/get_args() and launch the experiment
```
cd script

# run experiments on cifar10
# (4mins/epoch, 3.5hours totally | GPU device: P40×4)
python ./run_image_cls_task.py cifar10

# run experiments on imagenet1k
# (less than 5hours/epcoch, more than 10hours totally | GPU device: P40×4 )
python ./run_image_cls_task.py ILSVRC2012

# Tips:
# 1. Both DDP and FP16 Mixed Precision Training are adopted for accelerating
# 2. The ratio of acceleration depends on your specific GPU device
```
### 3.2 Train vanilla transformer from scratch on multi30k (en → de)
- Before run experiments
  - Set CUDA env in script/run_nmt_task.py/\_\_main__ according to your GPU device 
  - Adjust train/eval settings in script/run_nmt_task.py/get_args() and launch the experiment
```
# run experiments on multi30k (small dataset ,3mins total | GPU device : P40×4 | U can also fork and adjust the pipeline and run this experiments in a small capacity gpu device)
cd script

python ./run_nmt_task.py multi30k

# Tips:
# 1. DDP is adopted for accelerating
# 1. For inference, "greedy search" and "beam search" are also included in the nmt task pipeline.
```
## 4. Results
### 4.1 Fine-tuning ViT on imagenet1k, cifar10,
- This repo
  - **Imagenet1k : _ACC 84.9%_** (result on 50,000 val set |_resolution 384 | extra label smoothing confidence 0.9 | batch size 160, nearly 15,000 training steps_)
  > <img src="/readme_supply/imagenet1k_result.png" width=40% height=40%></img>
  - **Cifar10 : _ACC 99.04%_** (_resolution 224 | batch size 640, nearly 5500 training steps_) 
  > <img src="/readme_supply/cifar10_result.png" width=30% height=30%></img>
- Comparison to official result [ViT Implementation](https://github.com/google-research/vision_transformer) by Google
  > <img src="/readme_supply/both_benchmark.png" width=50% height=50%></img>
### 4.2 Train vanilla transformer from scratch on multi30k (en → de)
- This repo
  - **Multi30k : _BLEU 38.6_** (_en→de | nearly 17M #Params | batch size 512, nearly 1200 training steps_)
  > <img src="/readme_supply/multi30k_result.png" width=40% height=40%></img>
- Comparison to results in [Dynamic Context-guided Capsule Network for Multimodal Machine Translation](https://arxiv.org/pdf/2009.02016v1.pdf)
  > <img src="/readme_supply/multi30k_benchmark.png" width=50% height=50%></img>

# Reference materials for further study
- Transformer Survey
  - [Efficient Transformers: A Survey](https://arxiv.org/pdf/2009.06732.pdf)
  - [A Survey of Transformers](https://arxiv.org/pdf/2106.04554.pdf)
- Vanilla Transformer Component Structures 
  - self-attention
    - [ON THE RELATIONSHIP BETWEEN SELF-ATTENTION AND CONVOLUTIONAL LAYERS](https://arxiv.org/pdf/1911.03584.pdf)
    - [Online and Linear-Time Attention by Enforcing Monotonic Alignments](https://arxiv.org/pdf/1704.00784.pdf)
  - multi-head
    - [Are Sixteen Heads Really Better than One?](https://proceedings.neurips.cc/paper/2019/file/2c601ad9d2ff9bc8b282670cdd54f69f-Paper.pdf)
    - [Multi-head or Single-head? An Empirical Comparison for Transformer Training](https://arxiv.org/pdf/2106.09650.pdf)
  - feed forward network
    - [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/pdf/2012.14913.pdf)
  - residual connection & layer norm
    - [On Layer Normalization in the Transformer Architecture](https://arxiv.org/pdf/2002.04745.pdf)
    - [PowerNorm: Rethinking Batch Normalization in Transformers](https://arxiv.org/pdf/2003.07845.pdf)
  - label smoothing related
    - [When Does Label Smoothing Help?](https://proceedings.neurips.cc/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf)
    - [On Calibration of Modern Neural Networks](https://arxiv.org/pdf/1706.04599.pdf)
    - [Calibration of Pre-trained Transformers](https://aclanthology.org/2020.emnlp-main.21.pdf)
    - [Revisiting the Calibration of Modern Neural Networks](https://openreview.net/pdf?id=QRBvLayFXI)
- Recent Transformer Milestone Work in CV
  - [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)
  - [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)
  - [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/pdf/2111.06377.pdf)
  - [MetaFormer Is Actually What You Need for Vision](https://arxiv.org/pdf/2111.11418.pdf)
    
