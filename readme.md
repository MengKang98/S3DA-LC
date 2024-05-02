 # Alleviating Imbalanced Pseudo-label Distribution: Self-Supervised Multi-Source Domain Adaptation with Label-specific Confidence

## Official implementation for **S3DA-LC** (Based on [SImpAl](https://sites.google.com/view/simpal))

### Parameters:
- `--tau` : refer to $\tau$ in paper
- `--w_k` : whether to use weights $w_k$
- `--UTF` : refer to $\lambda$ in paper

**Please refer [main.py](./main.py) for the detailed parameters setting.**

### Training:
    python main.py --dataset office-31 --task DW_A --tau 0.9 --UTF 1.5 --w_k 1

### t-SNE visualization

[after warm up](./t-sne/afterwarmup.pdf)   
[after converge](./t-sne/convergence.pdf)
    
### Results on [Office-31](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view):
 
 | Method |$\rightarrow$ A | $\rightarrow$ W | $\rightarrow$ D | Avg |
 | :--- | --- | --- | --- | --- |
 | CAiDA | 75.8 | 98.9 | 99.8 | 91.6 |
 | DECISION | 75.4 | 98.4 | 99.6 | 91.1 |
 | SPS | 73.8 | **99.3** | **100.0** | 91.10 |
 | **S3DA-lc** | **78.1** | 99.0 | **100.0** | **92.4** |

### Results on [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view):

 | Method | $\rightarrow$ Ar | $\rightarrow$ Cl | $\rightarrow$ Pr | $\rightarrow$ Rw | Avg |
 | :--- | --- | --- | --- | --- | --- |
 | CAiDA | 75.2 | 60.5 | 84.7 | 84.2 | 76.2 |
 | DECISION | 74.5 | 59.4 | 84.4 | 83.6 | 75.5 |
 | SPS | 75.1 | 66.0 | 84.4 | 84.2 | 77.4 |
 | **S3DA-lc** | **78.1** | **70.0** | **87.4** | **87.2** | **80.7** |

### Results on [DomainNet](http://ai.bu.edu/DomainNet/):

 | Method | $\rightarrow$ Clp | $\rightarrow$ Inf | $\rightarrow$ Pnt | $\rightarrow$ Qdr | $\rightarrow$ Rel | $\rightarrow$ Skt | Avg |
 | :--- | --- | --- | --- | --- | --- | --- | --- |
 |MSCAN | 69.3 | 28.0 | 58.6 | **30.3** | 73.3 | 59.5 | 53.2 |
 | KD3A | **72.5** | 23.4 | 60.9 | 16.4 | 72.7 | 60.6 | 51.1 |
 | STEM | 72.0 | 28.2 | **61.5** | 25.7 | 72.6 | 60.2 | 53.4 |
 | **S3DA-lc** | 71.9 | **31.3** | 61.3 | 27.1 | **75.7** | **61.2** | **54.8** |
