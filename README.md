

# Deep Multimodal Feature Encoding for Video Ordering
### [Paper](https://arxiv.org/pdf/2004.02205.pdf) <br>

[Vivek Sharma](http://vivoutlaw.github.io), 
[Makarand Tapaswi](http://www.cs.toronto.edu/~makarand/), 
and [Rainer Stiefelhagen](https://cvhci.anthropomatik.kit.edu/people_596.php)

In IEEE International Conference on Computer Vision (ICCV) workshop on Large Scale Holistic Video Understanding, 2019 

### Temporal Compact Bilinear Pooling (TCBP) Layer

Demo:

    $ from TCBP import TCBP
    $ import torch
    $ data = torch.rand([10,8192,4,7,7]) 
    $ tcbp = TCBP(input_dim1=8192, input_dim2=8192,output_dim=512, temporal_window=4, spat_x=7, spat_y=7)
    $ tcbp_representation = tcbp(data,data)
    $ tcbp_representation.shape  
    $ ---> torch.Size([10, 512])

### Citation

If you find the code and datasets useful in your research, please cite:
    
    @inproceedings{ssiam,
        author    = {Sharma, Vivek and Tapaswi, Makarand and Stiefelhagen, Rainer}, 
        title     = {Deep Multimodal Feature Encoding for Video Ordering}, 
        booktitle = {IEEE ICCV Workshop on Large Scale Holistic Video Understanding},
        year      = {2019}
    }
