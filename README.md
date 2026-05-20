<div align="center">

<p align="center" style="border-radius: 10px">
  <img src="asset/logo3.png" width="35%" alt="logo"/>
</p>

<h1>MobileI2V: Fast and High-Resolution Image-to-Video on Mobile Devices</h1>


[Shuai Zhang](https://github.com/Shuaizhang7)<sup>\*</sup>, [Bao Tang](https://github.com/Tt-DAY)<sup>\*</sup>, Siyuan Yu<sup>\*</sup>, [Yueting Zhu](https://github.com/lazypomeloo), [Jingfeng Yao](https://github.com/JingfengYao),<br>[Ya Zou](https://github.com/ZouYa99), Shanglin Yuan, Li Yu, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu), [Xinggang Wang](https://xwcv.github.io/index.htm)<sup>📧</sup>


Huazhong University of Science and Technology (HUST) 

(\* equal contribution, 📧 corresponding author)

[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://hustvl.github.io/MobileI2V/)
[![arxiv paper](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org/abs/2511.21475)
[![checkpoints](https://img.shields.io/badge/HuggingFace-🤗-green)](https://arxiv.org/abs/2511.21475)

</div>






## 🎯 How to Use

### Installation
You can install the required environment using the provided requirements.txt file. 

```
pip install -r requirements.txt
```
### Data Processing
Process the data the same way as the main branch training code.

### Distillation training
You can use the provided ./train_scripts/train_i2v_distill.sh script for training. The configuration file is located at: ./configs/mobilei2v_config/. Before training, download the weights for [video-vae](https://huggingface.co/Lightricks/LTX-Video/tree/main/vae) and [qwen2-0.5B](https://huggingface.co/Qwen/Qwen2-0.5B/tree/main) and replace the model path in the configuration file.
```
bash ./train_scripts/train_i2v_distill.sh
```

## ❤️ Acknowledgements

Our MobileI2V codes are mainly built with [SANA](https://github.com/NVlabs/Sana) and [LTX-Video](https://github.com/Lightricks/LTX-Video). The data processing workflow is based on [OpenSora](https://github.com/hpcaitech/Open-Sora). Thanks for all these great works.




## 📝 Citation

If you find MobileI2V useful, please consider giving us a star 🌟 and citing it as follows:

```
@misc{MobileI2V,
      title={MobileI2V: Fast and High-Resolution Image-to-Video on Mobile Devices}, 
      author={Shuai Zhang and Bao Tang and Siyuan Yu and Yueting Zhu and Jingfeng Yao and Ya Zou and Shanglin Yuan and Li Yu and Wenyu Liu and Xinggang Wang},
      year={2025},
      eprint={2511.21475},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.21475}, 
}

```
