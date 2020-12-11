# Conditional-Glow

This code is a Python implementation of the conditional-Glow introduced in the paper 

"[Structured Output Learning with Conditional Generative Flows](https://arxiv.org/abs/1905.13288)". You Lu and Bert Huang. AAAI 2020.

Note: This code is used for the experiments of binary segmentation on the Weizmann Horse dataset. Some parts of the code are adapted from [chaiyujin](https://github.com/chaiyujin/glow-pytorch), and [openai](https://github.com/openai/glow). 

## Requirements:

This code was tested using the the following libraries.

- Python 3.6.7
- Numpy 1.14.6
- Pytorch 1.2.0
- Pillow 5.3.0

## Running

- Download the dataset from [here](https://www.msri.org/people/members/eranb/).
- Rename the forlders */rgb* and */figure_ground* to be */images*, and */labels*, respectively.
- Within the same folder, create files *train.txt*, *valid.txt*, and *test.txt*, which contain the names of images for training, validation, and test, respectively.
- Configure the parameters in the shell script *train_cglow.sh*
- In the terminal, run *./train_cglow.sh*

## Contact
Feel free to send me an email, if you have any questions or comments.
