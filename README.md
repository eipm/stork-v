# BELA: Accurate and Noninvasive Ploidy Prediction for Human Preimplantation Embryos

## Project Description

This project aims to enhance the assessment of fertilized human embryos, a critical step in the process of in vitro fertilization (VF). The research paper behind this project introduced BELA, the Blastocyst Evaluation Learning Algorithm, a novel model for embryo ploidy status prediction. It surpasses both image- and video-based ploidy models without necessitating any subjective input from embryologists. This project uses deep learning techniques to accurately and noninvasively predict ploidy in preimplantation human embryos. 

[![Actions Status](https://github.com/eipm/stork-v/workflows/Docker/badge.svg)](https://github.com/eipm/stork-v/actions) [![Github](https://img.shields.io/badge/github-1.0.0-green?style=flat&logo=github)](https://github.com/eipm/stork-v) [![EIPM Docker Hub](https://img.shields.io/badge/EIPM%20docker%20hub-1.0.0-blue?style=flat&logo=docker)](https://hub.docker.com/repository/docker/eipm/stork-v) [![GitHub Container Registry](https://img.shields.io/badge/GitHub%20Container%20Registry-1.0.0-blue?style=flat&logo=docker)](https://github.com/orgs/eipm/packages/container/package/stork-v) [![Python 3.8.16](https://img.shields.io/badge/python-3.8.16-blue.svg)](https://www.python.org/downloads/release/python-360/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Set up local environment and install dependencies

Install requirements from requirements.txt
`pip install -r requirements.txt`

Add the models in the folder
`src/stork_v/models`

## Execute a model as script

Add the zip file in the expeted location `src/data/78645636.zip` and run the command
`python src/test.py`

## Docker run command


```
PATH_TO_MODELS=<set the location of your models directory>
PATH_TO_DATA=<set the location of you data directory>
PATH_TO_TEMP=<set the location of the temp directory used for creating the videos>

docker run --name stork-v \
-v $PATH_TO_MODELS:/stork-v/stork_v/models \
-v $PATH_TO_DATA:/stork-v/data \
-v $PATH_TO_TEMP:/stork-v/temp \
-e USERS_DICT="{'user1': 'stork'}" \
-p 8080:80 \
stork-v
```


## Instructions for Use of Training Files

Files are available for training a BELA model in the 'main' folder.

Files available provide code for training and evaluating a BELA model through:
1. Video Creation
2. Annotation File Creation
3. Training and Prediction

## Contributors

- Suraj Rajendran, Institute for Computational Biomedicine, Department of Physiology and Biophysics, Weill Cornell Medicine of Cornell University, New York, NY, USA, Tri-Institutional Computational Biology & Medicine Program, Cornell University, NY, USA
- Matthew Brendel, Institute for Computational Biomedicine, Department of Physiology and Biophysics, Weill Cornell Medicine of Cornell University, New York, NY, USA
- Josue Barnes, Institute for Computational Biomedicine, Department of Physiology and Biophysics, Weill Cornell Medicine of Cornell University, New York, NY, USA
- Qiansheng Zhan, The Ronald O. Perelman and Claudia Cohen Center for Reproductive Medicine, Weill Cornell Medicine, New York, NY, USA
- Jonas E. Malmsten, The Ronald O. Perelman and Claudia Cohen Center for Reproductive Medicine, Weill Cornell Medicine, New York, NY, USA
- Pantelis Zisimopoulos, Institute for Computational Biomedicine, Department of Physiology and Biophysics, Weill Cornell Medicine of Cornell University, New York, NY, USA
- Alexandros Sigaras, Institute for Computational Biomedicine, Department of Physiology and Biophysics, Weill Cornell Medicine of Cornell University, New York, NY, USA
- Marcos Meseguer, IVI Valencia, Health Research Institute la Fe, Valencia, Spain
- Kathleen A Miller, IVF Florida Reproductive Associates, Fort Lauderdale, Florida, USA
- David Hoffman, IVF Florida Reproductive Associates, Fort Lauderdale, Florida, USA
- Zev Rosenwaks, The Ronald O. Perelman and Claudia Cohen Center for Reproductive Medicine, Weill Cornell Medicine, New York, NY, USA
- Olivier Elemento, Institute for Computational Biomedicine, Department of Physiology and Biophysics, Weill Cornell Medicine of Cornell University, New York, NY, USA
- Nikica Zaninovic, The Ronald O. Perelman and Claudia Cohen Center for Reproductive Medicine, Weill Cornell Medicine, New York, NY, USA
- Iman Hajirasouliha (Corresponding author), Institute for Computational Biomedicine, Department of Physiology and Biophysics, Weill Cornell Medicine of Cornell University, New York, NY, USA, imh2003@med.cornell.edu

## Contact

For any questions or comments, please contact Iman Hajirasouliha at imh2003@med.cornell.edu.

