# stork-v

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
