# Supplemental Documentation for MosaicML BERT

## Docker Image

Pull the base image

```shell
docker pull mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04
```

WIP, H100:
```shell
docker pull mosaicml/pytorch:2.1.2_cu121-python3.10-ubuntu20.04

# or:

docker pull mosaicml/pytorch:2.7.0_cu126-python3.12-ubuntu22.04
```

Launch the base image for the first time

```shell
docker run --gpus all -it \
  --shm-size=200g \
  --name mosaic-bert-dev \
  -v ~/workspace:/workspace \
  -w /workspace \
  --cap-add SYS_ADMIN \
  --device /dev/fuse \
  --security-opt apparmor:unconfined \
  mosaicml/pytorch:1.13.1_cu117-python3.10-ubuntu20.04 \
  bash
```

Commit running docker container

```shell
docker commit mosaic-bert-dev mosaic-bert-dev:latest

or

docker commit mosaic-bert-dev mosaic-bert:v100-dev-latest
```

Relaunch the saved docker image

```shell
docker run --gpus all -it \
  --shm-size=200g \
  --name mosaic-bert-dev \
  -v ~/workspace:/workspace \
  -w /workspace \
  --cap-add SYS_ADMIN \
  --device /dev/fuse \
  --security-opt apparmor:unconfined \
  mosaic-bert-dev:latest \
  bash
```

## Azure Container Registry

Login

```shell
az acr login --name msrmoldyn
```

Tag local image

```shell
docker tag mosaic-bert-dev/latest msrmoldyn.azurecr.io/genomic-research/mosaic-bert:v100-dev-latest
```

Push image to ACR

```shell
docker push msrmoldyn.azurecr.io/genomic-research/mosaic-bert:v100-dev-latest
```

Pull from ACR

```shell
docker pull msrmoldyn.azurecr.io/genomic-research/mosaic-bert:v100-dev-latest
```

## Processing C4 Data

It is advised to create a symbolic link that links to an external storage, such as mounted blob storage.

The link can be created under `bert` directory, as `bert/data`.

After that, to process c4 dataset, run

```shell
python src/convert_dataset.py --dataset c4 --data_subset en \
    --out_root ./data/c4 --splits train_small val
```

```shell
python src/convert_dataset.py --dataset c4 --data_subset en \
    --out_root ./data/c4 --splits train
```
