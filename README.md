
# TriFinger RL Datasets

This repository provides offline reinforcement learning datasets collected on the real TriFinger platform and in a simulated version of the environment. The paper ["Benchmarking Offline Reinforcement Learning on Real-Robot Hardware"](https://openreview.net/pdf?id=3k5CUGDLNdd) provides more details on the datasets and benchmarks offline RL algorithms on them. All datasets are available with camera images as well.

More detailed information about the simulated environment, the datasets and on how to run experiments on a cluster of real TriFinger robots can be found in the [documentation](https://webdav.tuebingen.mpg.de/trifinger-rl/docs/).

Some of the datasets were used during the [Real Robot Challenge 2022](https://real-robot-challenge.com).

## Installation 

```bash
cd trifinger_rl_datasets
pip install --upgrade pip  # make sure the most recent version of pip is installed
pip install .
cd ..
```

## Preprocess 

```bash
python preprocess.py --input_dataset trifinger-cube-lift-real-expert-v0 --output_dataset trifinger-cube-lift-real-expert-v0-masa
```

## Usege in Python

```python
    old_env = gym.make(
               args.input_dataset,
               flatten_obs=True)
    old_dataset = old_env.get_dataset(rng=(0,2))
    print('Observation Shape', old_dataset['observations'][0].shape)

    new_env = gym.make(
               args.input_dataset,
               flatten_obs=True,
               data_dir=f'output_datasets/{args.output_dataset}')

    new_dataset = new_env.get_dataset(rng=(0,2))
    print('Observation Shape', new_dataset['observations'][0].shape)
```
