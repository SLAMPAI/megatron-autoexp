# Megatron AutoExperiment Helper Scripts

This directory contains scripts and configuration for running experiments related to Megatron using the `autoexp` tool.

## Prerequisites

You need to have `autoexp` installed. Follow the full installation instructions here:
[https://github.com/SLAMPAI/autoexperiment](https://github.com/SLAMPAI/autoexperiment)

## Usage

Once `autoexp` is installed and your environment is set up, you can run experiments using the provided YAML configuration file.
```
autoexp build testing_autoexp.yaml 
autoexp run testing_autoexp.yaml
```

## Notes
 - Currently, the container image is defined in `train_container_single.yaml`. Make sure to update this if you change the container image.
 - There is no support for automated evaluation of the yet. We might add this in the future.
 - Make sure to change the paths in the YAML file to point to your local directories.
    - Specifically, all values with `PATH_TO` in the YAML file, and the `template` key.