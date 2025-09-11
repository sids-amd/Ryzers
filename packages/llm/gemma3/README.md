### OpenVLA

This directory contains the docker configuration files to run Gemma3-4B on RyzenAI platforms. Gemma3-4B is the smallest open-weights model released under the Gemma3 suite that offers multimodal support.

### Build and run the Docker Image

Gemma3 is a gated HF model, so it requires a HF access token and permissions from HF Gemma3 repository (accept TOS). Before you build gemma3 docker image, you need to specify your access token inside `config.yaml`. On launch, the docker container will automatically authenticate you to HF using your access token.

```sh
ryzers build gemma3
ryzers run
```

Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.