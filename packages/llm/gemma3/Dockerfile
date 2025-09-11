# Copyright(C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Install deps
RUN apt-get update && apt-get install -y git vim wget

# Download test image
RUN mkdir -p /ryzers/data/ && \
    wget https://github.com/AMDResearch/Riallto/blob/main/notebooks/images/jpg/toucan.jpg?raw=true -O /ryzers/data/toucan.jpg

RUN python3 -m pip install \
	accelerate \
	timm==0.9.10 \
    huggingface_hub

# TODO: Install packages needed to run Gemma3-4B model
RUN python3 -m pip install \
    git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3


COPY test.sh /ryzers/test_gemma3.sh
RUN chmod +x /ryzers/test_gemma3.sh
COPY test_pipeline.sh /ryzers/test_pipeline_gemma3.sh
RUN chmod +x /ryzers/test_pipeline_gemma3.sh

# Create an entrypoint to autmatically login to Hugging Face on docker start
RUN git config --global credential.helper store
RUN echo '#!/bin/bash\n\
if [ -n "$HF_TOKEN" ]; then\n\
  hf auth login --token "$HF_TOKEN" --add-to-git-credential\n\
fi\n\
export HF_USER=$(hf auth whoami | head -n1)\n\
echo "🚀 Logged in as $HF_USER"\n\
exec "$@"' > /ryzers/entrypoint.sh && chmod +x /ryzers/entrypoint.sh

ENTRYPOINT ["/ryzers/entrypoint.sh"]

CMD ["/bin/bash", "-c", "/ryzers/test_gemma3.sh"]
