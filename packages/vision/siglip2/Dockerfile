# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /ryzers

# Download test image
RUN mkdir -p /ryzers/data/ && \
    wget https://github.com/AMDResearch/Riallto/blob/main/notebooks/images/jpg/toucan.jpg?raw=true -O /ryzers/data/toucan.jpg

# Install transformers for SigLIP-2
RUN pip install git+https://github.com/huggingface/transformers@v4.49.0-SigLIP-2 'accelerate>=0.26.0'

# Copy the test scripts
COPY test.sh /ryzers/test.sh
COPY test.py /ryzers/test.py
RUN chmod +x /ryzers/test.sh

# Set the entrypoint to the test script
CMD /ryzers/test.sh

