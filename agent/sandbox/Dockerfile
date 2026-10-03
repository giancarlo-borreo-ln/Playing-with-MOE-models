# Sandbox image for the C agent.
#
# Deliberately small: just enough toolchains for the model to write a program.
# At runtime the agent runs this image with:
#   --network none --read-only --cap-drop ALL --pids-limit 128
#   --memory 256m --cpus 1 --user 65534:65534
# and only /work (a host bind mount) plus a small tmpfs are writable.
FROM debian:12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      gcc \
      g++ \
      libc6-dev \
      python3 \
      bash \
      coreutils \
      findutils \
      ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Everything runs as nobody (65534); $HOME points into the tmpfs.
ENV HOME=/tmp
WORKDIR /work

CMD ["bash"]
