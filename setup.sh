export TRITON_ALL_BLOCKS_PARALLEL=1
export RES_INTERNAL="-i http://resource.flagos.net:2116/repository/flagos-pypi-hosted/simple --trusted-host resource.flagos.net"

# FlagOS
pip install -e . $RES_INTERNAL

# vLLM
pip install -e . $RES_INTERNAL