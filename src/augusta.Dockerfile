

ENV LD_LIBRARY_PATH="/var/lib/tcpxo/lib64:${LD_LIBRARY_PATH}" \
    NCCL_TUNER_PLUGIN=libnccl-tuner.so \
    NCCL_TUNER_CONFIG_PATH=/var/lib/tcpxo/lib64/a3plus_tuner_config.textproto \
    NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/var/lib/tcpxo/lib64/a3plus_guest_config.textproto \
    NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices
