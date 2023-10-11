import tensorflow as tf

# Check for available GPUs
gpu_devices = tf.config.experimental.list_physical_devices('GPU')

if gpu_devices:
    for gpu_device in gpu_devices:
        print("Name:", gpu_device.name)
        print("Device type:", gpu_device.device_type)
else:
    print("No GPUs found.")
