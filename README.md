# Decode JPEG image on GPU using PyTorch api

## Install

- `python setup.py bdist_wheel`
- in `dist` directory, `pip install torchnvjpeg-0.1.0-cp36-cp36m-linux_x86_64.whl`

## How to use

### single decode

```python
import torch
import torchnvjpeg
decoder = torchnvjpeg.Decoder()

image_data = open("images/cat.jpg", 'rb').read()

image_tensor = decoder.decode(image_data)  # run on GPU
assert image_tensor.is_cuda

import torchvision
transform = torchvision.transform.Resize((224, 224))
resized_tensor = transform(image_tensor.permute((2, 0, 1))) # run on GPU
```

### batch decode

```python
import torch
import torchnvjpeg
batch_size = 8
max_cpu_threads = 8
device_id = 0
max_image_size = 3840 * 2160 * 3
decoder = torchnvjpeg.Decoder(0, 0, True, device_id, batch_size, max_cpu_threads, max_image_size, torch.cuda.current_stream(device_id))

image_path = "images/cat.jpg"
data = open(image_path, 'rb').read()
data_list = [data for _ in range(batch_size)]

image_tensor_list = decoder.batch_decode(data_list)
```

### parallel decode

```python
import torch
import torchnvjpeg
from multiprocessing.pool import ThreadPool

batch_size = 8
image_path = "images/cat.jpg"
data = open(image_path, 'rb').read()
data_list = [data for _ in range(batch_size)]

decoder_list = [torchnvjpeg.Decoder() for _ in range(batch_size)]

cpu_threads = 4
pool = ThreadPool(cpu_threads)

def run(args):
    decoder, data = args
    return decoder.decode(data)

image_tensor_list = pool.map(run, zip(decoder_list, data_list))


```

## Train

import py/train/gpu_preprocess.py file, wrap data_loader (torch.utils.data.DataLoader) with `gpu_loader`
```python
gpu_data_loader = gpu_loader(cpu_loader, data_transform)
```