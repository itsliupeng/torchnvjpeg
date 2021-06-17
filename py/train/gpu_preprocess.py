import copy
import queue
import threading
from multiprocessing.pool import ThreadPool
from queue import Queue

import cv2
import numpy as np
import torch
import torchnvjpeg
import torchvision


def read_image_bytes(image_path: str):
    return open(image_path, 'rb').read()


def identity_collate(batch):
    return batch


def pop_decode_and_to_tensor(t: torchvision.transforms.Compose):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print('*** Remove useless transform ***')
        print('Before filter:')
        print(t)

    filtered_t = []
    for x in t.transforms:
        if isinstance(x, torchvision.transforms.Normalize) or isinstance(x, torchvision.transforms.ToTensor):
            continue
        filtered_t.append(x)

    new_t = copy.deepcopy(t)
    new_t.transforms = filtered_t

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print('After filter:')
        print(new_t)

    return new_t


def gpu_loader(data_loader, transform, cpu_threads=8, image_index=[0]):
    """
    :param data_loader: pytorch data_loader
    :param transform: torchvision.transforms.Compose([...])
    :param cpu_threads:
    :param image_index: the index of image bytes in the outputs of pytorch data_loader, 0 by default.
    :return:
    """
    filtered_transform = pop_decode_and_to_tensor(transform)
    return GpuPreprocessIterator(data_loader,
                                 filtered_transform,
                                 device_id=torch.cuda.current_device(),
                                 cpu_threads=cpu_threads,
                                 image_index=image_index)


class BatchDecodeTransform(object):

    def __init__(self, cpu_threads, transform, device_id, stream=None):
        self.cpu_threads = cpu_threads
        self.transform = transform

        self._decoder_queue = Queue()

        for _ in range(self.cpu_threads):
            decoder = torchnvjpeg.Decoder(0, 0, True, device_id, 4096 * 4096 * 3, stream=stream)
            self._decoder_queue.put(decoder)

        # thread pool for decoding
        self.pool = ThreadPool(self.cpu_threads)
        self.torch_cuda_device = torch.cuda.device(device_id)
        self.device = torch.device(device_id)

        self.mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(
            (1, -1, 1, 1)) * 255
        self.std = torch.as_tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(
            (1, -1, 1, 1)) * 255

    def run_single(self, img_bytes):
        with self.torch_cuda_device:
            decoder = self._decoder_queue.get()
            try:
                img = decoder.decode(img_bytes)
                torch_tensor = img.permute([2, 0, 1]).float()
                resized_img = self.transform(torch_tensor)
                # [1, C, H, W]
                return resized_img.unsqueeze(0)
            except Exception as e:
                print(f'NvJPEG Exception {e}')
                cv2_img = cv2.imdecode(np.asarray(bytearray(img_bytes)), cv2.IMREAD_COLOR)
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                torch_tensor = torch.from_numpy(cv2_img)
                torch_tensor = torch_tensor.permute([2, 0, 1]).float()
                resized_img = self.transform(torch_tensor)
                # [1, C, H, W]
                resized_img = resized_img.unsqueeze(0).cuda(device=self.device)
                return resized_img
            finally:
                self._decoder_queue.put(decoder)

    def run(self, batch_image_bytes):
        images = self.pool.map(self.run_single, batch_image_bytes)
        image_tensor = torch.cat(images, dim=0)
        image_tensor.sub_(self.mean).div_(self.std)
        return image_tensor


class GpuPreprocessIterator(object):

    def __init__(self, data_loader, transform, device_id=0, cpu_threads=4, image_index=[0]):
        self.data_loader = data_loader
        # self.stream = torch.cuda.Stream(device_id)
        self.stream = torch.cuda.current_stream(device_id)
        self.image_decode_transform = BatchDecodeTransform(cpu_threads,
                                                           transform=transform,
                                                           device_id=device_id,
                                                           stream=self.stream)
        self.image_idx = image_index
        self._data_queue = queue.Queue(maxsize=5)

    def _enqueue(self):
        while True:
            try:
                data = next(self.iter)
            except StopIteration:
                return
            with torch.no_grad():
                with torch.cuda.stream(self.stream):
                    column_num = len(data[0])
                    columns = []
                    for _ in range(column_num):
                        columns.append([])

                    # row -> column
                    for row in data:
                        for idx, f in enumerate(row):
                            columns[idx].append(f)

                    result = []
                    for idx, col in enumerate(columns):
                        if idx in self.image_idx:
                            image_tensor = self.image_decode_transform.run(col)
                            result.append(image_tensor)
                        else:
                            if isinstance(col[0], np.ndarray):
                                t = torch.stack(col, dim=0).cuda(non_blocking=True)
                            else:
                                # numerical
                                t = torch.as_tensor(col).cuda(non_blocking=True)
                            result.append(t)
                    self._data_queue.put(result, block=True)

    def __iter__(self):
        self.iter = iter(self.data_loader)
        self._thread = threading.Thread(target=self._enqueue, name='enqueue')
        self._thread.start()
        print(f"start enqueue. Loaded data, queue size {self._data_queue.qsize()}")

        return self

    def __len__(self):
        return len(self.data_loader)

    def __next__(self):
        try:
            result = self._data_queue.get(block=True, timeout=30)
        except queue.Empty:
            self._thread.join(timeout=30)
            raise StopIteration
        return result
