//
// Created by liupeng on 2021/3/23.
//
#include "torchnvjpeg.h"

namespace torchnvjpeg {
static void* ctypes_void_ptr(const py::object& object) {
  PyObject* p_ptr = object.ptr();
  if (!PyObject_HasAttr(p_ptr, PyUnicode_FromString("value"))) {
    return nullptr;
  }
  PyObject* ptr_as_int = PyObject_GetAttr(p_ptr, PyUnicode_FromString("value"));
  if (ptr_as_int == Py_None) {
    return nullptr;
  }
  void* ptr = PyLong_AsVoidPtr(ptr_as_int);
  return ptr;
}

Decoder::Decoder(
    size_t device_padding,
    size_t host_padding,
    bool gpu_huffman,
    int device_id,
    size_t max_image_size,
    cudaStream_t stream)
    : device_allocator{&dev_malloc, &dev_free},
      pinned_allocator{&host_malloc, &host_free},
      device_id(device_id),
      max_image_size(max_image_size),
      cuda_stream(stream) {
  /**
   * using pytorch:  torch.cuda.set_device
   * torch version 1.8
   * https://github.com/Quansight/pytorch/commit/3788a42f5e4e16f86fc3d5b2062b20262d71a051
   *  torch::cuda::set_device(device_id);
   */

  CUDA(cudaSetDevice(device_id));

  nvjpegBackend_t backend = NVJPEG_BACKEND_HYBRID;
  if (gpu_huffman) {
    backend = NVJPEG_BACKEND_GPU_HYBRID;
  }

  NVJPEG(nvjpegCreateEx(backend, &device_allocator, &pinned_allocator, NVJPEG_FLAGS_DEFAULT, &handle))
  NVJPEG(nvjpegJpegStateCreate(handle, &state))

  NVJPEG(nvjpegSetDeviceMemoryPadding(device_padding, handle))
  NVJPEG(nvjpegSetPinnedMemoryPadding(host_padding, handle))
}

Decoder::Decoder(
    size_t device_padding,
    size_t host_padding,
    bool gpu_huffman,
    int device_id,
    size_t max_image_size,
    const py::object& py_cuda_stream) {
  cudaStream_t stream = py_cuda_stream.is_none() ? c10::cuda::getDefaultCUDAStream(device_id).stream()
                                                 : static_cast<cudaStream_t>(ctypes_void_ptr(py_cuda_stream));

  new (this) Decoder(device_padding, host_padding, gpu_huffman, device_id, max_image_size, stream);
}

Decoder::Decoder(
    size_t device_padding,
    size_t host_padding,
    bool gpu_huffman,
    int device_id,
    int bath_size,
    int max_cpu_threads,
    size_t max_image_size,
    const py::object& py_cuda_stream) {
  cudaStream_t stream = py_cuda_stream.is_none() ? c10::cuda::getCurrentCUDAStream(device_id).stream()
                                                 : static_cast<cudaStream_t>(ctypes_void_ptr(py_cuda_stream));
  new (this)
      Decoder(device_padding, host_padding, gpu_huffman, device_id, bath_size, max_cpu_threads, max_image_size, stream);
}

Decoder::Decoder(
    size_t device_padding,
    size_t host_padding,
    bool gpu_huffman,
    int device_id,
    int bath_size,
    int max_cpu_threads,
    size_t max_image_size,
    cudaStream_t stream)
    : batch_size(bath_size), max_cpu_threads(max_cpu_threads) {
  new (this) Decoder(device_padding, host_padding, gpu_huffman, device_id, max_image_size, stream);
  NVJPEG(nvjpegDecodeBatchedInitialize(handle, state, batch_size, max_cpu_threads, NVJPEG_OUTPUT_RGBI));
}

Decoder::~Decoder() {
  nvjpegJpegStateDestroy(state);
  nvjpegDestroy(handle);
}

int Decoder::get_device_id() const {
  return device_id;
}

torch::Tensor Decoder::decode(const std::string& data, bool stream_sync = true) {
  const auto* blob = (const unsigned char*)data.data();
  int nComponents;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  NVJPEG(nvjpegGetImageInfo(handle, blob, data.length(), &nComponents, &subsampling, widths, heights));

  if (!SupportedSubsampling(subsampling)) {
    throw std::invalid_argument("nvjpeg: not supported subsampling");
  }

  int h = heights[0];
  int w = widths[0];

  size_t image_size = h * w * 3;
  if (max_image_size < image_size) {
    std::ostringstream ss;
    ss << "image too large: " << image_size << " > max image size " << max_image_size;
    throw std::invalid_argument(ss.str());
  }

  auto options = at::TensorOptions()
                     .device(torch::kCUDA, device_id)
                     .dtype(torch::kUInt8)
                     .layout(torch::kStrided)
                     .requires_grad(false);
  auto image_tensor = at::empty({h, w, 3}, options, at::MemoryFormat::Contiguous);
  auto* image = image_tensor.data_ptr<unsigned char>();

  nvjpegImage_t nv_image;
  for (size_t i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
    nv_image.channel[i] = nullptr;
    nv_image.pitch[i] = 0;
  }
  nv_image.channel[0] = image;
  nv_image.pitch[0] = 3 * w;

  NVJPEG(nvjpegDecode(handle, state, blob, data.length(), NVJPEG_OUTPUT_RGBI, &nv_image, cuda_stream))
  if (stream_sync) {
    cudaStreamSynchronize(cuda_stream);
  }

  return image_tensor;
}

std::vector<torch::Tensor> Decoder::batch_decode(const std::vector<std::string>& data_list, bool stream_sync = true) {
  if (data_list.size() != static_cast<unsigned int>(batch_size)) {
    //        batch_size = data_list.size();
    //        NVJPEG(nvjpegDecodeBatchedInitialize(handle, state, batch_size, max_cpu_threads, NVJPEG_OUTPUT_RGBI));

    std::ostringstream ss;
    // to-do: std::format in C++ 20.
    ss << "data_list size " << data_list.size() << " != "
       << "batch_size " << batch_size;
    throw std::invalid_argument(ss.str());
  }

  std::vector<const unsigned char*> raw_inputs;
  std::vector<size_t> image_len_list;
  std::vector<torch::Tensor> tensor_list;
  std::vector<nvjpegImage_t> nv_image_list;

  raw_inputs.reserve(batch_size);
  image_len_list.reserve(batch_size);
  tensor_list.reserve(batch_size);
  nv_image_list.reserve(batch_size);

#ifdef OPENMP
#pragma omp parallel for
#endif
  for (const auto& data : data_list) {
    const auto* blob = (const unsigned char*)data.data();
    raw_inputs.emplace_back(blob);
    image_len_list.emplace_back(data.length());

    int nComponents;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
    NVJPEG(nvjpegGetImageInfo(handle, blob, data.length(), &nComponents, &subsampling, widths, heights))

    if (!SupportedSubsampling(subsampling)) {
      throw std::invalid_argument("nvjpeg: not supported subsampling");
    }

    int h = heights[0];
    int w = widths[0];

    size_t image_size = h * w * 3;
    if (max_image_size < image_size) {
      std::ostringstream ss;
      ss << "image too large: " << image_size << " > max image size " << max_image_size;
      throw std::invalid_argument(ss.str());
    }

    auto image_tensor = torch::empty(
        {h, w, 3},
        torch::TensorOptions()
            .device(torch::kCUDA, device_id)
            .dtype(torch::kUInt8)
            .layout(torch::kStrided)
            .requires_grad(false));
    tensor_list.emplace_back(image_tensor);

    auto* image = image_tensor.data_ptr<unsigned char>();
    nvjpegImage_t nv_image;
    for (size_t i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
      nv_image.channel[i] = nullptr;
      nv_image.pitch[i] = 0;
    }
    nv_image.channel[0] = image;
    nv_image.pitch[0] = 3 * w;

    nv_image_list.emplace_back(nv_image);
  }

  NVJPEG(
      nvjpegDecodeBatched(handle, state, raw_inputs.data(), image_len_list.data(), nv_image_list.data(), cuda_stream));
  if (stream_sync) {
    cudaStreamSynchronize(cuda_stream);
  }

  return tensor_list;
}

#ifdef PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<Decoder> decoder(m, "Decoder");
  decoder
      .def(
          py::init<size_t, size_t, bool, int, size_t, py::object>(),
          py::return_value_policy::take_ownership,
          R"docdelimiter(
                    Initialize nvjpeg decoder.

                    Parameters:
                        device_padding: int, set 0 by default
                        host_padding: int, set 0 by default
                        gpu_huffman: bool, whether to use GPU for Huffman decode, set true by default
                        device_id: int, gpu id, set 0 by default
                        max_image_size: int, maximum image size (h * w * c) to decode, set 3840*2160*3 by default
                        stream: torch.cuda.Stream, if None, using torch.cuda.current_stream()
                )docdelimiter",
          py::arg("device_padding") = 0,
          py::arg("host_padding") = 0,
          py::arg("gpu_huffman") = true,
          py::arg("device_id") = 0,
          py::arg("max_image_size") = 3840 * 2160 * 3,
          py::arg("stream") = py::none())
      .def(
          py::init<size_t, size_t, bool, int, int, int, size_t, py::object>(),
          py::return_value_policy::take_ownership,
          R"docdelimiter(
                    Initialize nvjpeg batch decoder.

                    Parameters:
                        device_padding: int
                        host_padding: int
                        gpu_huffman: bool
                        device_id: int
                        bath_size: int,
                        max_cpu_threads: int
                        max_image_size: int
                        stream: torch.cuda.Stream
                )docdelimiter",
          py::arg("device_padding"),
          py::arg("host_padding"),
          py::arg("gpu_huffman"),
          py::arg("device_id"),
          py::arg("bath_size"),
          py::arg("max_cpu_threads"),
          py::arg("max_image_size"),
          py::arg("stream"))
      .def(
          "decode",
          &Decoder::decode,
          py::call_guard<py::gil_scoped_release>(),
          py::return_value_policy::take_ownership,
          R"docdelimiter(
                    Decode image to torch cuda tensor.

                    Parameters:
                        data: string, image bytes
                        stream_sync: bool, whether to do steam.synchronize()

                    Returns:
                        image cuda tensor in HWC foramt.
                )docdelimiter",
          py::arg("data"),
          py::arg("stream_sync") = true)
      .def(
          "batch_decode",
          &Decoder::batch_decode,
          py::call_guard<py::gil_scoped_release>(),
          py::return_value_policy::take_ownership,
          R"docdelimiter(
                    Decode list of images to list of torch cuda tensor.

                    Parameters:
                        data: List[string], list of image bytes
                        stream_sync: bool, whether to do steam.synchronize()

                    Returns:
                        list of image cuda tensor in HWC foramt.
                )docdelimiter",
          py::arg("data"),
          py::arg("stream_sync") = true)
      .def("get_device_id", &Decoder::get_device_id, py::return_value_policy::take_ownership);
}
#endif
} // namespace torchnvjpeg

inline std::string read_image(const std::string& image_path) {
  std::ifstream instream(image_path, std::ios::in | std::ios::binary);
  std::string data((std::istreambuf_iterator<char>(instream)), std::istreambuf_iterator<char>());
  return data;
}

int main(int argc, const char** argv) {
  std::string image_path = "/home/liupeng/remote/torchnvjpeg/images/cat.jpg";
  if (argc > 1) {
    image_path = argv[1];
  }

  int device_id = 0;
  size_t max_size = 1920 * 1080 * 3;
  auto image_data = read_image(image_path);

  auto d = torchnvjpeg::Decoder(0, 0, true, device_id, max_size, c10::cuda::getDefaultCUDAStream());
  torch::Tensor t = d.decode(image_data);
  std::cout << "single deocde: " << std::endl << t.sizes() << std::endl;

  int batch_size = 4;
  int max_cpu_threads = 4;
  auto batch_decoder = torchnvjpeg::Decoder(
      0, 0, true, device_id, batch_size, max_cpu_threads, max_size, c10::cuda::getDefaultCUDAStream());

  std::vector<std::string> data_list;
  data_list.reserve(batch_size);
  for (int i = 0; i < batch_size; i++) {
    data_list.emplace_back(image_data);
  }

  auto tensor_list = batch_decoder.batch_decode(data_list);

  std::cout << "batch decode:" << std::endl;
  for (auto& t : tensor_list) {
    std::cout << t.sizes() << std::endl;
  }

  printf("Done\n");
  return 0;
}