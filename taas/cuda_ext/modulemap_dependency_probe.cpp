#include <onnxruntime_cxx_api.h>
#include <TROOT.h>

#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

namespace {

py::dict dependency_versions() {
  py::dict versions;
  versions["onnxruntime"] = std::string(Ort::GetVersionString());
  versions["root"] = std::string(gROOT->GetVersion());
  return versions;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dependency_versions", &dependency_versions,
        "Return linked ONNX Runtime and ROOT versions.");
}
