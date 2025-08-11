#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include "curio_core.cpp"

namespace py = pybind11;

static py::dict remesh(
    py::array_t<float, py::array::c_style | py::array::forcecast> verts,
    py::array_t<int, py::array::c_style | py::array::forcecast> faces,
    int target_faces,
    float feature_weight,
    float adaptivity,
    const std::string &engine,
    bool use_vdb,
    float vdb_voxel_size,
    float feature_angle_deg,
    int smooth_iters,
    float smooth_lambda) {
  // Minimal native: naive greedy tri pairing into quads, respecting feature angle.
  py::buffer_info vinfo = verts.request();
  py::buffer_info finfo = faces.request();
  if (vinfo.ndim != 2 || vinfo.shape[1] != 3) {
    throw std::runtime_error("verts must be (N,3) float32");
  }
  if (finfo.ndim != 2 || finfo.shape[1] != 3) {
    throw std::runtime_error("faces must be (M,3) int32 triangles");
  }
  const int nv = static_cast<int>(vinfo.shape[0]);
  const int nf = static_cast<int>(finfo.shape[0]);
  const float *vptr = static_cast<float*>(vinfo.ptr);
  const int *fptr = static_cast<int*>(finfo.ptr);

  std::vector<curio_core_ns::Vec3> V(nv);
  for (int i=0;i<nv;++i) {
    V[i] = {vptr[3*i+0], vptr[3*i+1], vptr[3*i+2]};
  }
  std::vector<curio_core_ns::Tri> F(nf);
  for (int i=0;i<nf;++i) {
    F[i] = {fptr[3*i+0], fptr[3*i+1], fptr[3*i+2]};
  }

  // blocked edges are not yet provided from Blender; keep empty for now.
  std::unordered_set<long long> blocked;
  std::vector<curio_core_ns::Quad> quads;
  std::vector<curio_core_ns::Tri>  tris_out;
  curio_core_ns::greedy_pair_tris(V, F, feature_angle_deg * float(M_PI/180.0), blocked, quads, tris_out);

  // Pack outputs
  py::array_t<float> outV({nv, 3});
  auto oV = outV.mutable_unchecked<2>();
  for (int i=0;i<nv;++i) {
    oV(i,0) = V[i].x; oV(i,1) = V[i].y; oV(i,2) = V[i].z;
  }
  py::array_t<int> outQ({(int)quads.size(), 4});
  auto oQ = outQ.mutable_unchecked<2>();
  for (int i=0;i<(int)quads.size(); ++i) {
    oQ(i,0)=quads[i].a; oQ(i,1)=quads[i].b; oQ(i,2)=quads[i].c; oQ(i,3)=quads[i].d;
  }
  py::array_t<int> outT({(int)tris_out.size(), 3});
  auto oT = outT.mutable_unchecked<2>();
  for (int i=0;i<(int)tris_out.size(); ++i) {
    oT(i,0)=tris_out[i].a; oT(i,1)=tris_out[i].b; oT(i,2)=tris_out[i].c;
  }
  py::dict out;
  out["verts"] = outV;
  out["quads"] = outQ;
  out["tris"]  = outT;
  return out;
}

PYBIND11_MODULE(curio_core, m) {
  m.doc() = "CurioMesh native core (stub)";
  m.def("remesh", &remesh, py::arg("verts"), py::arg("faces"),
        py::arg("target_faces"), py::arg("feature_weight"), py::arg("adaptivity"),
        py::arg("engine"), py::arg("use_vdb"), py::arg("vdb_voxel_size"),
        py::arg("feature_angle_deg"), py::arg("smooth_iters"), py::arg("smooth_lambda"));
}


