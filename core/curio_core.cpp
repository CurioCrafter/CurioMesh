#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_set>
#include <unordered_map>

namespace curio_core_ns {

struct Vec3 { float x, y, z; };
struct Tri { int a, b, c; };
struct Quad { int a, b, c, d; };

inline float dot3(const Vec3 &u, const Vec3 &v) { return u.x*v.x + u.y*v.y + u.z*v.z; }
inline Vec3 cross3(const Vec3 &u, const Vec3 &v) {
  return {u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x};
}
inline Vec3 sub3(const Vec3 &u, const Vec3 &v) { return {u.x-v.x, u.y-v.y, u.z-v.z}; }
inline float norm3(const Vec3 &u) { return std::sqrt(std::max(0.0f, dot3(u,u))); }

inline long long pack_edge_key(int a, int b) {
  const int u = a < b ? a : b;
  const int v = a < b ? b : a;
  return (static_cast<long long>(u) << 32) | static_cast<unsigned int>(v);
}

// Very small helper: Greedy pair tri-tri via shared edge, with a crude feature angle filter and blocked edges.
static void greedy_pair_tris(
  const std::vector<Vec3> &verts,
  const std::vector<Tri> &tris,
  float feature_angle_rad,
  const std::unordered_set<long long> &blocked_edges,
  std::vector<Quad> &out_quads,
  std::vector<Tri> &out_tris)
{
  const float cos_thresh = std::cos(feature_angle_rad);
  struct EdgeInfo { int face_index; };
  std::unordered_map<long long, EdgeInfo> edge_map; edge_map.reserve(tris.size()*3);

  std::vector<char> used(tris.size(), 0);
  out_quads.clear(); out_tris.clear();

  // Build face normals and edge map
  std::vector<Vec3> face_n(tris.size());
  for (size_t fi=0; fi<tris.size(); ++fi) {
    const auto &t = tris[fi];
    const Vec3 &a = verts[t.a], &b = verts[t.b], &c = verts[t.c];
    Vec3 n = cross3(sub3(b,a), sub3(c,a));
    const float inv = 1.0f / (norm3(n) + 1e-12f);
    face_n[fi] = {n.x*inv, n.y*inv, n.z*inv};
    long long k0 = pack_edge_key(t.a, t.b);
    long long k1 = pack_edge_key(t.b, t.c);
    long long k2 = pack_edge_key(t.c, t.a);
    if (!edge_map.count(k0)) edge_map[k0] = {(int)fi, 0};
    if (!edge_map.count(k1)) edge_map[k1] = {(int)fi, 1};
    if (!edge_map.count(k2)) edge_map[k2] = {(int)fi, 2};
  }

  // Greedy pairing
  for (size_t fi=0; fi<tris.size(); ++fi) {
    if (used[fi]) continue;
    const auto &t = tris[fi];
    const int e0[2] = {t.a, t.b};
    const int e1[2] = {t.b, t.c};
    const int e2[2] = {t.c, t.a};
    const int* edges[3] = {e0, e1, e2};
    bool paired = false;
    for (int ei=0; ei<3 && !paired; ++ei) {
      long long k = pack_edge_key(edges[ei][0], edges[ei][1]);
      if (blocked_edges.find(k) != blocked_edges.end()) continue; // respect blocked edges
      // find the other face sharing this edge
      int other = -1;
      for (size_t fj=0; fj<tris.size(); ++fj) {
        if ((int)fj == (int)fi) continue;
        const auto &ot = tris[fj];
        if ((ot.a==edges[ei][0]||ot.b==edges[ei][0]||ot.c==edges[ei][0]) &&
            (ot.a==edges[ei][1]||ot.b==edges[ei][1]||ot.c==edges[ei][1])) {
          other = (int)fj; break;
        }
      }
      if (other<0 || used[other]) continue;
      float c = dot3(face_n[fi], face_n[other]);
      if (c < cos_thresh) continue; // too sharp, keep as triangles
      // Build quad by ordering verts around the shared edge
      int a0 = t.a, b0 = t.b, c0 = t.c;
      int va = edges[ei][0], vb = edges[ei][1];
      // the opposite vertices
      int opp1 = (a0!=va && a0!=vb)? a0 : (b0!=va && b0!=vb? b0 : c0);
      const auto &ot = tris[other];
      int oa0=ot.a, ob0=ot.b, oc0=ot.c;
      int opp2 = (oa0!=va && oa0!=vb)? oa0 : (ob0!=va && ob0!=vb? ob0 : oc0);
      out_quads.push_back({opp1, va, opp2, vb});
      used[fi]=used[other]=1; paired=true;
    }
    if (!paired) {
      out_tris.push_back(tris[fi]);
      used[fi]=1;
    }
  }
}

} // namespace curio_core_ns


