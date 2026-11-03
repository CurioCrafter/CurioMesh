// Six orthographic depth pictures of a triangle mesh, the compute twin of `geom::pictures_along`.
// One thread per triangle, looping over the three axes. Pass `depth` keeps the nearest depth per pixel as an
// atomic min of its f32 bits (positive floats order like their bits). Pass `ids` then keeps, among the triangles
// whose depth bits equal that minimum, the lowest index: the CPU path's tie-break (strictly nearer wins, ascending ids).

struct Params {
    pixels: u32,
    tris: u32,
    stride: u32,
    half: f32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> points: array<f32>;
@group(0) @binding(2) var<storage, read> corners: array<u32>;
@group(0) @binding(3) var<storage, read_write> depths: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> ids: array<atomic<u32>>;

fn point(i: u32) -> vec3<f32> {
    return vec3<f32>(points[i * 3u], points[i * 3u + 1u], points[i * 3u + 2u]);
}

fn run(id: u32, second: bool) {
    let t = vec3<u32>(corners[id * 3u], corners[id * 3u + 1u], corners[id * 3u + 2u]);
    let p0 = point(t.x);
    let p1 = point(t.y);
    let p2 = point(t.z);
    let n = params.pixels;
    let cells = n * n;
    let last = f32(n) - 1.0;
    for (var axis = 0u; axis < 3u; axis++) {
        let u = (axis + 1u) % 3u;
        let v = (axis + 2u) % 3u;
        let a = vec2<f32>((p0[u] + params.half) * params.scale - 0.5, (p0[v] + params.half) * params.scale - 0.5);
        let b = vec2<f32>((p1[u] + params.half) * params.scale - 0.5, (p1[v] + params.half) * params.scale - 0.5) - a;
        let c = vec2<f32>((p2[u] + params.half) * params.scale - 0.5, (p2[v] + params.half) * params.scale - 0.5) - a;
        let area = b.x * c.y - c.x * b.y;
        if area == 0.0 {
            continue;
        }
        // A pixel center within a thousandth of a pixel outside an edge still counts, so shared edges leave no gaps.
        let tol = 0.001 / abs(area);
        let tol0 = tol * length(c - b);
        let tol1 = tol * length(c);
        let tol2 = tol * length(b);
        let lo = min(vec2<f32>(0.0), min(b, c)) + a;
        let hi = max(vec2<f32>(0.0), max(b, c)) + a;
        let r0 = u32(max(ceil(lo.x - 0.01), 0.0));
        let r1 = u32(min(floor(hi.x + 0.01), last));
        let c0 = u32(max(ceil(lo.y - 0.01), 0.0));
        let c1 = u32(min(floor(hi.y + 0.01), last));
        if hi.x < -0.01 || hi.y < -0.01 || lo.x > last + 0.01 || lo.y > last + 0.01 {
            continue;
        }
        let z0 = p0[axis];
        let z1 = p1[axis];
        let z2 = p2[axis];
        for (var row = r0; row <= r1; row++) {
            for (var col = c0; col <= c1; col++) {
                let p = vec2<f32>(f32(row), f32(col)) - a;
                let w1 = (p.x * c.y - c.x * p.y) / area;
                let w2 = (b.x * p.y - p.x * b.y) / area;
                let w0 = 1.0 - w1 - w2;
                if w0 < -tol0 || w1 < -tol1 || w2 < -tol2 {
                    continue;
                }
                let z = w0 * z0 + w1 * z1 + (1.0 - w0 - w1) * z2;
                let cell = row * n + col;
                let nearer = 2.0 * params.half - z;
                let farther = z + 2.0 * params.half;
                if nearer >= 0.0 {
                    let at = (axis * 2u) * cells + cell;
                    if second {
                        if bitcast<u32>(nearer) == atomicLoad(&depths[at]) {
                            atomicMin(&ids[at], id);
                        }
                    } else {
                        atomicMin(&depths[at], bitcast<u32>(nearer));
                    }
                }
                if farther >= 0.0 {
                    let at = (axis * 2u + 1u) * cells + cell;
                    if second {
                        if bitcast<u32>(farther) == atomicLoad(&depths[at]) {
                            atomicMin(&ids[at], id);
                        }
                    } else {
                        atomicMin(&depths[at], bitcast<u32>(farther));
                    }
                }
            }
        }
    }
}

@compute @workgroup_size(64)
fn depth(@builtin(global_invocation_id) g: vec3<u32>) {
    let id = g.x + g.y * params.stride;
    if id < params.tris {
        run(id, false);
    }
}

@compute @workgroup_size(64)
fn pick(@builtin(global_invocation_id) g: vec3<u32>) {
    let id = g.x + g.y * params.stride;
    if id < params.tris {
        run(id, true);
    }
}
