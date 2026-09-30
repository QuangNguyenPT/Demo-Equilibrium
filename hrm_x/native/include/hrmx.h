/* hrmx native backend: C ABI for the measured hot paths of HRM-X (spec v0.4 §23).
 *
 * The Python modules are the blueprint and the reference semantics; this library executes the
 * same operations faster and is tested for bit-identical results against them:
 *   grid   : connected components + one object-level program step   (hrm_x/domains/grid.py)
 *   egraph : union-find, hash-consing, congruence rebuild, e-matching, explanations
 *            (hrm_x/workspace.py + hrm_x/rules.py)
 * Arrays are passed as raw pointers to caller-owned NumPy buffers (no copies, no ownership).
 */
#ifndef HRMX_H
#define HRMX_H

#include <stdint.h>

#if defined(_WIN32)
#define HX_API __declspec(dllexport)
#else
#define HX_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

HX_API int32_t hx_version(void);

/* ---------- grid ---------- */

/* 4-connected same-colour components (0 = background). labels[h*w] gets 0 or the object number
 * 1..n in reading order of each object's first cell. Returns n. */
HX_API int32_t hx_components(const int32_t* grid, int32_t h, int32_t w, int32_t* labels);

/* One object-level step. sel: 0 all, 1 colour == sel_color, 2 largest, 3 smallest.
 * act: 0 keep, 1 delete, 2 recolour to a1, 3 move by (a1, a2), 4 flip (a1 = 0 horizontal, 1 vertical).
 * Writes out[h*w]. Returns 0 on success, 1 if invalid: an object leaves the grid; or, when strict >= 1,
 * the selector is ambiguous or two objects collide; or, when strict == 1 (a demo), it selects nothing. */
HX_API int32_t hx_object_step(const int32_t* grid, int32_t h, int32_t w, int32_t sel, int32_t sel_color,
                              int32_t act, int32_t a1, int32_t a2, int32_t strict, int32_t* out);

/* ---------- e-graph ---------- */

typedef struct hx_egraph hx_egraph;

HX_API hx_egraph* hx_eg_new(int32_t const_op);
HX_API void hx_eg_free(hx_egraph* g);
HX_API int32_t hx_eg_add(hx_egraph* g, int32_t op, const int32_t* args, int32_t nargs, int64_t payload);
HX_API int32_t hx_eg_find(hx_egraph* g, int32_t id);
/* 1 merged, 0 already equal, -1 refused: both classes hold different constants (nothing changes). */
HX_API int32_t hx_eg_union(hx_egraph* g, int32_t a, int32_t b, int32_t why,
                           const int32_t* premises, int32_t npremises);
/* Congruence closure. attempts/merges out-params; refused congruences are queued as conflicts. */
HX_API void hx_eg_rebuild(hx_egraph* g, int32_t* attempts, int32_t* merges);
HX_API int32_t hx_eg_take_conflicts(hx_egraph* g, int32_t* pairs, int32_t cap);
HX_API int32_t hx_eg_lookup(hx_egraph* g, int32_t op, const int32_t* args, int32_t nargs, int64_t payload);
HX_API int32_t hx_eg_const_node(hx_egraph* g, int32_t cls);
HX_API int32_t hx_eg_num_nodes(hx_egraph* g);
HX_API int32_t hx_eg_classes(hx_egraph* g, int32_t* out, int32_t cap);
/* Members of a class (e-node ids, in membership order). */
HX_API int32_t hx_eg_members(hx_egraph* g, int32_t cls, int32_t* out, int32_t cap);
/* One e-node: op, payload, and its arguments (canonical if canonical != 0). Returns nargs. */
HX_API int32_t hx_eg_node(hx_egraph* g, int32_t id, int32_t* op, int64_t* payload, int32_t* args,
                          int32_t cap, int32_t canonical);
/* Reasons (why ids >= 0) on the proof path a ~ b, congruences and premises expanded. */
HX_API int32_t hx_eg_explain(hx_egraph* g, int32_t a, int32_t b, int32_t* out, int32_t cap);
/* Snapshot "classes containing op" (sorted), used by the next ematch calls (one per saturation round). */
HX_API void hx_eg_snapshot_index(hx_egraph* g);
/* E-matching. Pattern preorder code: var = [0, slot]; any constant = [1]; node = [2, op, nargs, kids...].
 * Each match writes 1 + nvars ints: anchor e-node id (class id for a bare-variable pattern), then the
 * class bound to each slot. Root candidates come from the last snapshot if use_index != 0, else from
 * all current classes. Returns the number of matches (the caller retries with more room at cap). */
HX_API int32_t hx_eg_ematch(hx_egraph* g, const int32_t* code, int32_t ncode, int32_t nvars,
                            int32_t use_index, int32_t* out, int32_t cap);

/* ---------- construct (hrm_x/construct.py) ---------- */

/* Bottom-up enumeration of postfix expressions over {x, y, 1, 2, +, -, *, /, ifgt} up to max_size
 * tokens, observationally distinct on the n probe points (px, py). Writes the flattened tokens,
 * offsets[count + 1] into them, and count * n probe values, in the Python reference order.
 * Returns count, or -1 if cap_tokens or cap_exprs is too small. */
HX_API int32_t hx_enumerate(int32_t max_size, const double* px, const double* py, int32_t n,
                            int32_t* out_tokens, int32_t cap_tokens, int32_t* out_offsets,
                            double* out_values, int32_t cap_exprs);

#ifdef __cplusplus
}
#endif
#endif
