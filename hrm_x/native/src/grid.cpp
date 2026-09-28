// Grid perception and object-level steps; reference semantics: hrm_x/domains/grid.py (objects, apply_step).
#include "hrmx.h"

#include <algorithm>
#include <vector>

namespace {

struct Object {
    int32_t color = 0;
    std::vector<int32_t> cells;  // r * w + c
    int32_t r0 = 0, c0 = 0, r1 = 0, c1 = 0;
};

int32_t label(const int32_t* grid, int32_t h, int32_t w, int32_t* labels, std::vector<Object>* objects) {
    std::fill(labels, labels + h * w, 0);
    std::vector<int32_t> stack;
    int32_t n = 0;
    for (int32_t r = 0; r < h; ++r) {
        for (int32_t c = 0; c < w; ++c) {
            const int32_t start = r * w + c;
            if (grid[start] == 0 || labels[start] != 0) continue;
            ++n;
            Object obj;
            obj.color = grid[start];
            obj.r0 = obj.r1 = r;
            obj.c0 = obj.c1 = c;
            labels[start] = n;
            stack.assign(1, start);
            while (!stack.empty()) {
                const int32_t cell = stack.back();
                stack.pop_back();
                obj.cells.push_back(cell);
                const int32_t y = cell / w, x = cell % w;
                obj.r0 = std::min(obj.r0, y); obj.r1 = std::max(obj.r1, y);
                obj.c0 = std::min(obj.c0, x); obj.c1 = std::max(obj.c1, x);
                const int32_t dy[4] = {1, -1, 0, 0}, dx[4] = {0, 0, 1, -1};
                for (int k = 0; k < 4; ++k) {
                    const int32_t ny = y + dy[k], nx = x + dx[k];
                    if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                    const int32_t next = ny * w + nx;
                    if (labels[next] == 0 && grid[next] == obj.color) {
                        labels[next] = n;
                        stack.push_back(next);
                    }
                }
            }
            if (objects) objects->push_back(std::move(obj));
        }
    }
    return n;
}

}  // namespace

extern "C" {

HX_API int32_t hx_version(void) { return 1; }

HX_API int32_t hx_components(const int32_t* grid, int32_t h, int32_t w, int32_t* labels) {
    return label(grid, h, w, labels, nullptr);
}

HX_API int32_t hx_object_step(const int32_t* grid, int32_t h, int32_t w, int32_t sel, int32_t sel_color,
                              int32_t act, int32_t a1, int32_t a2, int32_t strict, int32_t* out) {
    std::vector<int32_t> labels(static_cast<size_t>(h) * w);
    std::vector<Object> objs;
    label(grid, h, w, labels.data(), &objs);

    std::vector<char> mask(objs.size(), 0);
    bool ambiguous = false;
    if (sel == 0) {
        std::fill(mask.begin(), mask.end(), 1);
    } else if (sel == 1) {
        for (size_t i = 0; i < objs.size(); ++i) mask[i] = objs[i].color == sel_color;
    } else if (!objs.empty()) {
        size_t target = objs[0].cells.size();
        for (const auto& o : objs)
            target = sel == 2 ? std::max(target, o.cells.size()) : std::min(target, o.cells.size());
        int32_t count = 0;
        for (size_t i = 0; i < objs.size(); ++i) {
            mask[i] = objs[i].cells.size() == target;
            count += mask[i];
        }
        ambiguous = count > 1;
    }
    bool any = std::any_of(mask.begin(), mask.end(), [](char m) { return m != 0; });
    if (strict && (ambiguous || (strict == 1 && !any))) return 1;   // 1: demo (must select), 2: test

    std::fill(out, out + h * w, 0);
    std::vector<int32_t> placed(static_cast<size_t>(h) * w, -1);
    auto paint = [&](int32_t r, int32_t c, int32_t color) -> bool {
        if (r < 0 || r >= h || c < 0 || c >= w) return false;
        const int32_t cell = r * w + c;
        if (placed[cell] != -1 && placed[cell] != color && strict) return false;
        placed[cell] = color;
        out[cell] = color;
        return true;
    };
    for (size_t i = 0; i < objs.size(); ++i) {
        const Object& o = objs[i];
        const bool chosen = mask[i] != 0;
        if (act == 0) {                                       // keep: only the selected survive
            if (!chosen) continue;
            for (int32_t cell : o.cells) if (!paint(cell / w, cell % w, o.color)) return 1;
            continue;
        }
        if (!chosen) {
            for (int32_t cell : o.cells) if (!paint(cell / w, cell % w, o.color)) return 1;
            continue;
        }
        for (int32_t cell : o.cells) {
            const int32_t r = cell / w, c = cell % w;
            bool ok = true;
            switch (act) {
                case 1: break;                                           // delete
                case 2: ok = paint(r, c, a1); break;                     // recolour
                case 3: ok = paint(r + a1, c + a2, o.color); break;      // move
                case 4: ok = a1 == 0 ? paint(r, o.c1 - (c - o.c0), o.color)
                                     : paint(o.r1 - (r - o.r0), c, o.color); break;
                default: return 1;
            }
            if (!ok) return 1;
        }
    }
    return 0;
}

}  // extern "C"
