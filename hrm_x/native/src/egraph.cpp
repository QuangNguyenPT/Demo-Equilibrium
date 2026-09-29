// E-graph core; reference semantics: hrm_x/workspace.py (Workspace) and hrm_x/rules.py (_match, matches).
// Every iteration order mirrors the Python reference (members lists, insertion-ordered memo, sorted op
// index, BFS over proof edges in insertion order), so both backends reach identical states.
#include "hrmx.h"

#include <algorithm>
#include <deque>
#include <map>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Key {
    int32_t op;
    int64_t payload;
    std::vector<int32_t> args;
    bool operator==(const Key& o) const { return op == o.op && payload == o.payload && args == o.args; }
};

struct KeyHash {
    size_t operator()(const Key& k) const {
        size_t h = std::hash<int64_t>()(k.payload) * 1000003u ^ std::hash<int32_t>()(k.op);
        for (int32_t a : k.args) h = h * 1000003u ^ std::hash<int32_t>()(a);
        return h;
    }
};

struct Node {
    int32_t op;
    int64_t payload;
    std::vector<int32_t> args;  // ids as given at creation (valid ids forever)
};

struct Edge {
    int32_t to;
    int32_t why;     // >= 0: caller's reason id; -1: congruence between e-nodes cx and cy
    int32_t cx, cy;
};

using Bind = std::vector<int32_t>;

}  // namespace

struct hx_egraph {
    int32_t const_op;
    std::vector<int32_t> parent;
    std::vector<Node> nodes;
    std::vector<std::vector<int32_t>> members;
    std::vector<int32_t> const_of;                      // root -> const e-node, -1 if none
    std::vector<std::pair<Key, int32_t>> memo_order;
    std::unordered_map<Key, int32_t, KeyHash> memo;
    std::vector<std::vector<Edge>> proof;
    std::map<int32_t, std::vector<std::pair<int32_t, int32_t>>> premises;
    std::vector<std::pair<int32_t, int32_t>> conflicts;
    std::map<int32_t, std::vector<int32_t>> index;       // op -> sorted classes (snapshot)
    bool has_index = false;

    int32_t find(int32_t i) {
        int32_t root = i;
        while (parent[root] != root) root = parent[root];
        while (parent[i] != root) {
            const int32_t next = parent[i];
            parent[i] = root;
            i = next;
        }
        return root;
    }

    Key canon(int32_t op, const int32_t* args, int32_t n, int64_t payload) {
        Key k{op, payload, std::vector<int32_t>(n)};
        for (int32_t i = 0; i < n; ++i) k.args[i] = find(args[i]);
        return k;
    }

    int32_t add(int32_t op, const int32_t* args, int32_t n, int64_t payload) {
        Key k = canon(op, args, n, payload);
        auto hit = memo.find(k);
        if (hit != memo.end()) return find(hit->second);
        const int32_t id = static_cast<int32_t>(parent.size());
        parent.push_back(id);
        nodes.push_back(Node{op, payload, k.args});
        members.push_back({id});
        const_of.push_back(op == const_op ? id : -1);
        proof.emplace_back();
        memo.emplace(k, id);
        memo_order.emplace_back(std::move(k), id);
        return id;
    }

    // 1 merged, 0 same, -1 refused (distinct constants)
    int32_t unite(int32_t a, int32_t b, int32_t why, int32_t cx, int32_t cy) {
        int32_t ra = find(a), rb = find(b);
        if (ra == rb) return 0;
        int32_t ca = const_of[ra], cb = const_of[rb];
        if (ca >= 0 && cb >= 0 && nodes[ca].payload != nodes[cb].payload) return -1;
        proof[a].push_back(Edge{b, why, cx, cy});
        proof[b].push_back(Edge{a, why, cx, cy});
        if (members[ra].size() < members[rb].size()) {
            std::swap(ra, rb);
            std::swap(ca, cb);
        }
        parent[rb] = ra;
        members[ra].insert(members[ra].end(), members[rb].begin(), members[rb].end());
        members[rb].clear();
        if (ca < 0 && cb >= 0) const_of[ra] = cb;
        const_of[rb] = -1;
        return 1;
    }

    void rebuild(int32_t* attempts, int32_t* merges) {
        *attempts = 0;
        *merges = 0;
        bool changed = true;
        while (changed) {
            changed = false;
            std::unordered_map<Key, int32_t, KeyHash> fresh;
            std::vector<std::pair<Key, int32_t>> order;
            for (auto& entry : memo_order) {
                Key k = canon(entry.first.op, entry.first.args.data(), static_cast<int32_t>(entry.first.args.size()),
                              entry.first.payload);
                const int32_t i = entry.second;
                auto hit = fresh.find(k);
                if (hit != fresh.end()) {
                    const int32_t j = hit->second;
                    if (find(j) != find(i)) {
                        ++*attempts;
                        const int32_t r = unite(j, i, -1, j, i);
                        if (r == 1) {
                            ++*merges;
                            changed = true;
                        } else if (r == -1) {
                            conflicts.emplace_back(j, i);
                        }
                    }
                } else {
                    fresh.emplace(k, i);
                    order.emplace_back(std::move(k), i);
                }
            }
            memo = std::move(fresh);
            memo_order = std::move(order);
        }
    }

    void explain(int32_t a, int32_t b, std::set<std::pair<int32_t, int32_t>>& seen, std::vector<int32_t>& out) {
        if (a == b) return;
        const std::pair<int32_t, int32_t> key{std::min(a, b), std::max(a, b)};
        if (!seen.insert(key).second) return;
        std::unordered_map<int32_t, std::pair<int32_t, const Edge*>> back;
        back[a] = {a, nullptr};
        std::deque<int32_t> queue{a};
        while (!queue.empty() && !back.count(b)) {
            const int32_t u = queue.front();
            queue.pop_front();
            for (const Edge& e : proof[u]) {
                if (!back.count(e.to)) {
                    back[e.to] = {u, &e};
                    queue.push_back(e.to);
                }
            }
        }
        if (!back.count(b)) return;
        int32_t node = b;
        while (node != a) {
            const auto& step = back[node];
            const Edge* e = step.second;
            if (e->why == -1) {
                const auto& xs = nodes[e->cx].args;
                const auto& ys = nodes[e->cy].args;
                for (size_t k = 0; k < xs.size() && k < ys.size(); ++k) explain(xs[k], ys[k], seen, out);
            } else {
                out.push_back(e->why);
                auto p = premises.find(e->why);
                if (p != premises.end())
                    for (const auto& pr : p->second) explain(pr.first, pr.second, seen, out);
            }
            node = step.first;
        }
    }

    // ---- e-matching (mirrors rules._match) ----
    void match(const int32_t* code, int32_t& pc, int32_t cls, const std::vector<Bind>& in, std::vector<Bind>& out) {
        const int32_t kind = code[pc];
        if (kind == 0) {
            const int32_t slot = code[pc + 1];
            pc += 2;
            for (const Bind& s : in) {
                if (s[slot] < 0) {
                    Bind t = s;
                    t[slot] = find(cls);
                    out.push_back(std::move(t));
                } else if (find(s[slot]) == find(cls)) {
                    out.push_back(s);
                }
            }
            return;
        }
        if (kind == 1) {
            pc += 1;
            if (const_of[find(cls)] >= 0) out.insert(out.end(), in.begin(), in.end());
            return;
        }
        const int32_t op = code[pc + 1], nargs = code[pc + 2];
        const int32_t start = pc + 3;
        int32_t end = start;
        skip_children(code, end, nargs);
        pc = end;
        for (const Bind& s : in) {
            const std::vector<int32_t>& mem = members[find(cls)];   // no unions happen during matching
            for (int32_t id : mem) {
                const Node& n = nodes[id];
                if (n.op != op || static_cast<int32_t>(n.args.size()) != nargs) continue;
                std::vector<Bind> partial{s};
                int32_t kpc = start;
                for (int32_t k = 0; k < nargs && !partial.empty(); ++k) {
                    std::vector<Bind> next;
                    int32_t here = kpc;
                    match(code, here, find(n.args[k]), partial, next);
                    partial.swap(next);
                    int32_t skip = kpc;
                    skip_children(code, skip, 1);
                    kpc = skip;
                }
                out.insert(out.end(), partial.begin(), partial.end());
            }
        }
    }

    static void skip_children(const int32_t* code, int32_t& pc, int32_t count) {
        for (int32_t k = 0; k < count; ++k) {
            const int32_t kind = code[pc];
            if (kind == 0) pc += 2;
            else if (kind == 1) pc += 1;
            else {
                const int32_t nargs = code[pc + 2];
                pc += 3;
                skip_children(code, pc, nargs);
            }
        }
    }

    std::vector<int32_t> classes() {
        std::vector<int32_t> out;
        for (size_t i = 0; i < members.size(); ++i)
            if (!members[i].empty()) out.push_back(static_cast<int32_t>(i));
        return out;
    }
};

extern "C" {

HX_API hx_egraph* hx_eg_new(int32_t const_op) {
    auto* g = new hx_egraph();
    g->const_op = const_op;
    return g;
}

HX_API void hx_eg_free(hx_egraph* g) { delete g; }

HX_API int32_t hx_eg_add(hx_egraph* g, int32_t op, const int32_t* args, int32_t nargs, int64_t payload) {
    return g->add(op, args, nargs, payload);
}

HX_API int32_t hx_eg_find(hx_egraph* g, int32_t id) { return g->find(id); }

HX_API int32_t hx_eg_union(hx_egraph* g, int32_t a, int32_t b, int32_t why, const int32_t* premises,
                           int32_t npremises) {
    if (npremises > 0) {
        auto& list = g->premises[why];
        for (int32_t k = 0; k < npremises; ++k) list.emplace_back(premises[2 * k], premises[2 * k + 1]);
    }
    return g->unite(a, b, why, -1, -1);
}

HX_API void hx_eg_rebuild(hx_egraph* g, int32_t* attempts, int32_t* merges) { g->rebuild(attempts, merges); }

HX_API int32_t hx_eg_take_conflicts(hx_egraph* g, int32_t* pairs, int32_t cap) {
    const int32_t n = static_cast<int32_t>(std::min<size_t>(g->conflicts.size(), static_cast<size_t>(cap)));
    for (int32_t k = 0; k < n; ++k) {
        pairs[2 * k] = g->conflicts[k].first;
        pairs[2 * k + 1] = g->conflicts[k].second;
    }
    g->conflicts.erase(g->conflicts.begin(), g->conflicts.begin() + n);
    return n;
}

HX_API int32_t hx_eg_lookup(hx_egraph* g, int32_t op, const int32_t* args, int32_t nargs, int64_t payload) {
    auto hit = g->memo.find(g->canon(op, args, nargs, payload));
    return hit == g->memo.end() ? -1 : hit->second;
}

HX_API int32_t hx_eg_const_node(hx_egraph* g, int32_t cls) { return g->const_of[g->find(cls)]; }

HX_API int32_t hx_eg_num_nodes(hx_egraph* g) { return static_cast<int32_t>(g->nodes.size()); }

HX_API int32_t hx_eg_classes(hx_egraph* g, int32_t* out, int32_t cap) {
    const auto all = g->classes();
    const int32_t n = static_cast<int32_t>(all.size());
    for (int32_t k = 0; k < n && k < cap; ++k) out[k] = all[k];
    return n;
}

HX_API int32_t hx_eg_members(hx_egraph* g, int32_t cls, int32_t* out, int32_t cap) {
    const auto& m = g->members[g->find(cls)];
    const int32_t n = static_cast<int32_t>(m.size());
    for (int32_t k = 0; k < n && k < cap; ++k) out[k] = m[k];
    return n;
}

HX_API int32_t hx_eg_node(hx_egraph* g, int32_t id, int32_t* op, int64_t* payload, int32_t* args, int32_t cap,
                          int32_t canonical) {
    const Node& n = g->nodes[id];
    *op = n.op;
    *payload = n.payload;
    const int32_t k = static_cast<int32_t>(n.args.size());
    for (int32_t i = 0; i < k && i < cap; ++i) args[i] = canonical ? g->find(n.args[i]) : n.args[i];
    return k;
}

HX_API int32_t hx_eg_explain(hx_egraph* g, int32_t a, int32_t b, int32_t* out, int32_t cap) {
    std::set<std::pair<int32_t, int32_t>> seen;
    std::vector<int32_t> reasons;
    g->explain(a, b, seen, reasons);
    const int32_t n = static_cast<int32_t>(reasons.size());
    for (int32_t k = 0; k < n && k < cap; ++k) out[k] = reasons[k];
    return n;
}

HX_API void hx_eg_snapshot_index(hx_egraph* g) {
    g->index.clear();
    for (int32_t cls : g->classes()) {
        for (int32_t id : g->members[cls]) {
            auto& list = g->index[g->nodes[id].op];
            if (list.empty() || list.back() != cls) list.push_back(cls);
        }
    }
    for (auto& kv : g->index) {
        std::sort(kv.second.begin(), kv.second.end());
        kv.second.erase(std::unique(kv.second.begin(), kv.second.end()), kv.second.end());
    }
    g->has_index = true;
}

HX_API int32_t hx_eg_ematch(hx_egraph* g, const int32_t* code, int32_t ncode, int32_t nvars, int32_t use_index,
                            int32_t* out, int32_t cap) {
    (void)ncode;
    int32_t count = 0;
    const int32_t stride = 1 + nvars;
    auto emit = [&](int32_t anchor, const Bind& b) {
        if (count >= cap) return;
        out[count * stride] = anchor;
        for (int32_t k = 0; k < nvars; ++k) out[count * stride + 1 + k] = b[k];
        ++count;
    };
    const Bind empty(nvars, -1);
    if (code[0] != 2) {                               // bare variable / constant pattern: every class
        for (int32_t cls : g->classes()) {
            std::vector<Bind> res;
            int32_t pc = 0;
            g->match(code, pc, cls, {empty}, res);
            for (const Bind& b : res) emit(cls, b);
        }
        return count;
    }
    const int32_t op = code[1], nargs = code[2];
    std::vector<int32_t> roots;
    if (use_index && g->has_index) {
        auto it = g->index.find(op);
        if (it != g->index.end()) roots = it->second;
    } else {
        roots = g->classes();
    }
    for (int32_t cls : roots) {
        const std::vector<int32_t> mem = g->members[g->find(cls)];
        for (int32_t id : mem) {
            const Node& n = g->nodes[id];
            if (n.op != op || static_cast<int32_t>(n.args.size()) != nargs) continue;
            std::vector<Bind> partial{empty};
            int32_t kpc = 3;
            for (int32_t k = 0; k < nargs && !partial.empty(); ++k) {
                std::vector<Bind> next;
                int32_t here = kpc;
                g->match(code, here, g->find(n.args[k]), partial, next);
                partial.swap(next);
                int32_t skip = kpc;
                hx_egraph::skip_children(code, skip, 1);
                kpc = skip;
            }
            for (const Bind& b : partial) emit(id, b);
        }
    }
    return count;
}

}  // extern "C"
