// Bottom-up enumeration of feature expressions with observational equivalence (hrm_x/construct.py).
//
// Mirrors construct.enumerate_bank loop for loop, so the bank -- expressions, their order and their
// probe values -- is bit-identical to the Python reference (tests/test_x_native.py). Library tokens
// are not handled here: a stored concept is an exact Python function, so construction with a library
// stays on the Python path. Values are IEEE doubles computed by the same operations numpy performs
// elementwise (+, -, *, /, compare-select), which is what makes identity, not tolerance, testable.
#include "hrmx.h"

#include <cmath>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace {

constexpr int32_t X = 0, Y = 1, ONE = 2, TWO = 3;
constexpr int32_t ADD = 10, SUB = 11, MUL = 12, DIV = 13, IFGT = 14;
constexpr double MAX_VALUE = 1e12;

struct Expr {
    int32_t start, len;   // into the token pool
    int32_t vals;         // offset into the value pool (n doubles)
};

struct KeyHash {
    int32_t n;
    const std::vector<double>* pool;
    size_t operator()(int32_t off) const {
        size_t h = 1469598103934665603ull;
        for (int32_t i = 0; i < n; ++i) {
            double k = std::rint((*pool)[off + i] * 1e9);
            if (k == 0.0) k = 0.0;                        // -0.0 and 0.0 are one key (as in Python)
            uint64_t b;
            std::memcpy(&b, &k, sizeof b);
            // splitmix64 finaliser: the low mantissa bits of rint(v * 1e9) are mostly zero, so a
            // plain xor-multiply hash put nearly every key in the same buckets (quadratic lookups).
            b += 0x9e3779b97f4a7c15ull;
            b = (b ^ (b >> 30)) * 0xbf58476d1ce4e5b9ull;
            b = (b ^ (b >> 27)) * 0x94d049bb133111ebull;
            b ^= b >> 31;
            h = (h ^ b) * 1099511628211ull + (h >> 29);
        }
        return h;
    }
};

struct KeyEq {
    int32_t n;
    const std::vector<double>* pool;
    bool operator()(int32_t a, int32_t b) const {
        for (int32_t i = 0; i < n; ++i)
            if (std::rint((*pool)[a + i] * 1e9) != std::rint((*pool)[b + i] * 1e9)) return false;
        return true;
    }
};

struct Bank {
    int32_t n;
    std::vector<int32_t> tokens;
    std::vector<double> values;                 // committed values, n per expression
    std::vector<std::vector<Expr>> by_size;
    std::unordered_set<int32_t, KeyHash, KeyEq> seen;
    std::vector<double> scratch;                // candidate values (appended to `values` on accept)

    explicit Bank(int32_t n_, int32_t max_size)
        : n(n_), by_size(max_size + 1), seen(1024, KeyHash{n_, &values}, KeyEq{n_, &values}), scratch(n_) {}

    bool valid(const double* v) const {
        for (int32_t i = 0; i < n; ++i)
            if (!std::isfinite(v[i]) || std::fabs(v[i]) > MAX_VALUE) return false;
        return true;
    }

    // Append expr = a ++ b ++ ... ++ [op] if its values are new; returns true when kept.
    bool add(const std::vector<const Expr*>& kids, int32_t op, const double* v) {
        if (!valid(v)) return false;
        int32_t off = static_cast<int32_t>(values.size());
        values.insert(values.end(), v, v + n);
        if (seen.count(off)) {
            values.resize(off);
            return false;
        }
        seen.insert(off);
        std::vector<int32_t> body;                    // copy first: inserting from `tokens` into itself
        for (const Expr* k : kids)                    // would read through invalidated iterators
            body.insert(body.end(), tokens.begin() + k->start, tokens.begin() + k->start + k->len);
        body.push_back(op);
        int32_t start = static_cast<int32_t>(tokens.size());
        tokens.insert(tokens.end(), body.begin(), body.end());
        int32_t len = static_cast<int32_t>(tokens.size()) - start;
        by_size[len].push_back(Expr{start, len, off});
        return true;
    }

    void leaf(int32_t tok, const double* v) {
        int32_t off = static_cast<int32_t>(values.size());
        values.insert(values.end(), v, v + n);
        if (seen.count(off)) { values.resize(off); return; }
        seen.insert(off);
        int32_t start = static_cast<int32_t>(tokens.size());
        tokens.push_back(tok);
        by_size[1].push_back(Expr{start, 1, off});
    }
};

}  // namespace

extern "C" HX_API int32_t hx_enumerate(int32_t max_size, const double* px, const double* py, int32_t n,
                                       int32_t* out_tokens, int32_t cap_tokens, int32_t* out_offsets,
                                       double* out_values, int32_t cap_exprs) {
    if (n <= 0 || max_size < 1) return 0;
    Bank bank(n, max_size);
    std::vector<double> ones(n, 1.0), twos(n, 2.0);
    bank.leaf(X, px);
    bank.leaf(Y, py);
    bank.leaf(ONE, ones.data());
    bank.leaf(TWO, twos.data());
    std::vector<double> v(n);
    for (int32_t size = 3; size <= max_size; ++size) {
        for (int32_t s1 = 1; s1 < size - 1; ++s1) {
            int32_t s2 = size - 1 - s1;
            // Indexed loops: by_size[size] grows while we iterate smaller sizes only.
            for (size_t i = 0; i < bank.by_size[s1].size(); ++i) {
                for (size_t j = 0; j < bank.by_size[s2].size(); ++j) {
                    Expr e1 = bank.by_size[s1][i], e2 = bank.by_size[s2][j];
                    const Expr* kids[2] = {&e1, &e2};
                    std::vector<const Expr*> k2(kids, kids + 2);
                    const double* v1 = &bank.values[e1.vals];
                    const double* v2 = &bank.values[e2.vals];
                    for (int32_t t = 0; t < n; ++t) v[t] = v1[t] + v2[t];
                    bank.add(k2, ADD, v.data());
                    v1 = &bank.values[e1.vals]; v2 = &bank.values[e2.vals];     // pool may have grown
                    for (int32_t t = 0; t < n; ++t) v[t] = v1[t] - v2[t];
                    bank.add(k2, SUB, v.data());
                    v1 = &bank.values[e1.vals]; v2 = &bank.values[e2.vals];
                    for (int32_t t = 0; t < n; ++t) v[t] = v1[t] * v2[t];
                    bank.add(k2, MUL, v.data());
                    v1 = &bank.values[e1.vals]; v2 = &bank.values[e2.vals];
                    bool nonzero = true;
                    for (int32_t t = 0; t < n; ++t) nonzero = nonzero && v2[t] != 0.0;
                    if (nonzero) {
                        for (int32_t t = 0; t < n; ++t) v[t] = v1[t] / v2[t];
                        bank.add(k2, DIV, v.data());
                    }
                }
            }
        }
        if (size >= 5) {
            for (int32_t s1 = 1; s1 < size - 3; ++s1)
                for (int32_t s2 = 1; s2 < size - 2 - s1; ++s2)
                    for (int32_t s3 = 1; s3 < size - 1 - s1 - s2; ++s3) {
                        int32_t s4 = size - 1 - s1 - s2 - s3;
                        if (s4 < 1) continue;
                        for (size_t a = 0; a < bank.by_size[s1].size(); ++a)
                            for (size_t b = 0; b < bank.by_size[s2].size(); ++b) {
                                Expr ea = bank.by_size[s1][a], eb = bank.by_size[s2][b];
                                int32_t gt = 0;
                                for (int32_t t = 0; t < n; ++t)
                                    gt += bank.values[ea.vals + t] > bank.values[eb.vals + t];
                                if (gt == 0 || gt == n) continue;       // a constant branch is c or d
                                for (size_t c = 0; c < bank.by_size[s3].size(); ++c)
                                    for (size_t d = 0; d < bank.by_size[s4].size(); ++d) {
                                        Expr ec = bank.by_size[s3][c], ed = bank.by_size[s4][d];
                                        for (int32_t t = 0; t < n; ++t)
                                            v[t] = bank.values[ea.vals + t] > bank.values[eb.vals + t]
                                                       ? bank.values[ec.vals + t] : bank.values[ed.vals + t];
                                        const Expr* kids[4] = {&ea, &eb, &ec, &ed};
                                        bank.add(std::vector<const Expr*>(kids, kids + 4), IFGT, v.data());
                                    }
                            }
                    }
        }
    }
    // Emit in the Python order: by size, then insertion order within a size.
    int32_t count = 0, tok = 0;
    for (int32_t size = 1; size <= max_size; ++size)
        for (const Expr& e : bank.by_size[size]) {
            if (count >= cap_exprs || tok + e.len > cap_tokens) return -1;
            out_offsets[count] = tok;
            std::memcpy(out_tokens + tok, &bank.tokens[e.start], sizeof(int32_t) * e.len);
            std::memcpy(out_values + static_cast<size_t>(count) * n, &bank.values[e.vals], sizeof(double) * n);
            tok += e.len;
            ++count;
        }
    out_offsets[count] = tok;
    return count;
}
