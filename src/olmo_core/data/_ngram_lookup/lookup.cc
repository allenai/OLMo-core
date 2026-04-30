// C++ adapter over a KenLM trie binary plus a forward-indexed ngram
// continuation file (built by data_gen/build_forward_index.py).
//
// At open time:
//   - Load the KenLM trie binary (probability source for KN-smoothed scoring).
//   - Capture the vocabulary strings; build dolma2-token-id <-> KenLM-WordIndex
//     translation tables (the ARPA was built with decimal token IDs as words).
//   - mmap the forward index; parse its header and per-order sections.
//   - Precompute a top-N unigram shortlist for the order-1 fallback.
//
// At query time `enumerate_top_k(prefix, K)` does:
//   1. Walk a KenLM State through the context (one BaseScore per ctx token).
//   2. Enumerate candidate next-tokens by binary-searching the forward
//      index at every relevant order (suffix of length 1 .. min(N_max, |h|)).
//      Union those continuations with the unigram shortlist.
//   3. For each candidate w, call BaseScore(state, w) — KenLM does the full
//      KN cross-order combine.
//   4. Top-K by log_p, written out as (dolma2 token id, log10 prob) pairs.
//
// All log-probs are log10 (KenLM ARPA semantics). The Python wrapper
// renormalizes the top-K to sum to 1.

#include "lm/model.hh"
#include "lm/state.hh"
#include "lm/word_index.hh"
#include "lm/enumerate_vocab.hh"
#include "lm/return.hh"
#include "util/string_piece.hh"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using KenLMModel = lm::ngram::QuantArrayTrieModel;
using lm::WordIndex;

constexpr uint32_t kInvalidTokenId = UINT32_MAX;

// Forward-index file format constants — must match build_forward_index.py.
// Version 2: continuations within each prefix are sorted by descending KN
// log10 probability so the data loader can take the first K' as a top-K'
// candidate set without scoring all of them.
constexpr char kForwardIndexMagic[4] = {'F', 'I', 'X', '1'};
constexpr uint32_t kForwardIndexVersion = 2;
constexpr size_t kForwardIndexHeaderSize = 64;
constexpr size_t kForwardIndexSectionHeaderSize = 48;

// EnumerateVocab subclass that captures every vocab string during model
// load. Lives only on the stack during ngram_lookup_open.
class CaptureVocab : public lm::EnumerateVocab {
 public:
    std::vector<std::string> words;

    void Add(WordIndex index, const StringPiece& str) override {
        if (index >= words.size()) words.resize(index + 1);
        words[index].assign(str.data(), str.size());
    }
};

// One per-order section of the forward index. All pointers are into the
// mmap'd file — read-only, no ownership.
struct ForwardSection {
    unsigned int order = 0;
    uint64_t n_prefixes = 0;
    uint64_t n_continuations = 0;
    // prefix_words: n_prefixes rows of (order-1) uint32 each, sorted lex.
    const uint32_t* prefix_words = nullptr;
    // cont_offsets: n_prefixes + 1 uint32. continuations[cont_offsets[i] :
    // cont_offsets[i+1]] is prefix i's continuations.
    const uint32_t* cont_offsets = nullptr;
    // continuations: n_continuations uint32 (dolma2 token IDs).
    const uint32_t* continuations = nullptr;
};

struct ModelWrapper {
    KenLMModel* model = nullptr;
    unsigned int n_max = 0;

    // KenLM word index -> dolma2 token ID, or kInvalidTokenId for non-token
    // vocab entries (<unk>, <s>, </s>).
    std::vector<uint32_t> kenlm_id_to_token_id;
    // Inverse: dolma2 token ID -> KenLM word index. Sparse, hashmap.
    std::unordered_map<uint32_t, WordIndex> token_id_to_kenlm_id;

    // Forward index mmap.
    void* forward_index_base = MAP_FAILED;
    size_t forward_index_size = 0;
    std::vector<ForwardSection> sections;  // sorted by order ascending

    // Cap on continuations enumerated per prefix per order at lookup time.
    // The continuations are sorted by descending KN prob at build time, so
    // the first N' are the top-N' candidates by prob. Bounds candidate set
    // size and BaseScore work per position.
    unsigned int max_continuations_per_prefix = 64;

    // Top-N unigrams by log10-prob, descending (kenlm word index + log10p).
    std::vector<std::pair<WordIndex, float>> unigram_shortlist;
};

// Try to parse `s` as an unsigned 32-bit integer in base 10. Returns
// kInvalidTokenId on failure.
uint32_t parse_token_id(const std::string& s) {
    if (s.empty()) return kInvalidTokenId;
    char* end = nullptr;
    errno = 0;
    unsigned long val = std::strtoul(s.c_str(), &end, 10);
    if (errno != 0 || end == s.c_str() || *end != '\0' || val > UINT32_MAX) {
        return kInvalidTokenId;
    }
    return static_cast<uint32_t>(val);
}

// Binary-search a forward-index section for `prefix_tokens` (length =
// sect.order - 1). Returns pointer to the prefix's continuations slice,
// or nullptr if the prefix isn't observed at this order. Sets *n_cont.
const uint32_t* lookup_forward_index(const ForwardSection& sect,
                                     const uint32_t* prefix_tokens,
                                     uint64_t* n_cont) {
    *n_cont = 0;
    if (sect.order < 2 || sect.n_prefixes == 0) return nullptr;
    unsigned int plen = sect.order - 1;
    int64_t lo = 0;
    int64_t hi = static_cast<int64_t>(sect.n_prefixes);
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        const uint32_t* row = sect.prefix_words + static_cast<size_t>(mid) * plen;
        int cmp = 0;
        for (unsigned int i = 0; i < plen; ++i) {
            if (row[i] < prefix_tokens[i]) { cmp = -1; break; }
            if (row[i] > prefix_tokens[i]) { cmp = 1; break; }
        }
        if (cmp == 0) {
            uint64_t start = sect.cont_offsets[mid];
            uint64_t end = sect.cont_offsets[mid + 1];
            *n_cont = end - start;
            return sect.continuations + start;
        }
        if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return nullptr;
}

// mmap the forward index file, parse its header + section headers.
// Returns true on success; on failure returns false and leaves wrapper
// fields cleared. Errors are written to stderr (no exceptions across
// the C boundary).
bool load_forward_index(const char* path, ModelWrapper* w) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "ngram_lookup: open(%s) failed\n", path);
        return false;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0) {
        fprintf(stderr, "ngram_lookup: fstat(%s) failed\n", path);
        ::close(fd);
        return false;
    }
    void* base = ::mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (base == MAP_FAILED) {
        fprintf(stderr, "ngram_lookup: mmap(%s) failed\n", path);
        return false;
    }
    w->forward_index_base = base;
    w->forward_index_size = st.st_size;

    if (st.st_size < (off_t)kForwardIndexHeaderSize) {
        fprintf(stderr, "ngram_lookup: forward index too small\n");
        return false;
    }
    const char* p = static_cast<const char*>(base);
    if (std::memcmp(p, kForwardIndexMagic, 4) != 0) {
        fprintf(stderr, "ngram_lookup: forward index bad magic\n");
        return false;
    }
    uint32_t version, n_orders, vocab_size;
    std::memcpy(&version, p + 4, 4);
    std::memcpy(&n_orders, p + 8, 4);
    std::memcpy(&vocab_size, p + 12, 4);
    if (version != kForwardIndexVersion) {
        fprintf(stderr,
                "ngram_lookup: forward_index.bin version %u != expected %u — "
                "rebuild with current code (data_gen/build_forward_index.py)\n",
                version, kForwardIndexVersion);
        return false;
    }

    w->sections.clear();
    w->sections.reserve(n_orders);
    size_t hdr_off = kForwardIndexHeaderSize;
    for (unsigned int i = 0; i < n_orders; ++i) {
        const char* sh = p + hdr_off;
        // Layout (must match build_forward_index.py):
        //   off  0: order (uint32)
        //   off  4: pad   (uint32, zero — for 8-byte alignment)
        //   off  8: n_prefixes (uint64)
        //   off 16: n_continuations (uint64)
        //   off 24: prefix_words_off (uint64)
        //   off 32: cont_offsets_off (uint64)
        //   off 40: continuations_off (uint64)
        uint32_t order;
        uint64_t n_pref, n_cont, pw_off, co_off, c_off;
        std::memcpy(&order, sh + 0, 4);
        std::memcpy(&n_pref, sh + 8, 8);
        std::memcpy(&n_cont, sh + 16, 8);
        std::memcpy(&pw_off, sh + 24, 8);
        std::memcpy(&co_off, sh + 32, 8);
        std::memcpy(&c_off, sh + 40, 8);

        ForwardSection s;
        s.order = order;
        s.n_prefixes = n_pref;
        s.n_continuations = n_cont;
        s.prefix_words = reinterpret_cast<const uint32_t*>(p + pw_off);
        s.cont_offsets = reinterpret_cast<const uint32_t*>(p + co_off);
        s.continuations = reinterpret_cast<const uint32_t*>(p + c_off);
        w->sections.push_back(s);
        hdr_off += kForwardIndexSectionHeaderSize;
    }
    std::sort(w->sections.begin(), w->sections.end(),
              [](const ForwardSection& a, const ForwardSection& b) {
                  return a.order < b.order;
              });
    return true;
}

}  // namespace

extern "C" {

// Open both the KenLM trie binary and the forward index. Returns a
// handle owned by the caller; pass it to ngram_lookup_close when done.
// On failure returns nullptr and writes a message to stderr.
//
//   trie_path           path to pilot.binary
//   forward_index_path  path to forward_index.bin
//   unigram_shortlist_size       size of the order-1 fallback set
//   max_continuations_per_prefix top-N' to enumerate per prefix per order
void* ngram_lookup_open(const char* trie_path,
                        const char* forward_index_path,
                        unsigned int unigram_shortlist_size,
                        unsigned int max_continuations_per_prefix) {
    if (trie_path == nullptr || forward_index_path == nullptr) return nullptr;
    auto* w = new ModelWrapper();
    w->max_continuations_per_prefix = max_continuations_per_prefix;
    try {
        // Load the trie model with vocabulary-string capture.
        CaptureVocab cap;
        lm::ngram::Config config;
        config.enumerate_vocab = &cap;
        w->model = new KenLMModel(trie_path, config);
        w->n_max = w->model->Order();

        // Build translation tables.
        size_t vocab_count = cap.words.size();
        w->kenlm_id_to_token_id.assign(vocab_count, kInvalidTokenId);
        for (size_t i = 0; i < vocab_count; ++i) {
            uint32_t t = parse_token_id(cap.words[i]);
            w->kenlm_id_to_token_id[i] = t;
            if (t != kInvalidTokenId) {
                w->token_id_to_kenlm_id[t] = static_cast<WordIndex>(i);
            }
        }

        // Build the unigram shortlist by reading log-probs from the
        // unigram table for every parseable word.
        const auto& search = w->model->GetSearch();
        std::vector<std::pair<WordIndex, float>> all;
        all.reserve(vocab_count);
        for (WordIndex word = 0; word < vocab_count; ++word) {
            if (w->kenlm_id_to_token_id[word] == kInvalidTokenId) continue;
            lm::ngram::trie::NodeRange ignored_node;
            bool ignored_ind;
            uint64_t ignored_ext;
            auto u = search.LookupUnigram(word, ignored_node, ignored_ind, ignored_ext);
            all.emplace_back(word, u.Prob());
        }
        std::sort(all.begin(), all.end(),
                  [](const std::pair<WordIndex, float>& a,
                     const std::pair<WordIndex, float>& b) {
                      return a.second > b.second;
                  });
        size_t keep = std::min(static_cast<size_t>(unigram_shortlist_size), all.size());
        w->unigram_shortlist.assign(all.begin(), all.begin() + keep);

        // Load the forward index.
        if (!load_forward_index(forward_index_path, w)) {
            delete w->model;
            delete w;
            return nullptr;
        }

        return w;
    } catch (const std::exception& e) {
        fprintf(stderr, "ngram_lookup_open: %s\n", e.what());
        delete w->model;
        delete w;
        return nullptr;
    }
}

void ngram_lookup_close(void* handle) {
    if (handle == nullptr) return;
    auto* w = static_cast<ModelWrapper*>(handle);
    if (w->forward_index_base != MAP_FAILED && w->forward_index_size > 0) {
        ::munmap(w->forward_index_base, w->forward_index_size);
    }
    delete w->model;
    delete w;
}

unsigned int ngram_lookup_order(void* handle) {
    if (handle == nullptr) return 0;
    return static_cast<ModelWrapper*>(handle)->n_max;
}

uint64_t ngram_lookup_vocab_size(void* handle) {
    if (handle == nullptr) return 0;
    return static_cast<ModelWrapper*>(handle)->kenlm_id_to_token_id.size();
}

unsigned int ngram_lookup_n_forward_orders(void* handle) {
    if (handle == nullptr) return 0;
    return static_cast<unsigned int>(static_cast<ModelWrapper*>(handle)->sections.size());
}

// Top-K continuations for a single position with full KN-smoothed scoring.
//
//   handle       -- model wrapper from ngram_lookup_open
//   prefix       -- dolma2 token IDs of the context, oldest first
//   prefix_len   -- number of context tokens (any length; oversized contexts
//                   only use the suffix that fits the highest-order forward
//                   section)
//   k            -- max output entries to write
//   out_token_ids[k]  -- output token IDs (top-K, descending log-prob)
//   out_log_probs[k]  -- their log10 probabilities (RAW from BaseScore;
//                        NOT renormalized — the Python wrapper does that)
//
// Returns the number of entries actually written (≤ k).
int ngram_lookup_enumerate_top_k(void* handle,
                                 const uint32_t* prefix,
                                 unsigned int prefix_len,
                                 unsigned int k,
                                 uint32_t* out_token_ids,
                                 float* out_log_probs) {
    if (handle == nullptr || k == 0) return 0;
    auto* w = static_cast<ModelWrapper*>(handle);

    // Translate prefix tokens to KenLM word IDs for the state walk. If a
    // token isn't in KenLM's vocab, mark it as a barrier — we drop everything
    // up to and including it (a context walk can't pass through OOV).
    std::vector<WordIndex> kenlm_prefix(prefix_len);
    unsigned int eff_start = 0;
    for (unsigned int i = 0; i < prefix_len; ++i) {
        auto it = w->token_id_to_kenlm_id.find(prefix[i]);
        if (it == w->token_id_to_kenlm_id.end()) {
            eff_start = i + 1;
        } else {
            kenlm_prefix[i] = it->second;
        }
    }
    unsigned int eff_prefix_len = prefix_len - eff_start;

    // Walk the state through the effective context.
    lm::ngram::State state, out_state;
    w->model->NullContextWrite(&state);
    for (unsigned int i = eff_start; i < prefix_len; ++i) {
        w->model->BaseScore(&state, kenlm_prefix[i], &out_state);
        state = out_state;
    }

    // Collect candidate tokens (dolma2 IDs) from the forward index at every
    // relevant order, plus the unigram shortlist.
    std::unordered_set<uint32_t> candidate_token_ids;
    candidate_token_ids.reserve(256);

    for (const auto& sect : w->sections) {
        unsigned int target_order = sect.order;
        if (target_order < 2) continue;
        unsigned int plen = target_order - 1;
        if (plen > eff_prefix_len) continue;
        const uint32_t* h = prefix + (prefix_len - plen);
        uint64_t n_cont = 0;
        const uint32_t* conts = lookup_forward_index(sect, h, &n_cont);
        if (conts == nullptr) continue;
        // Continuations are sorted by descending KN prob at build time.
        // Take the first max_continuations_per_prefix as the top-N' by
        // prob; this bounds the candidate set when a prefix has thousands
        // of observed continuations (common for short prefixes like a
        // single high-frequency unigram).
        uint64_t take = std::min<uint64_t>(n_cont, w->max_continuations_per_prefix);
        for (uint64_t i = 0; i < take; ++i) {
            candidate_token_ids.insert(conts[i]);
        }
    }
    for (const auto& wp : w->unigram_shortlist) {
        uint32_t tok = w->kenlm_id_to_token_id[wp.first];
        if (tok != kInvalidTokenId) {
            candidate_token_ids.insert(tok);
        }
    }

    // Score each candidate via BaseScore, keep top-K via min-heap.
    using Entry = std::pair<float, uint32_t>;  // (log10p, dolma2 token id)
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> heap;
    for (uint32_t tok : candidate_token_ids) {
        auto it = w->token_id_to_kenlm_id.find(tok);
        if (it == w->token_id_to_kenlm_id.end()) continue;
        WordIndex kenlm_w = it->second;
        float log_p = w->model->BaseScore(&state, kenlm_w, &out_state);
        if (heap.size() < k) {
            heap.emplace(log_p, tok);
        } else if (log_p > heap.top().first) {
            heap.pop();
            heap.emplace(log_p, tok);
        }
    }

    // Pop into the output buffer; reverse for descending order.
    std::vector<Entry> sorted_out;
    sorted_out.reserve(heap.size());
    while (!heap.empty()) {
        sorted_out.push_back(heap.top());
        heap.pop();
    }
    std::reverse(sorted_out.begin(), sorted_out.end());

    int n_out = static_cast<int>(sorted_out.size());
    for (int i = 0; i < n_out; ++i) {
        out_token_ids[i] = sorted_out[i].second;
        out_log_probs[i] = sorted_out[i].first;
    }
    return n_out;
}

// Debug: do exactly what kenlm.Model.BaseScore does. Walk state through
// ctx, then BaseScore on target. Returns log10P and writes ngram_length.
// Returns 1.0 (sentinel) if any context token is OOV.
float ngram_lookup_basescore_debug(void* handle,
                                   const uint32_t* ctx_tokens,
                                   unsigned int ctx_len,
                                   uint32_t target_token,
                                   unsigned int* out_ngram_length) {
    *out_ngram_length = 0;
    if (handle == nullptr) return 1.0f;
    auto* w = static_cast<ModelWrapper*>(handle);
    std::vector<WordIndex> kenlm_ctx;
    for (unsigned int i = 0; i < ctx_len; ++i) {
        auto it = w->token_id_to_kenlm_id.find(ctx_tokens[i]);
        if (it == w->token_id_to_kenlm_id.end()) return 1.0f;
        kenlm_ctx.push_back(it->second);
    }
    auto it_target = w->token_id_to_kenlm_id.find(target_token);
    if (it_target == w->token_id_to_kenlm_id.end()) return 1.0f;
    WordIndex kenlm_target = it_target->second;

    lm::ngram::State state, out_state;
    w->model->NullContextWrite(&state);
    for (auto kw : kenlm_ctx) {
        w->model->BaseScore(&state, kw, &out_state);
        state = out_state;
    }
    auto ret = w->model->BaseFullScore(&state, kenlm_target, &out_state);
    *out_ngram_length = ret.ngram_length;
    return ret.prob;
}

}  // extern "C"
