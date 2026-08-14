#ifndef SHUFTRI_PSI_BACKEND_H
#define SHUFTRI_PSI_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PSI_BACKEND_VERSION "1.0.0"
#define PSI_POINT_BYTES 32u
#define PSI_FRAME_HEADER_BYTES 32u

/*
 * Timings cover local protocol computation and application-frame
 * serialization/parsing. total_ms begins before protocol allocations and ends
 * after sensitive buffers are wiped/freed. It deliberately excludes the
 * plaintext oracle, task scheduling, filesystem I/O, and network transport.
 */
typedef struct psi_measurement {
    uint64_t session_id;
    uint64_t initiator_items;
    uint64_t responder_items;
    uint64_t cardinality;
    uint64_t plaintext_cardinality;

    double total_ms;
    double scalar_rng_ms;
    double hash_to_group_ms;
    double initiator_blind_ms;
    double request_serialize_ms;
    double responder_parse_ms;
    double responder_compute_ms;
    double responder_shuffle_ms;
    double response_serialize_ms;
    double initiator_parse_ms;
    double initiator_finalize_ms;
    double matching_ms;

    uint64_t request_bytes;
    uint64_t response_bytes;
    uint64_t serialized_bytes;
    uint64_t payload_bytes;
    uint64_t framing_overhead_bytes;
    uint64_t allocation_bytes;
    uint64_t rss_before_bytes;
    uint64_t rss_after_bytes;
    uint64_t process_peak_rss_bytes;
} psi_measurement;

/* Initializes libsodium. Safe to call more than once. */
int psi_backend_init(void);

/*
 * Executes the paper's two-message cardinality-only protocol locally.
 * Both sets must contain unique uint64_t identifiers. The function samples
 * fresh independent initiator/responder scalars and fresh responder shuffle
 * randomness from libsodium's CSPRNG for every call.
 *
 * Returns 0 on success, a negative value on malformed input, allocation
 * failure, invalid received point/frame, or a correctness-check failure.
 */
int psi_ca_session(const uint64_t *initiator_set,
                   size_t initiator_count,
                   const uint64_t *responder_set,
                   size_t responder_count,
                   uint64_t session_id,
                   psi_measurement *measurement);

const char *psi_backend_error(int code);

#ifdef __cplusplus
}
#endif

#endif
