#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE

#include "psi_backend.h"

#include <errno.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif
#include <pthread.h>
#include <sodium.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define FRAME_VERSION 1u
#define FRAME_REQUEST 1u
#define FRAME_RESPONSE 2u
#define MAX_WORKLOAD_BYTES 95u
#define H2G_DOMAIN "ShufTri-PSI-CA/Ristretto255/H2G/BLAKE2b-512/v1"

enum {
    PSI_OK = 0,
    PSI_ERR_INPUT = -1,
    PSI_ERR_MEMORY = -2,
    PSI_ERR_CRYPTO = -3,
    PSI_ERR_FRAME = -4,
    PSI_ERR_POINT = -5,
    PSI_ERR_CORRECTNESS = -6,
    PSI_ERR_IO = -7
};

typedef struct {
    uint64_t owner;
    uint64_t item;
} Pair;

typedef struct {
    uint64_t owner;
    size_t offset;
    size_t count;
} OwnerSet;

typedef struct {
    uint64_t *items;
    size_t item_count;
    OwnerSet *owners;
    size_t owner_count;
} SetDatabase;

typedef struct {
    uint64_t session_id;
    uint64_t initiator;
    uint64_t responder;
    char workload[MAX_WORKLOAD_BYTES + 1u];
} Session;

typedef struct {
    unsigned char *data;
    size_t len;
} Frame;

typedef struct {
    const unsigned char *first;
    const unsigned char *second;
    uint32_t first_count;
    uint32_t second_count;
} ParsedFrame;

typedef struct {
    psi_measurement measurement;
    size_t task_index;
    size_t schedule_position;
    size_t session_index;
    size_t repetition;
    int status;
} TaskResult;

typedef struct {
    const SetDatabase *sets;
    const Session *sessions;
    size_t session_count;
    const size_t *order;
    size_t task_count;
    atomic_size_t *next_position;
    atomic_int *failed;
    TaskResult *results;
} WorkerContext;

static double monotonic_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0.0;
    }
    return (double) ts.tv_sec * 1000.0 + (double) ts.tv_nsec / 1.0e6;
}

static uint64_t current_rss_bytes(void) {
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t) &info, &count) != KERN_SUCCESS) {
        return 0;
    }
    return (uint64_t) info.resident_size;
#else
    FILE *fp = fopen("/proc/self/statm", "r");
    unsigned long ignored = 0;
    unsigned long resident = 0;
    long page_size;
    if (fp == NULL) {
        return 0;
    }
    if (fscanf(fp, "%lu %lu", &ignored, &resident) != 2) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    page_size = sysconf(_SC_PAGESIZE);
    return page_size > 0 ? (uint64_t) resident * (uint64_t) page_size : 0;
#endif
}

static uint64_t process_peak_rss_bytes(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
#ifdef __APPLE__
    return (uint64_t) usage.ru_maxrss;
#else
    return (uint64_t) usage.ru_maxrss * 1024u;
#endif
}

static void put_u16le(unsigned char *out, uint16_t value) {
    out[0] = (unsigned char) value;
    out[1] = (unsigned char) (value >> 8);
}

static void put_u32le(unsigned char *out, uint32_t value) {
    out[0] = (unsigned char) value;
    out[1] = (unsigned char) (value >> 8);
    out[2] = (unsigned char) (value >> 16);
    out[3] = (unsigned char) (value >> 24);
}

static void put_u64le(unsigned char *out, uint64_t value) {
    for (unsigned int i = 0; i < 8; ++i) {
        out[i] = (unsigned char) (value >> (8u * i));
    }
}

static uint16_t get_u16le(const unsigned char *in) {
    return (uint16_t) ((uint16_t) in[0] | ((uint16_t) in[1] << 8));
}

static uint32_t get_u32le(const unsigned char *in) {
    return (uint32_t) in[0] |
           ((uint32_t) in[1] << 8) |
           ((uint32_t) in[2] << 16) |
           ((uint32_t) in[3] << 24);
}

static uint64_t get_u64le(const unsigned char *in) {
    uint64_t value = 0;
    for (unsigned int i = 0; i < 8; ++i) {
        value |= (uint64_t) in[i] << (8u * i);
    }
    return value;
}

static bool checked_add_size(size_t a, size_t b, size_t *out) {
    if (a > SIZE_MAX - b) {
        return false;
    }
    *out = a + b;
    return true;
}

static bool checked_mul_size(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > SIZE_MAX / a) {
        return false;
    }
    *out = a * b;
    return true;
}

static unsigned char *allocate_bytes(size_t count) {
    if (count == 0) {
        return NULL;
    }
    return (unsigned char *) malloc(count);
}

static void wipe_and_free(unsigned char *buffer, size_t length) {
    if (buffer != NULL) {
        sodium_memzero(buffer, length);
        free(buffer);
    }
}

static int compare_pair(const void *left, const void *right) {
    const Pair *a = (const Pair *) left;
    const Pair *b = (const Pair *) right;
    if (a->owner != b->owner) {
        return (a->owner > b->owner) - (a->owner < b->owner);
    }
    return (a->item > b->item) - (a->item < b->item);
}

static int compare_point(const void *left, const void *right) {
    return memcmp(left, right, PSI_POINT_BYTES);
}

static int compare_double(const void *left, const void *right) {
    double a = *(const double *) left;
    double b = *(const double *) right;
    return (a > b) - (a < b);
}

static bool is_strictly_sorted(const uint64_t *items, size_t count) {
    if (count > 0 && items == NULL) {
        return false;
    }
    for (size_t i = 1; i < count; ++i) {
        if (items[i - 1] >= items[i]) {
            return false;
        }
    }
    return true;
}

static uint64_t plaintext_intersection_sorted(const uint64_t *a, size_t na,
                                              const uint64_t *b, size_t nb) {
    size_t i = 0;
    size_t j = 0;
    uint64_t result = 0;
    while (i < na && j < nb) {
        if (a[i] == b[j]) {
            ++result;
            ++i;
            ++j;
        } else if (a[i] < b[j]) {
            ++i;
        } else {
            ++j;
        }
    }
    return result;
}

static int hash_identifier_to_group(uint64_t identifier,
                                    unsigned char point[PSI_POINT_BYTES]) {
    crypto_generichash_state state;
    unsigned char digest[crypto_core_ristretto255_HASHBYTES];
    unsigned char encoded_id[8];
    unsigned char counter_bytes[4];
    const unsigned char domain[] = H2G_DOMAIN;

    put_u64le(encoded_id, identifier);
    for (uint32_t counter = 0; counter < 16; ++counter) {
        put_u32le(counter_bytes, counter);
        if (crypto_generichash_init(&state, NULL, 0, sizeof digest) != 0) {
            sodium_memzero(&state, sizeof state);
            return PSI_ERR_CRYPTO;
        }
        crypto_generichash_update(&state, domain, sizeof domain - 1u);
        crypto_generichash_update(&state, encoded_id, sizeof encoded_id);
        crypto_generichash_update(&state, counter_bytes, sizeof counter_bytes);
        crypto_generichash_final(&state, digest, sizeof digest);
        sodium_memzero(&state, sizeof state);
        if (crypto_core_ristretto255_from_hash(point, digest) != 0) {
            sodium_memzero(digest, sizeof digest);
            return PSI_ERR_CRYPTO;
        }
        if (crypto_core_ristretto255_is_valid_point(point) == 1 &&
            sodium_is_zero(point, PSI_POINT_BYTES) == 0) {
            sodium_memzero(digest, sizeof digest);
            sodium_memzero(encoded_id, sizeof encoded_id);
            sodium_memzero(counter_bytes, sizeof counter_bytes);
            return PSI_OK;
        }
    }
    sodium_memzero(digest, sizeof digest);
    sodium_memzero(encoded_id, sizeof encoded_id);
    sodium_memzero(counter_bytes, sizeof counter_bytes);
    return PSI_ERR_POINT;
}

static void random_nonzero_scalar(unsigned char scalar[crypto_core_ristretto255_SCALARBYTES]) {
    do {
        crypto_core_ristretto255_scalar_random(scalar);
    } while (sodium_is_zero(scalar, crypto_core_ristretto255_SCALARBYTES) == 1);
}

/* The responder alone invokes this CSPRNG-backed Fisher-Yates shuffle. */
static void private_shuffle_points(unsigned char *points, uint32_t count) {
    unsigned char temporary[PSI_POINT_BYTES];
    if (count < 2) {
        return;
    }
    for (uint32_t i = count - 1; i > 0; --i) {
        uint32_t j = randombytes_uniform(i + 1u);
        if (i != j) {
            memcpy(temporary, points + (size_t) i * PSI_POINT_BYTES,
                   PSI_POINT_BYTES);
            memcpy(points + (size_t) i * PSI_POINT_BYTES,
                   points + (size_t) j * PSI_POINT_BYTES,
                   PSI_POINT_BYTES);
            memcpy(points + (size_t) j * PSI_POINT_BYTES, temporary,
                   PSI_POINT_BYTES);
        }
    }
    sodium_memzero(temporary, sizeof temporary);
}

static bool valid_received_point(const unsigned char point[PSI_POINT_BYTES]) {
    /* libsodium checks canonical Ristretto encoding and curve/subgroup validity.
     * The explicit all-zero check additionally rejects the group identity. */
    return crypto_core_ristretto255_is_valid_point(point) == 1 &&
           sodium_is_zero(point, PSI_POINT_BYTES) == 0;
}

static int build_frame(uint16_t message_type, uint64_t session_id,
                       const unsigned char *first, uint32_t first_count,
                       const unsigned char *second, uint32_t second_count,
                       Frame *frame) {
    size_t point_count;
    size_t payload_length;
    size_t frame_length;
    if (frame == NULL ||
        !checked_add_size((size_t) first_count, (size_t) second_count,
                          &point_count) ||
        !checked_mul_size(point_count, PSI_POINT_BYTES, &payload_length) ||
        !checked_add_size(PSI_FRAME_HEADER_BYTES, payload_length,
                          &frame_length)) {
        return PSI_ERR_INPUT;
    }
    if ((first_count > 0 && first == NULL) ||
        (second_count > 0 && second == NULL)) {
        return PSI_ERR_INPUT;
    }
    frame->data = allocate_bytes(frame_length);
    if (frame->data == NULL) {
        return PSI_ERR_MEMORY;
    }
    frame->len = frame_length;
    memcpy(frame->data, "SPCA", 4);
    put_u16le(frame->data + 4, FRAME_VERSION);
    put_u16le(frame->data + 6, message_type);
    put_u64le(frame->data + 8, session_id);
    put_u32le(frame->data + 16, first_count);
    put_u32le(frame->data + 20, second_count);
    put_u64le(frame->data + 24, (uint64_t) payload_length);
    if (first_count > 0) {
        memcpy(frame->data + PSI_FRAME_HEADER_BYTES, first,
               (size_t) first_count * PSI_POINT_BYTES);
    }
    if (second_count > 0) {
        memcpy(frame->data + PSI_FRAME_HEADER_BYTES +
                   (size_t) first_count * PSI_POINT_BYTES,
               second, (size_t) second_count * PSI_POINT_BYTES);
    }
    return PSI_OK;
}

static int parse_frame(const Frame *frame, uint16_t expected_type,
                       uint64_t expected_session_id,
                       uint32_t expected_first_count,
                       uint32_t expected_second_count,
                       ParsedFrame *parsed) {
    size_t point_count;
    size_t expected_payload;
    size_t expected_length;
    uint32_t first_count;
    uint32_t second_count;
    uint64_t payload_length;

    if (frame == NULL || parsed == NULL || frame->data == NULL ||
        frame->len < PSI_FRAME_HEADER_BYTES ||
        memcmp(frame->data, "SPCA", 4) != 0 ||
        get_u16le(frame->data + 4) != FRAME_VERSION ||
        get_u16le(frame->data + 6) != expected_type ||
        get_u64le(frame->data + 8) != expected_session_id) {
        return PSI_ERR_FRAME;
    }
    first_count = get_u32le(frame->data + 16);
    second_count = get_u32le(frame->data + 20);
    payload_length = get_u64le(frame->data + 24);
    if (first_count != expected_first_count ||
        second_count != expected_second_count ||
        !checked_add_size((size_t) first_count, (size_t) second_count,
                          &point_count) ||
        !checked_mul_size(point_count, PSI_POINT_BYTES, &expected_payload) ||
        payload_length != (uint64_t) expected_payload ||
        !checked_add_size(PSI_FRAME_HEADER_BYTES, expected_payload,
                          &expected_length) ||
        frame->len != expected_length) {
        return PSI_ERR_FRAME;
    }
    parsed->first = frame->data + PSI_FRAME_HEADER_BYTES;
    parsed->second = parsed->first + (size_t) first_count * PSI_POINT_BYTES;
    parsed->first_count = first_count;
    parsed->second_count = second_count;
    for (size_t i = 0; i < point_count; ++i) {
        const unsigned char *point =
            frame->data + PSI_FRAME_HEADER_BYTES + i * PSI_POINT_BYTES;
        if (!valid_received_point(point)) {
            return PSI_ERR_POINT;
        }
    }
    return PSI_OK;
}

int psi_backend_init(void) {
    return sodium_init() < 0 ? PSI_ERR_CRYPTO : PSI_OK;
}

const char *psi_backend_error(int code) {
    switch (code) {
        case PSI_OK: return "success";
        case PSI_ERR_INPUT: return "invalid input";
        case PSI_ERR_MEMORY: return "allocation failure";
        case PSI_ERR_CRYPTO: return "cryptographic operation failed";
        case PSI_ERR_FRAME: return "invalid application frame";
        case PSI_ERR_POINT: return "invalid, non-canonical, or identity point";
        case PSI_ERR_CORRECTNESS: return "PSI/plaintext cardinality mismatch";
        case PSI_ERR_IO: return "I/O failure";
        default: return "unknown error";
    }
}

int psi_ca_session(const uint64_t *initiator_set, size_t initiator_count,
                   const uint64_t *responder_set, size_t responder_count,
                   uint64_t session_id, psi_measurement *measurement) {
    int status = PSI_OK;
    uint64_t oracle;
    uint32_t di;
    uint32_t dj;
    size_t initiator_bytes;
    size_t responder_bytes;
    size_t allocation_bytes = 0;
    double total_start = 0.0;
    double phase_start;
    unsigned char initiator_scalar[crypto_core_ristretto255_SCALARBYTES] = {0};
    unsigned char responder_scalar[crypto_core_ristretto255_SCALARBYTES] = {0};
    unsigned char *initiator_hashed = NULL;
    unsigned char *responder_hashed = NULL;
    unsigned char *initiator_blinded = NULL;
    unsigned char *responder_blinded = NULL;
    unsigned char *initiator_double_blinded = NULL;
    unsigned char *responder_double_blinded = NULL;
    Frame request = {0};
    Frame response = {0};
    ParsedFrame parsed_request = {0};
    ParsedFrame parsed_response = {0};

    if (measurement == NULL || initiator_count > UINT32_MAX ||
        responder_count > UINT32_MAX ||
        !is_strictly_sorted(initiator_set, initiator_count) ||
        !is_strictly_sorted(responder_set, responder_count) ||
        psi_backend_init() != PSI_OK) {
        return PSI_ERR_INPUT;
    }
    memset(measurement, 0, sizeof *measurement);
    di = (uint32_t) initiator_count;
    dj = (uint32_t) responder_count;
    if (!checked_mul_size(initiator_count, PSI_POINT_BYTES,
                          &initiator_bytes) ||
        !checked_mul_size(responder_count, PSI_POINT_BYTES,
                          &responder_bytes)) {
        return PSI_ERR_INPUT;
    }

    /* The plaintext oracle is only a correctness assertion. It is completed
     * before both the total and all phase timers begin. */
    oracle = plaintext_intersection_sorted(initiator_set, initiator_count,
                                           responder_set, responder_count);
    measurement->plaintext_cardinality = oracle;
    measurement->session_id = session_id;
    measurement->initiator_items = initiator_count;
    measurement->responder_items = responder_count;
    measurement->rss_before_bytes = current_rss_bytes();
    total_start = monotonic_ms();

    initiator_hashed = allocate_bytes(initiator_bytes);
    initiator_blinded = allocate_bytes(initiator_bytes);
    if (initiator_bytes > 0 &&
        (initiator_hashed == NULL || initiator_blinded == NULL)) {
        status = PSI_ERR_MEMORY;
        goto cleanup;
    }
    if (!checked_mul_size(initiator_bytes, 2u, &allocation_bytes)) {
        status = PSI_ERR_INPUT;
        goto cleanup;
    }

    phase_start = monotonic_ms();
    random_nonzero_scalar(initiator_scalar);
    measurement->scalar_rng_ms = monotonic_ms() - phase_start;

    phase_start = monotonic_ms();
    for (uint32_t i = 0; i < di; ++i) {
        status = hash_identifier_to_group(initiator_set[i],
                                          initiator_hashed +
                                              (size_t) i * PSI_POINT_BYTES);
        if (status != PSI_OK) {
            goto cleanup;
        }
    }
    measurement->hash_to_group_ms = monotonic_ms() - phase_start;

    phase_start = monotonic_ms();
    for (uint32_t i = 0; i < di; ++i) {
        if (crypto_scalarmult_ristretto255(
                initiator_blinded + (size_t) i * PSI_POINT_BYTES,
                initiator_scalar,
                initiator_hashed + (size_t) i * PSI_POINT_BYTES) != 0) {
            status = PSI_ERR_CRYPTO;
            goto cleanup;
        }
    }
    measurement->initiator_blind_ms = monotonic_ms() - phase_start;

    phase_start = monotonic_ms();
    status = build_frame(FRAME_REQUEST, session_id, initiator_blinded, di,
                         NULL, 0, &request);
    measurement->request_serialize_ms = monotonic_ms() - phase_start;
    if (status != PSI_OK) {
        goto cleanup;
    }
    if (!checked_add_size(allocation_bytes, request.len,
                          &allocation_bytes)) {
        status = PSI_ERR_INPUT;
        goto cleanup;
    }

    phase_start = monotonic_ms();
    status = parse_frame(&request, FRAME_REQUEST, session_id, di, 0,
                         &parsed_request);
    measurement->responder_parse_ms = monotonic_ms() - phase_start;
    if (status != PSI_OK) {
        goto cleanup;
    }

    /* Counts, lengths, session ID, and every received point have now been
     * validated. Only now does the responder allocate protocol workspace. */
    responder_hashed = allocate_bytes(responder_bytes);
    responder_blinded = allocate_bytes(responder_bytes);
    initiator_double_blinded = allocate_bytes(initiator_bytes);
    if ((responder_bytes > 0 &&
         (responder_hashed == NULL || responder_blinded == NULL)) ||
        (initiator_bytes > 0 && initiator_double_blinded == NULL)) {
        status = PSI_ERR_MEMORY;
        goto cleanup;
    }
    {
        size_t responder_workspace;
        size_t new_total;
        if (!checked_mul_size(responder_bytes, 2u,
                              &responder_workspace) ||
            !checked_add_size(responder_workspace, initiator_bytes,
                              &responder_workspace) ||
            !checked_add_size(allocation_bytes, responder_workspace,
                              &new_total)) {
            status = PSI_ERR_INPUT;
            goto cleanup;
        }
        allocation_bytes = new_total;
    }

    phase_start = monotonic_ms();
    random_nonzero_scalar(responder_scalar);
    measurement->scalar_rng_ms += monotonic_ms() - phase_start;

    phase_start = monotonic_ms();
    for (uint32_t i = 0; i < dj; ++i) {
        status = hash_identifier_to_group(responder_set[i],
                                          responder_hashed +
                                              (size_t) i * PSI_POINT_BYTES);
        if (status != PSI_OK) {
            goto cleanup;
        }
    }
    measurement->hash_to_group_ms += monotonic_ms() - phase_start;

    phase_start = monotonic_ms();
    for (uint32_t i = 0; i < di; ++i) {
        if (crypto_scalarmult_ristretto255(
                initiator_double_blinded + (size_t) i * PSI_POINT_BYTES,
                responder_scalar,
                parsed_request.first + (size_t) i * PSI_POINT_BYTES) != 0) {
            status = PSI_ERR_CRYPTO;
            goto cleanup;
        }
    }
    for (uint32_t i = 0; i < dj; ++i) {
        if (crypto_scalarmult_ristretto255(
                responder_blinded + (size_t) i * PSI_POINT_BYTES,
                responder_scalar,
                responder_hashed + (size_t) i * PSI_POINT_BYTES) != 0) {
            status = PSI_ERR_CRYPTO;
            goto cleanup;
        }
    }
    measurement->responder_compute_ms = monotonic_ms() - phase_start;

    phase_start = monotonic_ms();
    private_shuffle_points(initiator_double_blinded, di);
    private_shuffle_points(responder_blinded, dj);
    measurement->responder_shuffle_ms = monotonic_ms() - phase_start;

    phase_start = monotonic_ms();
    status = build_frame(FRAME_RESPONSE, session_id,
                         initiator_double_blinded, di,
                         responder_blinded, dj, &response);
    measurement->response_serialize_ms = monotonic_ms() - phase_start;
    if (status != PSI_OK) {
        goto cleanup;
    }
    if (!checked_add_size(allocation_bytes, response.len,
                          &allocation_bytes)) {
        status = PSI_ERR_INPUT;
        goto cleanup;
    }

    phase_start = monotonic_ms();
    status = parse_frame(&response, FRAME_RESPONSE, session_id, di, dj,
                         &parsed_response);
    measurement->initiator_parse_ms = monotonic_ms() - phase_start;
    if (status != PSI_OK) {
        goto cleanup;
    }

    /* The initiator likewise allocates its finalization workspace only after
     * validating the complete Round-2 frame and both point arrays. */
    responder_double_blinded = allocate_bytes(responder_bytes);
    if (responder_bytes > 0 && responder_double_blinded == NULL) {
        status = PSI_ERR_MEMORY;
        goto cleanup;
    }
    if (!checked_add_size(allocation_bytes, responder_bytes,
                          &allocation_bytes)) {
        status = PSI_ERR_INPUT;
        goto cleanup;
    }

    phase_start = monotonic_ms();
    for (uint32_t i = 0; i < dj; ++i) {
        if (crypto_scalarmult_ristretto255(
                responder_double_blinded + (size_t) i * PSI_POINT_BYTES,
                initiator_scalar,
                parsed_response.second + (size_t) i * PSI_POINT_BYTES) != 0) {
            status = PSI_ERR_CRYPTO;
            goto cleanup;
        }
    }
    measurement->initiator_finalize_ms = monotonic_ms() - phase_start;

    phase_start = monotonic_ms();
    if (di > 1) {
        qsort((void *) parsed_response.first, di, PSI_POINT_BYTES,
              compare_point);
    }
    if (dj > 1) {
        qsort(responder_double_blinded, dj, PSI_POINT_BYTES, compare_point);
    }
    {
        uint32_t i = 0;
        uint32_t j = 0;
        uint64_t cardinality = 0;
        while (i < di && j < dj) {
            int comparison = memcmp(
                parsed_response.first + (size_t) i * PSI_POINT_BYTES,
                responder_double_blinded + (size_t) j * PSI_POINT_BYTES,
                PSI_POINT_BYTES);
            if (comparison == 0) {
                ++cardinality;
                ++i;
                ++j;
            } else if (comparison < 0) {
                ++i;
            } else {
                ++j;
            }
        }
        measurement->cardinality = cardinality;
    }
    measurement->matching_ms = monotonic_ms() - phase_start;
    measurement->request_bytes = request.len;
    measurement->response_bytes = response.len;
    {
        size_t serialized_bytes;
        if (!checked_add_size(request.len, response.len, &serialized_bytes)) {
            status = PSI_ERR_INPUT;
            goto cleanup;
        }
        measurement->serialized_bytes = serialized_bytes;
    }
    measurement->payload_bytes =
        ((uint64_t) di * 2u + (uint64_t) dj) * PSI_POINT_BYTES;
    measurement->framing_overhead_bytes =
        measurement->serialized_bytes - measurement->payload_bytes;
    measurement->allocation_bytes = allocation_bytes;
    if (measurement->cardinality != oracle) {
        status = PSI_ERR_CORRECTNESS;
    }

cleanup:
    sodium_memzero(initiator_scalar, sizeof initiator_scalar);
    sodium_memzero(responder_scalar, sizeof responder_scalar);
    wipe_and_free(initiator_hashed, initiator_bytes);
    wipe_and_free(responder_hashed, responder_bytes);
    wipe_and_free(initiator_blinded, initiator_bytes);
    wipe_and_free(responder_blinded, responder_bytes);
    wipe_and_free(initiator_double_blinded, initiator_bytes);
    wipe_and_free(responder_double_blinded, responder_bytes);
    wipe_and_free(request.data, request.len);
    wipe_and_free(response.data, response.len);
    measurement->total_ms = monotonic_ms() - total_start;
    measurement->rss_after_bytes = current_rss_bytes();
    measurement->process_peak_rss_bytes = process_peak_rss_bytes();
    return status;
}

static void free_set_database(SetDatabase *sets) {
    if (sets != NULL) {
        free(sets->items);
        free(sets->owners);
        memset(sets, 0, sizeof *sets);
    }
}

static int load_sets_csv(const char *path, SetDatabase *sets) {
    FILE *input = NULL;
    Pair *pairs = NULL;
    size_t pair_count = 0;
    size_t pair_capacity = 1024;
    char *line = NULL;
    size_t line_capacity = 0;
    ssize_t line_length;
    int status = PSI_ERR_IO;

    memset(sets, 0, sizeof *sets);
    input = fopen(path, "r");
    if (input == NULL) {
        perror(path);
        return PSI_ERR_IO;
    }
    line_length = getline(&line, &line_capacity, input);
    if (line_length < 0 || strncmp(line, "owner,item", 10) != 0) {
        fprintf(stderr, "%s: expected header owner,item\n", path);
        goto cleanup;
    }
    pairs = (Pair *) malloc(pair_capacity * sizeof *pairs);
    if (pairs == NULL) {
        status = PSI_ERR_MEMORY;
        goto cleanup;
    }
    while ((line_length = getline(&line, &line_capacity, input)) >= 0) {
        unsigned long long owner;
        unsigned long long item;
        char extra;
        if (line_length == 0 || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        if (sscanf(line, "%llu,%llu %c", &owner, &item, &extra) != 2) {
            fprintf(stderr, "%s: malformed set row: %s", path, line);
            status = PSI_ERR_INPUT;
            goto cleanup;
        }
        if (pair_count == pair_capacity) {
            size_t new_capacity;
            Pair *expanded;
            if (!checked_mul_size(pair_capacity, 2, &new_capacity)) {
                status = PSI_ERR_MEMORY;
                goto cleanup;
            }
            expanded = (Pair *) realloc(pairs, new_capacity * sizeof *pairs);
            if (expanded == NULL) {
                status = PSI_ERR_MEMORY;
                goto cleanup;
            }
            pairs = expanded;
            pair_capacity = new_capacity;
        }
        pairs[pair_count].owner = (uint64_t) owner;
        pairs[pair_count].item = (uint64_t) item;
        ++pair_count;
    }
    qsort(pairs, pair_count, sizeof *pairs, compare_pair);
    if (pair_count > 0) {
        sets->items = (uint64_t *) malloc(pair_count * sizeof *sets->items);
        sets->owners = (OwnerSet *) malloc(pair_count * sizeof *sets->owners);
        if (sets->items == NULL || sets->owners == NULL) {
            status = PSI_ERR_MEMORY;
            goto cleanup;
        }
        for (size_t i = 0; i < pair_count; ++i) {
            if (i > 0 && pairs[i].owner == pairs[i - 1].owner &&
                pairs[i].item == pairs[i - 1].item) {
                fprintf(stderr,
                        "%s: duplicate (owner,item) pair (%llu,%llu)\n",
                        path, (unsigned long long) pairs[i].owner,
                        (unsigned long long) pairs[i].item);
                status = PSI_ERR_INPUT;
                goto cleanup;
            }
            if (i == 0 || pairs[i].owner != pairs[i - 1].owner) {
                OwnerSet *owner = &sets->owners[sets->owner_count++];
                owner->owner = pairs[i].owner;
                owner->offset = sets->item_count;
                owner->count = 0;
            }
            sets->items[sets->item_count++] = pairs[i].item;
            ++sets->owners[sets->owner_count - 1].count;
            if (sets->owners[sets->owner_count - 1].count > UINT32_MAX) {
                fprintf(stderr, "%s: owner set exceeds uint32 frame count\n",
                        path);
                status = PSI_ERR_INPUT;
                goto cleanup;
            }
        }
    }
    status = PSI_OK;

cleanup:
    free(line);
    free(pairs);
    fclose(input);
    if (status != PSI_OK) {
        free_set_database(sets);
    }
    return status;
}

static const uint64_t *lookup_set(const SetDatabase *sets, uint64_t owner,
                                  size_t *count) {
    size_t low = 0;
    size_t high = sets->owner_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        if (sets->owners[middle].owner < owner) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    if (low < sets->owner_count && sets->owners[low].owner == owner) {
        *count = sets->owners[low].count;
        return sets->items + sets->owners[low].offset;
    }
    *count = 0;
    return NULL;
}

static int load_sessions_csv(const char *path, Session **sessions_out,
                             size_t *session_count_out) {
    FILE *input = NULL;
    Session *sessions = NULL;
    size_t count = 0;
    size_t capacity = 1024;
    char *line = NULL;
    size_t line_capacity = 0;
    ssize_t line_length;
    int status = PSI_ERR_IO;

    *sessions_out = NULL;
    *session_count_out = 0;
    input = fopen(path, "r");
    if (input == NULL) {
        perror(path);
        return PSI_ERR_IO;
    }
    line_length = getline(&line, &line_capacity, input);
    if (line_length < 0 ||
        strncmp(line, "session_id,initiator,responder,workload", 39) != 0) {
        fprintf(stderr,
                "%s: expected header session_id,initiator,responder,workload\n",
                path);
        goto cleanup;
    }
    sessions = (Session *) malloc(capacity * sizeof *sessions);
    if (sessions == NULL) {
        status = PSI_ERR_MEMORY;
        goto cleanup;
    }
    while ((line_length = getline(&line, &line_capacity, input)) >= 0) {
        unsigned long long session_id;
        unsigned long long initiator;
        unsigned long long responder;
        char workload[MAX_WORKLOAD_BYTES + 2u] = {0};
        if (line_length == 0 || line[0] == '\n' || line[0] == '\r') {
            continue;
        }
        if (sscanf(line, "%llu,%llu,%llu,%96[^\r\n]", &session_id,
                   &initiator, &responder, workload) != 4 ||
            workload[0] == '\0' || strchr(workload, ',') != NULL ||
            strlen(workload) > MAX_WORKLOAD_BYTES) {
            fprintf(stderr, "%s: malformed session row: %s", path, line);
            status = PSI_ERR_INPUT;
            goto cleanup;
        }
        if (count == capacity) {
            size_t new_capacity;
            Session *expanded;
            if (!checked_mul_size(capacity, 2, &new_capacity)) {
                status = PSI_ERR_MEMORY;
                goto cleanup;
            }
            expanded = (Session *) realloc(sessions,
                                            new_capacity * sizeof *sessions);
            if (expanded == NULL) {
                status = PSI_ERR_MEMORY;
                goto cleanup;
            }
            sessions = expanded;
            capacity = new_capacity;
        }
        sessions[count].session_id = (uint64_t) session_id;
        sessions[count].initiator = (uint64_t) initiator;
        sessions[count].responder = (uint64_t) responder;
        snprintf(sessions[count].workload,
                 sizeof sessions[count].workload, "%s", workload);
        ++count;
    }
    if (count == 0) {
        fprintf(stderr, "%s: no sessions found\n", path);
        status = PSI_ERR_INPUT;
        goto cleanup;
    }
    *sessions_out = sessions;
    *session_count_out = count;
    sessions = NULL;
    status = PSI_OK;

cleanup:
    free(line);
    free(sessions);
    fclose(input);
    return status;
}

/* This non-cryptographic PRNG is intentionally scoped to public task order.
 * It never supplies scalars, blinds, salts, or shuffle randomness. */
static uint64_t scheduling_prng(uint64_t *state) {
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static void shuffle_task_order(size_t *order, size_t count, uint64_t seed) {
    uint64_t state = seed != 0 ? seed : UINT64_C(0x9e3779b97f4a7c15);
    if (count < 2) {
        return;
    }
    for (size_t i = count - 1; i > 0; --i) {
        size_t j = (size_t) (scheduling_prng(&state) % (i + 1));
        size_t temporary = order[i];
        order[i] = order[j];
        order[j] = temporary;
    }
}

static void *batch_worker(void *opaque) {
    WorkerContext *context = (WorkerContext *) opaque;
    for (;;) {
        size_t schedule_position =
            atomic_fetch_add(context->next_position, (size_t) 1);
        size_t task_index;
        size_t session_index;
        size_t repetition;
        const Session *session;
        const uint64_t *initiator_set;
        const uint64_t *responder_set;
        size_t initiator_count;
        size_t responder_count;
        TaskResult *result;
        int status;

        if (schedule_position >= context->task_count) {
            break;
        }
        task_index = context->order[schedule_position];
        session_index = task_index % context->session_count;
        repetition = task_index / context->session_count;
        session = &context->sessions[session_index];
        initiator_set = lookup_set(context->sets, session->initiator,
                                   &initiator_count);
        responder_set = lookup_set(context->sets, session->responder,
                                   &responder_count);
        result = &context->results[task_index];
        result->task_index = task_index;
        result->schedule_position = schedule_position;
        result->session_index = session_index;
        result->repetition = repetition;
        status = psi_ca_session(initiator_set, initiator_count,
                                responder_set, responder_count,
                                session->session_id, &result->measurement);
        result->status = status;
        if (status != PSI_OK) {
            atomic_store(context->failed, status);
        }
    }
    return NULL;
}

static double quantile_sorted(const double *values, size_t count, double q) {
    double position;
    size_t low;
    size_t high;
    double fraction;
    if (count == 0) {
        return 0.0;
    }
    position = q * (double) (count - 1);
    low = (size_t) position;
    high = low + (low + 1 < count ? 1u : 0u);
    fraction = position - (double) low;
    return values[low] * (1.0 - fraction) + values[high] * fraction;
}

static int write_raw_csv(const char *path, const Session *sessions,
                         const TaskResult *results, size_t task_count) {
    FILE *output = fopen(path, "w");
    if (output == NULL) {
        perror(path);
        return PSI_ERR_IO;
    }
    fputs("task_index,schedule_position,session_id,workload,rep,initiator,"
          "responder,d_i,d_j,cardinality,plaintext_cardinality,correct,status,"
          "latency_ms,scalar_rng_ms,hash_to_group_ms,initiator_blind_ms,"
          "request_serialize_ms,responder_parse_ms,responder_compute_ms,"
          "responder_shuffle_ms,response_serialize_ms,initiator_parse_ms,"
          "initiator_finalize_ms,matching_ms,group_payload_bytes,"
          "framing_overhead_bytes,round1_serialized_bytes,"
          "round2_serialized_bytes,total_serialized_bytes,allocation_bytes,"
          "rss_before_bytes,rss_after_bytes,process_peak_rss_bytes\n", output);
    for (size_t i = 0; i < task_count; ++i) {
        const TaskResult *result = &results[i];
        const Session *session = &sessions[result->session_index];
        const psi_measurement *m = &result->measurement;
        fprintf(output,
                "%zu,%zu,%llu,%s,%zu,%llu,%llu,%llu,%llu,%llu,%llu,%d,%d,"
                "%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,"
                "%.9f,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",
                result->task_index, result->schedule_position,
                (unsigned long long) session->session_id, session->workload,
                result->repetition,
                (unsigned long long) session->initiator,
                (unsigned long long) session->responder,
                (unsigned long long) m->initiator_items,
                (unsigned long long) m->responder_items,
                (unsigned long long) m->cardinality,
                (unsigned long long) m->plaintext_cardinality,
                m->cardinality == m->plaintext_cardinality,
                result->status, m->total_ms, m->scalar_rng_ms,
                m->hash_to_group_ms, m->initiator_blind_ms,
                m->request_serialize_ms, m->responder_parse_ms,
                m->responder_compute_ms, m->responder_shuffle_ms,
                m->response_serialize_ms, m->initiator_parse_ms,
                m->initiator_finalize_ms, m->matching_ms,
                (unsigned long long) m->payload_bytes,
                (unsigned long long) m->framing_overhead_bytes,
                (unsigned long long) m->request_bytes,
                (unsigned long long) m->response_bytes,
                (unsigned long long) m->serialized_bytes,
                (unsigned long long) m->allocation_bytes,
                (unsigned long long) m->rss_before_bytes,
                (unsigned long long) m->rss_after_bytes,
                (unsigned long long) m->process_peak_rss_bytes);
    }
    if (fclose(output) != 0) {
        return PSI_ERR_IO;
    }
    return PSI_OK;
}

static int write_summary_csv(const char *path, unsigned int threads,
                             size_t repetitions, size_t warmup,
                             uint64_t order_seed, size_t session_count,
                             size_t task_count, double wall_seconds,
                             const TaskResult *results,
                             uint64_t baseline_rss, uint64_t peak_rss) {
    FILE *output = NULL;
    double *latencies = NULL;
    double median;
    double p95;
    double maximum;

    latencies = (double *) malloc(task_count * sizeof *latencies);
    if (latencies == NULL) {
        return PSI_ERR_MEMORY;
    }
    for (size_t i = 0; i < task_count; ++i) {
        latencies[i] = results[i].measurement.total_ms;
    }
    qsort(latencies, task_count, sizeof *latencies, compare_double);
    median = quantile_sorted(latencies, task_count, 0.50);
    p95 = quantile_sorted(latencies, task_count, 0.95);
    maximum = task_count > 0 ? latencies[task_count - 1] : 0.0;
    output = fopen(path, "w");
    if (output == NULL) {
        perror(path);
        free(latencies);
        return PSI_ERR_IO;
    }
    fputs("backend_version,libsodium_version,group,hash_to_group,frame_version,"
          "threads,repetitions,warmup_per_session,order_seed,session_rows,"
          "completed_calls,wall_s,throughput_calls_per_s,latency_median_ms,"
          "latency_p95_ms,latency_max_ms,baseline_rss_bytes,peak_rss_bytes,"
          "incremental_peak_rss_bytes,network_transport\n", output);
    fprintf(output,
            "%s,%s,Ristretto255,BLAKE2b-512-domain-separated,%u,%u,%zu,%zu,"
            "%llu,%zu,%zu,%.9f,%.9f,%.9f,%.9f,%.9f,%llu,%llu,%llu,none\n",
            PSI_BACKEND_VERSION, sodium_version_string(), FRAME_VERSION,
            threads, repetitions, warmup,
            (unsigned long long) order_seed, session_count, task_count,
            wall_seconds,
            wall_seconds > 0.0 ? (double) task_count / wall_seconds : 0.0,
            median, p95, maximum,
            (unsigned long long) baseline_rss,
            (unsigned long long) peak_rss,
            (unsigned long long) (peak_rss > baseline_rss
                                      ? peak_rss - baseline_rss
                                      : 0));
    free(latencies);
    if (fclose(output) != 0) {
        return PSI_ERR_IO;
    }
    return PSI_OK;
}

static int run_batch(const char *sets_path, const char *sessions_path,
                     const char *raw_path, const char *summary_path,
                     unsigned int threads, size_t repetitions,
                     size_t warmup_per_session, uint64_t order_seed) {
    SetDatabase sets = {0};
    Session *sessions = NULL;
    size_t session_count = 0;
    size_t task_count;
    size_t *order = NULL;
    TaskResult *results = NULL;
    pthread_t *worker_threads = NULL;
    WorkerContext *contexts = NULL;
    atomic_size_t next_position;
    atomic_int failed;
    uint64_t baseline_rss;
    uint64_t peak_rss;
    double wall_start;
    double wall_seconds;
    int status;

    if (threads == 0 || threads > 1024 || repetitions == 0 ||
        !checked_mul_size(repetitions, (size_t) 1, &repetitions)) {
        return PSI_ERR_INPUT;
    }
    status = load_sets_csv(sets_path, &sets);
    if (status != PSI_OK) {
        return status;
    }
    status = load_sessions_csv(sessions_path, &sessions, &session_count);
    if (status != PSI_OK) {
        free_set_database(&sets);
        return status;
    }
    if (!checked_mul_size(session_count, repetitions, &task_count)) {
        status = PSI_ERR_MEMORY;
        goto cleanup;
    }

    /* Warmups execute the identical protocol with fresh private randomness and
     * are intentionally excluded from raw rows and the wall-throughput timer. */
    for (size_t warm = 0; warm < warmup_per_session; ++warm) {
        for (size_t i = 0; i < session_count; ++i) {
            const uint64_t *initiator_set;
            const uint64_t *responder_set;
            size_t initiator_count;
            size_t responder_count;
            psi_measurement measurement;
            initiator_set = lookup_set(&sets, sessions[i].initiator,
                                       &initiator_count);
            responder_set = lookup_set(&sets, sessions[i].responder,
                                       &responder_count);
            status = psi_ca_session(initiator_set, initiator_count,
                                    responder_set, responder_count,
                                    sessions[i].session_id, &measurement);
            if (status != PSI_OK) {
                fprintf(stderr, "warmup session %zu failed: %s\n", i,
                        psi_backend_error(status));
                goto cleanup;
            }
        }
    }

    order = (size_t *) malloc(task_count * sizeof *order);
    results = (TaskResult *) calloc(task_count, sizeof *results);
    worker_threads = (pthread_t *) malloc((size_t) threads *
                                          sizeof *worker_threads);
    contexts = (WorkerContext *) calloc((size_t) threads, sizeof *contexts);
    if (order == NULL || results == NULL || worker_threads == NULL ||
        contexts == NULL) {
        status = PSI_ERR_MEMORY;
        goto cleanup;
    }
    for (size_t i = 0; i < task_count; ++i) {
        order[i] = i;
    }
    shuffle_task_order(order, task_count, order_seed);
    atomic_init(&next_position, 0);
    atomic_init(&failed, PSI_OK);
    baseline_rss = current_rss_bytes();
    wall_start = monotonic_ms();
    for (unsigned int i = 0; i < threads; ++i) {
        contexts[i].sets = &sets;
        contexts[i].sessions = sessions;
        contexts[i].session_count = session_count;
        contexts[i].order = order;
        contexts[i].task_count = task_count;
        contexts[i].next_position = &next_position;
        contexts[i].failed = &failed;
        contexts[i].results = results;
        if (pthread_create(&worker_threads[i], NULL, batch_worker,
                           &contexts[i]) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            atomic_store(&failed, PSI_ERR_IO);
            threads = i;
            break;
        }
    }
    for (unsigned int i = 0; i < threads; ++i) {
        pthread_join(worker_threads[i], NULL);
    }
    wall_seconds = (monotonic_ms() - wall_start) / 1000.0;
    peak_rss = process_peak_rss_bytes();
    status = atomic_load(&failed);
    if (status != PSI_OK) {
        fprintf(stderr, "batch failed: %s\n", psi_backend_error(status));
        goto cleanup;
    }
    status = write_raw_csv(raw_path, sessions, results, task_count);
    if (status != PSI_OK) {
        goto cleanup;
    }
    status = write_summary_csv(summary_path, threads, repetitions,
                               warmup_per_session, order_seed, session_count,
                               task_count, wall_seconds, results,
                               baseline_rss, peak_rss);

cleanup:
    free(contexts);
    free(worker_threads);
    free(results);
    free(order);
    free(sessions);
    free_set_database(&sets);
    return status;
}

static int run_selftest(void) {
    const uint64_t a[] = {1, 2, 3, 8};
    const uint64_t b[] = {2, 3, 4, 9};
    const uint64_t c[] = {10, 11};
    psi_measurement first;
    psi_measurement second;
    psi_measurement disjoint;
    psi_measurement empty;
    unsigned char valid_point[PSI_POINT_BYTES];
    unsigned char identity[PSI_POINT_BYTES] = {0};
    Frame frame = {0};
    ParsedFrame parsed = {0};
    int status;

    status = psi_ca_session(a, 4, b, 4, UINT64_C(1001), &first);
    if (status != PSI_OK || first.cardinality != 2 ||
        first.plaintext_cardinality != 2 ||
        first.request_bytes != PSI_FRAME_HEADER_BYTES + 4 * PSI_POINT_BYTES ||
        first.response_bytes !=
            PSI_FRAME_HEADER_BYTES + 8 * PSI_POINT_BYTES) {
        fprintf(stderr, "selftest overlap failed: %s\n",
                psi_backend_error(status));
        return 1;
    }
    status = psi_ca_session(a, 4, b, 4, UINT64_C(1001), &second);
    if (status != PSI_OK || second.cardinality != 2) {
        fprintf(stderr, "selftest fresh-randomness rerun failed\n");
        return 1;
    }
    status = psi_ca_session(a, 4, c, 2, UINT64_C(1002), &disjoint);
    if (status != PSI_OK || disjoint.cardinality != 0) {
        fprintf(stderr, "selftest disjoint failed\n");
        return 1;
    }
    status = psi_ca_session(NULL, 0, b, 4, UINT64_C(1003), &empty);
    if (status != PSI_OK || empty.cardinality != 0) {
        fprintf(stderr, "selftest empty set failed\n");
        return 1;
    }
    if (hash_identifier_to_group(UINT64_C(42), valid_point) != PSI_OK ||
        build_frame(FRAME_REQUEST, UINT64_C(2001), valid_point, 1,
                    NULL, 0, &frame) != PSI_OK ||
        parse_frame(&frame, FRAME_REQUEST, UINT64_C(2001), 1, 0,
                    &parsed) != PSI_OK) {
        fprintf(stderr, "selftest valid frame failed\n");
        wipe_and_free(frame.data, frame.len);
        sodium_memzero(valid_point, sizeof valid_point);
        return 1;
    }
    memcpy(frame.data + PSI_FRAME_HEADER_BYTES, identity, sizeof identity);
    if (parse_frame(&frame, FRAME_REQUEST, UINT64_C(2001), 1, 0,
                    &parsed) != PSI_ERR_POINT) {
        fprintf(stderr, "selftest identity rejection failed\n");
        wipe_and_free(frame.data, frame.len);
        sodium_memzero(valid_point, sizeof valid_point);
        return 1;
    }
    wipe_and_free(frame.data, frame.len);
    sodium_memzero(valid_point, sizeof valid_point);
    sodium_memzero(identity, sizeof identity);
    printf("selftest: PASS\n");
    printf("backend_version=%s libsodium_version=%s group=Ristretto255 "
           "hash=BLAKE2b-512-domain-separated frame_version=%u\n",
           PSI_BACKEND_VERSION, sodium_version_string(), FRAME_VERSION);
    printf("overlap_cardinality=%llu request_bytes=%llu response_bytes=%llu "
           "serialized_bytes=%llu\n",
           (unsigned long long) first.cardinality,
           (unsigned long long) first.request_bytes,
           (unsigned long long) first.response_bytes,
           (unsigned long long) first.serialized_bytes);
    return 0;
}

static bool parse_size_argument(const char *text, size_t *value) {
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' ||
        parsed > (unsigned long long) SIZE_MAX) {
        return false;
    }
    *value = (size_t) parsed;
    return true;
}

static bool parse_u64_argument(const char *text, uint64_t *value) {
    char *end = NULL;
    unsigned long long parsed;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *value = (uint64_t) parsed;
    return true;
}

static void print_usage(const char *program) {
    fprintf(stderr,
            "Usage:\n"
            "  %s selftest\n"
            "  %s batch SETS.csv SESSIONS.csv RAW.csv SUMMARY.csv "
            "THREADS REPS WARMUP ORDER_SEED\n\n"
            "SETS.csv header: owner,item\n"
            "SESSIONS.csv header: session_id,initiator,responder,workload\n"
            "ORDER_SEED controls public task order only; all protocol secrets "
            "use libsodium CSPRNG.\n",
            program, program);
}

int main(int argc, char **argv) {
    int status;
    if (psi_backend_init() != PSI_OK) {
        fprintf(stderr, "libsodium initialization failed\n");
        return 1;
    }
    if (argc == 2 && strcmp(argv[1], "selftest") == 0) {
        return run_selftest();
    }
    if (argc == 10 && strcmp(argv[1], "batch") == 0) {
        size_t threads;
        size_t repetitions;
        size_t warmup;
        uint64_t order_seed;
        if (!parse_size_argument(argv[6], &threads) || threads > UINT32_MAX ||
            !parse_size_argument(argv[7], &repetitions) ||
            !parse_size_argument(argv[8], &warmup) ||
            !parse_u64_argument(argv[9], &order_seed)) {
            print_usage(argv[0]);
            return 1;
        }
        status = run_batch(argv[2], argv[3], argv[4], argv[5],
                           (unsigned int) threads, repetitions, warmup,
                           order_seed);
        if (status != PSI_OK) {
            fprintf(stderr, "batch: %s\n", psi_backend_error(status));
            return 1;
        }
        return 0;
    }
    print_usage(argv[0]);
    return 1;
}
