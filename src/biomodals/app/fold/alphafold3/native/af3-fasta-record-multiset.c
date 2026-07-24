
#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PROGRAM_VERSION "af3-fasta-record-multiset-v1"
#define CANONICALIZATION \
    "full-header-and-sequence-case-sensitive-line-ending-independent-v1"
#define AGGREGATE "sha256-lane-sum-xor-and-square-sum-with-counts-v1"
#define IO_BUFFER_BYTES (8U * 1024U * 1024U)
#define ERROR_BYTES 2048U

static const unsigned char RECORD_DOMAIN[] = "AF3_FASTA_RECORD_V1";

typedef struct {
    uint64_t records;
    uint64_t header_bytes;
    uint64_t sequence_bytes;
    uint64_t sum[4];
    uint64_t xor_value[4];
    uint64_t square_sum[4];
} aggregate_t;

typedef struct {
    char **paths;
    size_t path_count;
    size_t next_path;
    pthread_mutex_t mutex;
    aggregate_t total;
    bool failed;
    char error[ERROR_BYTES];
} work_queue_t;

typedef struct {
    uint32_t state[8];
    uint64_t total_bytes;
    unsigned char block[64];
    size_t block_size;
} sha256_context_t;

static const uint32_t SHA256_CONSTANTS[64] = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

static uint32_t rotate_right_u32(uint32_t value, unsigned int bits) {
    return (value >> bits) | (value << (32U - bits));
}

static uint32_t load_u32_be(const unsigned char *bytes) {
    return ((uint32_t) bytes[0] << 24)
        | ((uint32_t) bytes[1] << 16)
        | ((uint32_t) bytes[2] << 8)
        | (uint32_t) bytes[3];
}

static void store_u32_be(unsigned char *bytes, uint32_t value) {
    bytes[0] = (unsigned char) (value >> 24);
    bytes[1] = (unsigned char) (value >> 16);
    bytes[2] = (unsigned char) (value >> 8);
    bytes[3] = (unsigned char) value;
}

static void sha256_transform(
    sha256_context_t *context,
    const unsigned char block[64]
) {
    uint32_t words[64];
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
    uint32_t d = 0;
    uint32_t e = 0;
    uint32_t f = 0;
    uint32_t g = 0;
    uint32_t h = 0;

    for (size_t index = 0; index < 16; ++index) {
        words[index] = load_u32_be(block + index * 4);
    }
    for (size_t index = 16; index < 64; ++index) {
        uint32_t left = words[index - 15];
        uint32_t right = words[index - 2];
        uint32_t sigma_zero = rotate_right_u32(left, 7)
            ^ rotate_right_u32(left, 18)
            ^ (left >> 3);
        uint32_t sigma_one = rotate_right_u32(right, 17)
            ^ rotate_right_u32(right, 19)
            ^ (right >> 10);
        words[index] = words[index - 16]
            + sigma_zero
            + words[index - 7]
            + sigma_one;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];
    for (size_t index = 0; index < 64; ++index) {
        uint32_t choice = (e & f) ^ ((~e) & g);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t upper_zero = rotate_right_u32(a, 2)
            ^ rotate_right_u32(a, 13)
            ^ rotate_right_u32(a, 22);
        uint32_t upper_one = rotate_right_u32(e, 6)
            ^ rotate_right_u32(e, 11)
            ^ rotate_right_u32(e, 25);
        uint32_t temporary_one = h
            + upper_one
            + choice
            + SHA256_CONSTANTS[index]
            + words[index];
        uint32_t temporary_two = upper_zero + majority;

        h = g;
        g = f;
        f = e;
        e = d + temporary_one;
        d = c;
        c = b;
        b = a;
        a = temporary_one + temporary_two;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void sha256_init(sha256_context_t *context) {
    static const uint32_t initial_state[8] = {
        0x6a09e667U,
        0xbb67ae85U,
        0x3c6ef372U,
        0xa54ff53aU,
        0x510e527fU,
        0x9b05688cU,
        0x1f83d9abU,
        0x5be0cd19U,
    };
    memcpy(context->state, initial_state, sizeof(initial_state));
    context->total_bytes = 0;
    context->block_size = 0;
}

static void sha256_update(
    sha256_context_t *context,
    const void *data,
    size_t size
) {
    const unsigned char *input = data;
    context->total_bytes += (uint64_t) size;
    while (size > 0) {
        size_t available = sizeof(context->block) - context->block_size;
        size_t copy_size = size < available ? size : available;
        memcpy(context->block + context->block_size, input, copy_size);
        context->block_size += copy_size;
        input += copy_size;
        size -= copy_size;
        if (context->block_size == sizeof(context->block)) {
            sha256_transform(context, context->block);
            context->block_size = 0;
        }
    }
}

static void sha256_final(
    sha256_context_t *context,
    unsigned char digest[32]
) {
    uint64_t bit_length = context->total_bytes * 8;
    context->block[context->block_size++] = 0x80;
    if (context->block_size > 56) {
        while (context->block_size < sizeof(context->block)) {
            context->block[context->block_size++] = 0;
        }
        sha256_transform(context, context->block);
        context->block_size = 0;
    }
    while (context->block_size < 56) {
        context->block[context->block_size++] = 0;
    }
    for (size_t index = 0; index < 8; ++index) {
        context->block[63 - index] = (unsigned char) (bit_length >> (index * 8));
    }
    sha256_transform(context, context->block);
    for (size_t index = 0; index < 8; ++index) {
        store_u32_be(digest + index * 4, context->state[index]);
    }
}

static uint64_t load_u64_le(const unsigned char *bytes) {
    uint64_t value = 0;
    for (size_t index = 0; index < 8; ++index) {
        value |= ((uint64_t) bytes[index]) << (index * 8);
    }
    return value;
}

static void store_u64_le(unsigned char *bytes, uint64_t value) {
    for (size_t index = 0; index < 8; ++index) {
        bytes[index] = (unsigned char) (value >> (index * 8));
    }
}

static int checked_add(
    uint64_t left,
    uint64_t right,
    uint64_t *result,
    const char *field,
    char *error,
    size_t error_bytes
) {
    if (UINT64_MAX - left < right) {
        (void) snprintf(error, error_bytes, "%s exceeds uint64", field);
        return -1;
    }
    *result = left + right;
    return 0;
}

static int merge_aggregate(
    aggregate_t *destination,
    const aggregate_t *source,
    char *error,
    size_t error_bytes
) {
    if (
        checked_add(
            destination->records,
            source->records,
            &destination->records,
            "record count",
            error,
            error_bytes
        ) != 0
        || checked_add(
            destination->header_bytes,
            source->header_bytes,
            &destination->header_bytes,
            "header bytes",
            error,
            error_bytes
        ) != 0
        || checked_add(
            destination->sequence_bytes,
            source->sequence_bytes,
            &destination->sequence_bytes,
            "sequence bytes",
            error,
            error_bytes
        ) != 0
    ) {
        return -1;
    }
    for (size_t lane = 0; lane < 4; ++lane) {
        destination->sum[lane] += source->sum[lane];
        destination->xor_value[lane] ^= source->xor_value[lane];
        destination->square_sum[lane] += source->square_sum[lane];
    }
    return 0;
}

static void start_record(
    sha256_context_t *context,
    const char *header,
    size_t header_size
) {
    static const unsigned char header_tag = 1;
    static const unsigned char sequence_tag = 2;

    sha256_init(context);
    sha256_update(context, RECORD_DOMAIN, sizeof(RECORD_DOMAIN) - 1);
    sha256_update(context, &header_tag, sizeof(header_tag));
    sha256_update(context, header, header_size);
    sha256_update(context, &sequence_tag, sizeof(sequence_tag));
}

static int finish_record(
    sha256_context_t *context,
    uint64_t header_bytes,
    uint64_t sequence_bytes,
    aggregate_t *aggregate,
    char *error,
    size_t error_bytes
) {
    unsigned char lengths[16];
    unsigned char digest[32];

    store_u64_le(lengths, header_bytes);
    store_u64_le(lengths + 8, sequence_bytes);
    sha256_update(context, lengths, sizeof(lengths));
    sha256_final(context, digest);
    if (
        checked_add(
            aggregate->records,
            1,
            &aggregate->records,
            "record count",
            error,
            error_bytes
        ) != 0
        || checked_add(
            aggregate->header_bytes,
            header_bytes,
            &aggregate->header_bytes,
            "header bytes",
            error,
            error_bytes
        ) != 0
        || checked_add(
            aggregate->sequence_bytes,
            sequence_bytes,
            &aggregate->sequence_bytes,
            "sequence bytes",
            error,
            error_bytes
        ) != 0
    ) {
        return -1;
    }
    for (size_t lane = 0; lane < 4; ++lane) {
        uint64_t value = load_u64_le(digest + lane * 8);
        aggregate->sum[lane] += value;
        aggregate->xor_value[lane] ^= value;
        aggregate->square_sum[lane] += value * value;
    }
    return 0;
}

static size_t line_content_size(const char *line, ssize_t read_size) {
    size_t size = (size_t) read_size;
    if (size > 0 && line[size - 1] == '\n') {
        --size;
    }
    if (size > 0 && line[size - 1] == '\r') {
        --size;
    }
    return size;
}

static int digest_file(
    const char *path,
    aggregate_t *aggregate,
    char *error,
    size_t error_bytes
) {
    FILE *input = NULL;
    sha256_context_t context = {0};
    char *line = NULL;
    char *io_buffer = NULL;
    size_t line_capacity = 0;
    ssize_t read_size = 0;
    bool active_record = false;
    uint64_t header_bytes = 0;
    uint64_t sequence_bytes = 0;
    int result = -1;

    input = fopen(path, "rb");
    if (input == NULL) {
        (void) snprintf(
            error,
            error_bytes,
            "cannot open %s: %s",
            path,
            strerror(errno)
        );
        goto cleanup;
    }
    io_buffer = malloc(IO_BUFFER_BYTES);
    if (io_buffer == NULL) {
        (void) snprintf(error, error_bytes, "cannot allocate input buffer");
        goto cleanup;
    }
    if (setvbuf(input, io_buffer, _IOFBF, IO_BUFFER_BYTES) != 0) {
        (void) snprintf(error, error_bytes, "cannot configure input buffer");
        goto cleanup;
    }
    while ((read_size = getline(&line, &line_capacity, input)) >= 0) {
        size_t content_size = line_content_size(line, read_size);
        if (content_size > 0 && line[0] == '>') {
            if (
                active_record
                && finish_record(
                    &context,
                    header_bytes,
                    sequence_bytes,
                    aggregate,
                    error,
                    error_bytes
                ) != 0
            ) {
                goto cleanup;
            }
            header_bytes = (uint64_t) (content_size - 1);
            sequence_bytes = 0;
            start_record(&context, line + 1, content_size - 1);
            active_record = true;
            continue;
        }
        if (!active_record) {
            if (content_size == 0) {
                continue;
            }
            (void) snprintf(
                error,
                error_bytes,
                "nonempty data precedes first FASTA header in %s",
                path
            );
            goto cleanup;
        }
        if (UINT64_MAX - sequence_bytes < content_size) {
            (void) snprintf(
                error,
                error_bytes,
                "sequence length exceeds uint64 in %s",
                path
            );
            goto cleanup;
        }
        sha256_update(&context, line, content_size);
        sequence_bytes += (uint64_t) content_size;
    }
    if (ferror(input)) {
        (void) snprintf(
            error,
            error_bytes,
            "error reading %s: %s",
            path,
            strerror(errno)
        );
        goto cleanup;
    }
    if (!active_record) {
        (void) snprintf(error, error_bytes, "no FASTA records in %s", path);
        goto cleanup;
    }
    if (
        finish_record(
            &context,
            header_bytes,
            sequence_bytes,
            aggregate,
            error,
            error_bytes
        ) != 0
    ) {
        goto cleanup;
    }
    result = 0;

cleanup:
    free(line);
    if (input != NULL && fclose(input) != 0 && result == 0) {
        (void) snprintf(
            error,
            error_bytes,
            "cannot close %s: %s",
            path,
            strerror(errno)
        );
        result = -1;
    }
    free(io_buffer);
    return result;
}

static void *worker_main(void *argument) {
    work_queue_t *queue = argument;
    for (;;) {
        size_t path_index = 0;
        aggregate_t local = {0};
        char error[ERROR_BYTES] = {0};

        (void) pthread_mutex_lock(&queue->mutex);
        if (queue->failed || queue->next_path >= queue->path_count) {
            (void) pthread_mutex_unlock(&queue->mutex);
            return NULL;
        }
        path_index = queue->next_path++;
        (void) pthread_mutex_unlock(&queue->mutex);

        if (
            digest_file(
                queue->paths[path_index],
                &local,
                error,
                sizeof(error)
            ) != 0
        ) {
            (void) pthread_mutex_lock(&queue->mutex);
            if (!queue->failed) {
                queue->failed = true;
                (void) snprintf(
                    queue->error,
                    sizeof(queue->error),
                    "%s",
                    error
                );
            }
            (void) pthread_mutex_unlock(&queue->mutex);
            return NULL;
        }

        (void) pthread_mutex_lock(&queue->mutex);
        if (
            !queue->failed
            && merge_aggregate(
                &queue->total,
                &local,
                queue->error,
                sizeof(queue->error)
            ) != 0
        ) {
            queue->failed = true;
        }
        (void) pthread_mutex_unlock(&queue->mutex);
    }
}

static int parse_thread_count(const char *text, size_t *thread_count) {
    char *end = NULL;
    unsigned long value = 0;

    errno = 0;
    value = strtoul(text, &end, 10);
    if (
        errno != 0
        || end == text
        || *end != '\0'
        || value == 0
        || value > 64
    ) {
        return -1;
    }
    *thread_count = (size_t) value;
    return 0;
}

static int write_report(
    const char *path,
    size_t path_count,
    size_t thread_count,
    const aggregate_t *aggregate
) {
    FILE *output = fopen(path, "wx");
    int close_result = 0;
    if (output == NULL) {
        (void) fprintf(
            stderr,
            "cannot create %s: %s\n",
            path,
            strerror(errno)
        );
        return -1;
    }
    if (
        fprintf(
            output,
            "{\n"
            "  \"version\":\"%s\",\n"
            "  \"canonicalization\":\"%s\",\n"
            "  \"digest\":\"sha256\",\n"
            "  \"aggregate\":\"%s\",\n"
            "  \"files\":%zu,\n"
            "  \"threads\":%zu,\n"
            "  \"records\":%" PRIu64 ",\n"
            "  \"header_bytes\":%" PRIu64 ",\n"
            "  \"sequence_bytes\":%" PRIu64 ",\n"
            "  \"sum_sha256_lanes\":["
            "\"%016" PRIx64 "\",\"%016" PRIx64 "\","
            "\"%016" PRIx64 "\",\"%016" PRIx64 "\"],\n"
            "  \"xor_sha256_lanes\":["
            "\"%016" PRIx64 "\",\"%016" PRIx64 "\","
            "\"%016" PRIx64 "\",\"%016" PRIx64 "\"],\n"
            "  \"sum_square_sha256_lanes\":["
            "\"%016" PRIx64 "\",\"%016" PRIx64 "\","
            "\"%016" PRIx64 "\",\"%016" PRIx64 "\"]\n"
            "}\n",
            PROGRAM_VERSION,
            CANONICALIZATION,
            AGGREGATE,
            path_count,
            thread_count,
            aggregate->records,
            aggregate->header_bytes,
            aggregate->sequence_bytes,
            aggregate->sum[0],
            aggregate->sum[1],
            aggregate->sum[2],
            aggregate->sum[3],
            aggregate->xor_value[0],
            aggregate->xor_value[1],
            aggregate->xor_value[2],
            aggregate->xor_value[3],
            aggregate->square_sum[0],
            aggregate->square_sum[1],
            aggregate->square_sum[2],
            aggregate->square_sum[3]
        ) < 0
    ) {
        (void) fprintf(stderr, "cannot write %s\n", path);
        (void) fclose(output);
        return -1;
    }
    if (fflush(output) != 0) {
        (void) fprintf(stderr, "cannot flush %s: %s\n", path, strerror(errno));
        (void) fclose(output);
        return -1;
    }
    close_result = fclose(output);
    if (close_result != 0) {
        (void) fprintf(stderr, "cannot close %s: %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

int main(int argc, char **argv) {
    size_t requested_threads = 0;
    size_t thread_count = 0;
    size_t path_count = 0;
    size_t created_threads = 0;
    pthread_t *threads = NULL;
    work_queue_t queue = {0};
    int result = EXIT_FAILURE;

    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        (void) printf("%s\n", PROGRAM_VERSION);
        return EXIT_SUCCESS;
    }
    if (
        argc < 4
        || parse_thread_count(argv[1], &requested_threads) != 0
    ) {
        (void) fprintf(
            stderr,
            "usage: %s THREADS OUTPUT_JSON FASTA [FASTA ...]\n",
            argv[0]
        );
        return EXIT_FAILURE;
    }
    path_count = (size_t) argc - 3;
    thread_count = requested_threads < path_count
        ? requested_threads
        : path_count;
    queue.paths = argv + 3;
    queue.path_count = path_count;
    if (pthread_mutex_init(&queue.mutex, NULL) != 0) {
        (void) fprintf(stderr, "cannot initialize worker mutex\n");
        return EXIT_FAILURE;
    }
    threads = calloc(thread_count, sizeof(*threads));
    if (threads == NULL) {
        (void) fprintf(stderr, "cannot allocate worker handles\n");
        goto cleanup;
    }
    for (; created_threads < thread_count; ++created_threads) {
        int create_result = pthread_create(
            &threads[created_threads],
            NULL,
            worker_main,
            &queue
        );
        if (create_result != 0) {
            (void) pthread_mutex_lock(&queue.mutex);
            queue.failed = true;
            (void) snprintf(
                queue.error,
                sizeof(queue.error),
                "cannot create worker: %s",
                strerror(create_result)
            );
            (void) pthread_mutex_unlock(&queue.mutex);
            break;
        }
    }
    for (size_t index = 0; index < created_threads; ++index) {
        int join_result = pthread_join(threads[index], NULL);
        if (join_result != 0) {
            (void) fprintf(
                stderr,
                "cannot join worker: %s\n",
                strerror(join_result)
            );
            return EXIT_FAILURE;
        }
    }
    if (queue.failed) {
        (void) fprintf(stderr, "%s\n", queue.error);
        goto cleanup;
    }
    if (
        write_report(
            argv[2],
            path_count,
            thread_count,
            &queue.total
        ) != 0
    ) {
        goto cleanup;
    }
    result = EXIT_SUCCESS;

cleanup:
    free(threads);
    (void) pthread_mutex_destroy(&queue.mutex);
    return result;
}
