
#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#define _XOPEN_SOURCE 700

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define PROGRAM_VERSION "af3-fasta-two-pass-v2"
#define INDEX_MAGIC UINT64_C(0x4146334f46465331)
#define INDEX_VERSION UINT64_C(1)
#define SCAN_BUFFER_BYTES (64U * 1024U * 1024U)
#define OFFSET_BUFFER_RECORDS (1024U * 1024U)

struct index_header {
    uint64_t magic;
    uint64_t version;
    uint64_t source_size;
    uint64_t output_size;
    uint64_t record_count;
    uint64_t maximum_record_size;
};

struct offset_writer {
    int fd;
    uint64_t *values;
    size_t count;
    uint64_t total;
};

struct read_task {
    uint64_t source_offset;
    size_t length;
    size_t destination_offset;
    bool append_newline;
};

struct read_pool {
    int source_fd;
    size_t thread_count;
    pthread_t *threads;
    pthread_barrier_t start_barrier;
    pthread_barrier_t finish_barrier;
    atomic_size_t next_task;
    atomic_int failed;
    atomic_int error_number;
    atomic_bool stop;
    struct read_task *tasks;
    size_t task_count;
    unsigned char *buffer;
};

static void fail_message(const char *message) {
    fprintf(stderr, "af3-fasta-two-pass: %s\n", message);
    exit(EXIT_FAILURE);
}

static void fail_errno(const char *operation) {
    fprintf(
        stderr,
        "af3-fasta-two-pass: %s: %s\n",
        operation,
        strerror(errno)
    );
    exit(EXIT_FAILURE);
}

static uint64_t parse_u64(const char *value, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (
        errno != 0
        || end == value
        || *end != '\0'
    ) {
        fprintf(stderr, "af3-fasta-two-pass: invalid %s: %s\n", name, value);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)parsed;
}

static double monotonic_seconds(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        fail_errno("clock_gettime");
    }
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static void write_all(int fd, const void *data, size_t size) {
    const unsigned char *cursor = data;
    size_t remaining = size;
    while (remaining > 0) {
        ssize_t written = write(fd, cursor, remaining);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written < 0) {
            fail_errno("write");
        }
        if (written == 0) {
            fail_message("write returned zero");
        }
        cursor += (size_t)written;
        remaining -= (size_t)written;
    }
}

static void pwrite_all(int fd, const void *data, size_t size, off_t offset) {
    const unsigned char *cursor = data;
    size_t remaining = size;
    off_t position = offset;
    while (remaining > 0) {
        ssize_t written = pwrite(fd, cursor, remaining, position);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written < 0) {
            fail_errno("pwrite");
        }
        if (written == 0) {
            fail_message("pwrite returned zero");
        }
        cursor += (size_t)written;
        remaining -= (size_t)written;
        position += written;
    }
}

static void flush_offsets(struct offset_writer *writer) {
    if (writer->count == 0) {
        return;
    }
    write_all(
        writer->fd,
        writer->values,
        writer->count * sizeof(*writer->values)
    );
    writer->total += (uint64_t)writer->count;
    writer->count = 0;
}

static void append_offset(struct offset_writer *writer, uint64_t value) {
    writer->values[writer->count++] = value;
    if (writer->count == OFFSET_BUFFER_RECORDS) {
        flush_offsets(writer);
    }
}

static struct index_header build_offset_index(
    int source_fd,
    int staged_fd,
    int index_fd,
    uint64_t expected_records
) {
    struct stat source_stat;
    if (fstat(source_fd, &source_stat) != 0) {
        fail_errno("fstat source");
    }
    if (!S_ISREG(source_stat.st_mode) || source_stat.st_size <= 0) {
        fail_message("source must be a nonempty regular file");
    }
    int allocation_error = posix_fallocate(
        staged_fd,
        0,
        source_stat.st_size
    );
    if (allocation_error != 0) {
        errno = allocation_error;
        fail_errno("preallocate staged source");
    }
    if (lseek(staged_fd, 0, SEEK_SET) < 0) {
        fail_errno("seek staged source");
    }

    struct index_header header = {0};
    write_all(index_fd, &header, sizeof(header));

    unsigned char *scan_buffer = malloc(SCAN_BUFFER_BYTES);
    uint64_t *offset_buffer = malloc(
        OFFSET_BUFFER_RECORDS * sizeof(*offset_buffer)
    );
    if (scan_buffer == NULL || offset_buffer == NULL) {
        fail_message("failed to allocate first-pass buffers");
    }
    struct offset_writer writer = {
        .fd = index_fd,
        .values = offset_buffer,
        .count = 0,
        .total = 0,
    };

    (void)posix_fadvise(source_fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    (void)posix_fadvise(staged_fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    uint64_t source_position = 0;
    uint64_t last_record_offset = 0;
    uint64_t maximum_record_size = 0;
    uint64_t record_count = 0;
    unsigned char previous_byte = 0;
    bool have_previous_byte = false;
    double last_progress = monotonic_seconds();

    for (;;) {
        ssize_t read_size = read(source_fd, scan_buffer, SCAN_BUFFER_BYTES);
        if (read_size < 0 && errno == EINTR) {
            continue;
        }
        if (read_size < 0) {
            fail_errno("read source during first pass");
        }
        if (read_size == 0) {
            break;
        }
        size_t chunk_size = (size_t)read_size;
        if (source_position == 0 && scan_buffer[0] != '>') {
            fail_message("source does not begin with a FASTA header");
        }
        write_all(staged_fd, scan_buffer, chunk_size);

        unsigned char *cursor = scan_buffer;
        unsigned char *chunk_end = scan_buffer + chunk_size;
        while (cursor < chunk_end) {
            unsigned char *candidate = memchr(
                cursor,
                '>',
                (size_t)(chunk_end - cursor)
            );
            if (candidate == NULL) {
                break;
            }
            size_t local_offset = (size_t)(candidate - scan_buffer);
            uint64_t absolute_offset = source_position + (uint64_t)local_offset;
            bool at_line_start = absolute_offset == 0;
            if (!at_line_start) {
                unsigned char before = local_offset > 0
                    ? candidate[-1]
                    : previous_byte;
                at_line_start = (
                    (local_offset > 0 || have_previous_byte)
                    && before == '\n'
                );
            }
            if (at_line_start) {
                if (record_count > 0) {
                    uint64_t record_size = absolute_offset - last_record_offset;
                    if (record_size == 0) {
                        fail_message("encountered an empty FASTA record span");
                    }
                    if (record_size > maximum_record_size) {
                        maximum_record_size = record_size;
                    }
                }
                append_offset(&writer, absolute_offset);
                last_record_offset = absolute_offset;
                record_count += 1;
            }
            cursor = candidate + 1;
        }

        previous_byte = scan_buffer[chunk_size - 1];
        have_previous_byte = true;
        source_position += (uint64_t)chunk_size;
        double now = monotonic_seconds();
        if (now - last_progress >= 60.0) {
            fprintf(
                stderr,
                "{\"event\":\"staging-progress\",\"records\":%" PRIu64
                ",\"bytes\":%" PRIu64 "}\n",
                record_count,
                source_position
            );
            fflush(stderr);
            last_progress = now;
        }
    }

    uint64_t source_size = (uint64_t)source_stat.st_size;
    if (source_position != source_size) {
        fail_message("first-pass byte count does not match source size");
    }
    if (record_count == 0) {
        fail_message("source contains no FASTA records");
    }
    uint64_t final_source_record_size = source_size - last_record_offset;
    if (final_source_record_size == 0) {
        fail_message("final FASTA record is empty");
    }
    bool append_final_newline = previous_byte != '\n';
    uint64_t final_output_record_size = (
        final_source_record_size + (append_final_newline ? 1U : 0U)
    );
    if (final_output_record_size > maximum_record_size) {
        maximum_record_size = final_output_record_size;
    }
    append_offset(&writer, source_size);
    flush_offsets(&writer);
    if (writer.total != record_count + 1) {
        fail_message("offset index count is inconsistent");
    }
    if (record_count != expected_records) {
        fprintf(
            stderr,
            "af3-fasta-two-pass: indexed records %" PRIu64
            " != expected %" PRIu64 "\n",
            record_count,
            expected_records
        );
        exit(EXIT_FAILURE);
    }

    header.magic = INDEX_MAGIC;
    header.version = INDEX_VERSION;
    header.source_size = source_size;
    header.output_size = source_size + (append_final_newline ? 1U : 0U);
    header.record_count = record_count;
    header.maximum_record_size = maximum_record_size;
    pwrite_all(index_fd, &header, sizeof(header), 0);
    if (fdatasync(index_fd) != 0) {
        fail_errno("fdatasync index");
    }
    fprintf(
        stderr,
        "{\"event\":\"staging-sync\",\"bytes\":%" PRIu64 "}\n",
        source_size
    );
    fflush(stderr);
    if (fdatasync(staged_fd) != 0) {
        fail_errno("fdatasync staged source");
    }
    (void)posix_fadvise(staged_fd, 0, 0, POSIX_FADV_DONTNEED);
    free(offset_buffer);
    free(scan_buffer);
    return header;
}

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t value = (*state += UINT64_C(0x9e3779b97f4a7c15));
    value = (
        (value ^ (value >> 30))
        * UINT64_C(0xbf58476d1ce4e5b9)
    );
    value = (
        (value ^ (value >> 27))
        * UINT64_C(0x94d049bb133111eb)
    );
    return value ^ (value >> 31);
}

static uint64_t bounded_random(uint64_t *state, uint64_t bound) {
    if (bound == 0) {
        fail_message("random bound must be positive");
    }
    uint64_t threshold = (uint64_t)(-bound) % bound;
    for (;;) {
        uint64_t value = splitmix64_next(state);
        if (value >= threshold) {
            return value % bound;
        }
    }
}

static uint32_t *build_permutation(uint64_t count, uint64_t seed) {
    if (count == 0 || count > UINT32_MAX) {
        fail_message("record count is outside the uint32 permutation domain");
    }
    if (count > SIZE_MAX / sizeof(uint32_t)) {
        fail_message("permutation size overflows size_t");
    }
    size_t allocation_size = (size_t)count * sizeof(uint32_t);
    uint32_t *permutation = mmap(
        NULL,
        allocation_size,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS,
        -1,
        0
    );
    if (permutation == MAP_FAILED) {
        fail_errno("mmap permutation");
    }
#ifdef MADV_HUGEPAGE
    (void)madvise(permutation, allocation_size, MADV_HUGEPAGE);
#endif
    for (uint64_t index = 0; index < count; ++index) {
        permutation[index] = (uint32_t)index;
    }
    uint64_t state = seed;
    for (uint64_t index = count - 1; index > 0; --index) {
        uint64_t swap_index = bounded_random(&state, index + 1);
        uint32_t temporary = permutation[index];
        permutation[index] = permutation[swap_index];
        permutation[swap_index] = temporary;
    }
    return permutation;
}

static void set_pool_error(struct read_pool *pool, int error_number) {
    int expected = 0;
    if (
        atomic_compare_exchange_strong(
            &pool->failed,
            &expected,
            1
        )
    ) {
        atomic_store(
            &pool->error_number,
            error_number == 0 ? EIO : error_number
        );
    }
}

static void *read_worker(void *argument) {
    struct read_pool *pool = argument;
    for (;;) {
        (void)pthread_barrier_wait(&pool->start_barrier);
        if (atomic_load(&pool->stop)) {
            (void)pthread_barrier_wait(&pool->finish_barrier);
            return NULL;
        }
        for (;;) {
            size_t task_index = atomic_fetch_add(&pool->next_task, 1);
            if (task_index >= pool->task_count) {
                break;
            }
            if (atomic_load(&pool->failed)) {
                continue;
            }
            const struct read_task *task = &pool->tasks[task_index];
            size_t remaining = task->length;
            size_t completed = 0;
            while (remaining > 0) {
                ssize_t read_size = pread(
                    pool->source_fd,
                    pool->buffer + task->destination_offset + completed,
                    remaining,
                    (off_t)(task->source_offset + completed)
                );
                if (read_size < 0 && errno == EINTR) {
                    continue;
                }
                if (read_size < 0) {
                    set_pool_error(pool, errno);
                    break;
                }
                if (read_size == 0) {
                    set_pool_error(pool, EIO);
                    break;
                }
                completed += (size_t)read_size;
                remaining -= (size_t)read_size;
            }
            if (!atomic_load(&pool->failed) && task->append_newline) {
                pool->buffer[
                    task->destination_offset + task->length
                ] = '\n';
            }
        }
        (void)pthread_barrier_wait(&pool->finish_barrier);
    }
}

static void initialize_pool(
    struct read_pool *pool,
    int source_fd,
    size_t thread_count,
    struct read_task *tasks,
    unsigned char *buffer
) {
    memset(pool, 0, sizeof(*pool));
    pool->source_fd = source_fd;
    pool->thread_count = thread_count;
    pool->tasks = tasks;
    pool->buffer = buffer;
    pool->threads = calloc(thread_count, sizeof(*pool->threads));
    if (pool->threads == NULL) {
        fail_message("failed to allocate worker handles");
    }
    if (
        pthread_barrier_init(
            &pool->start_barrier,
            NULL,
            (unsigned int)thread_count + 1
        ) != 0
        || pthread_barrier_init(
            &pool->finish_barrier,
            NULL,
            (unsigned int)thread_count + 1
        ) != 0
    ) {
        fail_message("failed to initialize worker barriers");
    }
    atomic_init(&pool->next_task, 0);
    atomic_init(&pool->failed, 0);
    atomic_init(&pool->error_number, 0);
    atomic_init(&pool->stop, false);
    for (size_t index = 0; index < thread_count; ++index) {
        if (
            pthread_create(
                &pool->threads[index],
                NULL,
                read_worker,
                pool
            ) != 0
        ) {
            fail_message("failed to create read worker");
        }
    }
}

static void dispatch_reads(struct read_pool *pool, size_t task_count) {
    pool->task_count = task_count;
    atomic_store(&pool->next_task, 0);
    atomic_store(&pool->failed, 0);
    atomic_store(&pool->error_number, 0);
    (void)pthread_barrier_wait(&pool->start_barrier);
    (void)pthread_barrier_wait(&pool->finish_barrier);
    if (atomic_load(&pool->failed)) {
        errno = atomic_load(&pool->error_number);
        fail_errno("pread source during second pass");
    }
}

static void destroy_pool(struct read_pool *pool) {
    atomic_store(&pool->stop, true);
    (void)pthread_barrier_wait(&pool->start_barrier);
    (void)pthread_barrier_wait(&pool->finish_barrier);
    for (size_t index = 0; index < pool->thread_count; ++index) {
        if (pthread_join(pool->threads[index], NULL) != 0) {
            fail_message("failed to join read worker");
        }
    }
    (void)pthread_barrier_destroy(&pool->start_barrier);
    (void)pthread_barrier_destroy(&pool->finish_barrier);
    free(pool->threads);
}

static uint64_t shuffle_to_output(
    int source_fd,
    int output_fd,
    const uint64_t *offsets,
    const uint32_t *permutation,
    const struct index_header *header,
    size_t thread_count,
    size_t prefetch_records,
    size_t prefetch_bytes,
    uint64_t *peak_batch_bytes
) {
    if (
        header->maximum_record_size > SIZE_MAX
        || prefetch_bytes > SIZE_MAX
    ) {
        fail_message("prefetch buffer is too large for this platform");
    }
    size_t buffer_size = prefetch_bytes;
    if ((size_t)header->maximum_record_size > buffer_size) {
        buffer_size = (size_t)header->maximum_record_size;
    }
    struct read_task *tasks = calloc(prefetch_records, sizeof(*tasks));
    unsigned char *buffer = malloc(buffer_size);
    if (tasks == NULL || buffer == NULL) {
        fail_message("failed to allocate second-pass buffers");
    }

    struct read_pool pool;
    initialize_pool(&pool, source_fd, thread_count, tasks, buffer);
    (void)posix_fadvise(source_fd, 0, 0, POSIX_FADV_RANDOM);
    uint64_t output_bytes = 0;
    uint64_t output_records = 0;
    double last_progress = monotonic_seconds();
    while (output_records < header->record_count) {
        size_t task_count = 0;
        size_t batch_bytes = 0;
        while (
            output_records + (uint64_t)task_count < header->record_count
            && task_count < prefetch_records
        ) {
            uint64_t position = output_records + (uint64_t)task_count;
            uint32_t ordinal = permutation[position];
            uint64_t start = offsets[ordinal];
            uint64_t finish = offsets[(uint64_t)ordinal + 1];
            if (finish <= start || finish - start > SIZE_MAX) {
                fail_message("invalid record span in offset index");
            }
            size_t source_record_size = (size_t)(finish - start);
            bool append_newline = (
                (uint64_t)ordinal + 1 == header->record_count
                && header->output_size > header->source_size
            );
            size_t output_record_size = (
                source_record_size + (append_newline ? 1U : 0U)
            );
            if (
                task_count > 0
                && output_record_size > prefetch_bytes - batch_bytes
            ) {
                break;
            }
            tasks[task_count] = (struct read_task){
                .source_offset = start,
                .length = source_record_size,
                .destination_offset = batch_bytes,
                .append_newline = append_newline,
            };
            batch_bytes += output_record_size;
            task_count += 1;
        }
        if (task_count == 0 || batch_bytes == 0 || batch_bytes > buffer_size) {
            fail_message("failed to construct a nonempty bounded read batch");
        }
        if ((uint64_t)batch_bytes > *peak_batch_bytes) {
            *peak_batch_bytes = (uint64_t)batch_bytes;
        }
        dispatch_reads(&pool, task_count);
        write_all(output_fd, buffer, batch_bytes);
        output_bytes += (uint64_t)batch_bytes;
        output_records += (uint64_t)task_count;

        double now = monotonic_seconds();
        if (now - last_progress >= 60.0) {
            fprintf(
                stderr,
                "{\"event\":\"progress\",\"records\":%" PRIu64
                ",\"bytes\":%" PRIu64 "}\n",
                output_records,
                output_bytes
            );
            fflush(stderr);
            last_progress = now;
        }
    }
    destroy_pool(&pool);
    free(buffer);
    free(tasks);
    return output_bytes;
}

static void usage(void) {
    fprintf(
        stderr,
        "usage: af3-fasta-two-pass --source PATH --staged-source PATH "
        "--output PATH --index PATH "
        "--expected-records N --seed N --threads N "
        "--prefetch-records N --prefetch-bytes N\n"
    );
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        printf("%s\n", PROGRAM_VERSION);
        return EXIT_SUCCESS;
    }
    const char *source_path = NULL;
    const char *staged_source_path = NULL;
    const char *output_path = NULL;
    const char *index_path = NULL;
    uint64_t expected_records = 0;
    uint64_t seed = 0;
    uint64_t thread_count_u64 = 0;
    uint64_t prefetch_records_u64 = 0;
    uint64_t prefetch_bytes_u64 = 0;

    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            usage();
            return EXIT_FAILURE;
        }
        const char *flag = argv[index];
        const char *value = argv[index + 1];
        if (strcmp(flag, "--source") == 0) {
            source_path = value;
        } else if (strcmp(flag, "--staged-source") == 0) {
            staged_source_path = value;
        } else if (strcmp(flag, "--output") == 0) {
            output_path = value;
        } else if (strcmp(flag, "--index") == 0) {
            index_path = value;
        } else if (strcmp(flag, "--expected-records") == 0) {
            expected_records = parse_u64(value, "expected-records");
        } else if (strcmp(flag, "--seed") == 0) {
            seed = parse_u64(value, "seed");
        } else if (strcmp(flag, "--threads") == 0) {
            thread_count_u64 = parse_u64(value, "threads");
        } else if (strcmp(flag, "--prefetch-records") == 0) {
            prefetch_records_u64 = parse_u64(value, "prefetch-records");
        } else if (strcmp(flag, "--prefetch-bytes") == 0) {
            prefetch_bytes_u64 = parse_u64(value, "prefetch-bytes");
        } else {
            usage();
            return EXIT_FAILURE;
        }
    }

    if (
        source_path == NULL
        || staged_source_path == NULL
        || output_path == NULL
        || index_path == NULL
        || expected_records == 0
        || thread_count_u64 == 0
        || thread_count_u64 > 256
        || prefetch_records_u64 == 0
        || prefetch_records_u64 > SIZE_MAX
        || prefetch_bytes_u64 == 0
        || prefetch_bytes_u64 > SIZE_MAX
    ) {
        usage();
        return EXIT_FAILURE;
    }
    size_t thread_count = (size_t)thread_count_u64;
    size_t prefetch_records = (size_t)prefetch_records_u64;
    size_t prefetch_bytes = (size_t)prefetch_bytes_u64;

    int source_fd = open(source_path, O_RDONLY | O_CLOEXEC);
    if (source_fd < 0) {
        fail_errno("open source");
    }
    int staged_fd = open(
        staged_source_path,
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
        0600
    );
    if (staged_fd < 0) {
        fail_errno("open staged source");
    }
    int index_fd = open(
        index_path,
        O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
        0600
    );
    if (index_fd < 0) {
        fail_errno("open index");
    }
    fprintf(
        stderr,
        "{\"event\":\"start\",\"version\":\"%s\","
        "\"expected_records\":%" PRIu64 ","
        "\"random_read_source\":\"container-local-staged-copy\"}\n",
        PROGRAM_VERSION,
        expected_records
    );
    fflush(stderr);

    double first_pass_started = monotonic_seconds();
    struct index_header header = build_offset_index(
        source_fd,
        staged_fd,
        index_fd,
        expected_records
    );
    double first_pass_seconds = monotonic_seconds() - first_pass_started;
    fprintf(
        stderr,
        "{\"event\":\"indexed\",\"records\":%" PRIu64
        ",\"bytes\":%" PRIu64 ",\"seconds\":%.6f}\n",
        header.record_count,
        header.source_size,
        first_pass_seconds
    );
    fflush(stderr);
    if (close(staged_fd) != 0) {
        fail_errno("close staged source writer");
    }
    if (close(source_fd) != 0) {
        fail_errno("close source after staging");
    }
    int shuffled_source_fd = open(
        staged_source_path,
        O_RDONLY | O_CLOEXEC
    );
    if (shuffled_source_fd < 0) {
        fail_errno("open staged source for second pass");
    }
    struct stat staged_stat;
    if (fstat(shuffled_source_fd, &staged_stat) != 0) {
        fail_errno("fstat staged source");
    }
    if (
        !S_ISREG(staged_stat.st_mode)
        || staged_stat.st_size < 0
        || (uint64_t)staged_stat.st_size != header.source_size
    ) {
        fail_message("staged source size is inconsistent");
    }

    struct stat index_stat;
    if (fstat(index_fd, &index_stat) != 0) {
        fail_errno("fstat index");
    }
    size_t expected_index_size = (
        sizeof(struct index_header)
        + (size_t)(header.record_count + 1) * sizeof(uint64_t)
    );
    if (
        index_stat.st_size < 0
        || (uint64_t)index_stat.st_size != (uint64_t)expected_index_size
    ) {
        fail_message("offset index size is inconsistent");
    }
    void *index_mapping = mmap(
        NULL,
        expected_index_size,
        PROT_READ,
        MAP_SHARED,
        index_fd,
        0
    );
    if (index_mapping == MAP_FAILED) {
        fail_errno("mmap index");
    }
    const struct index_header *mapped_header = index_mapping;
    if (
        mapped_header->magic != INDEX_MAGIC
        || mapped_header->version != INDEX_VERSION
        || mapped_header->source_size != header.source_size
        || mapped_header->output_size != header.output_size
        || mapped_header->record_count != header.record_count
    ) {
        fail_message("mapped index header is invalid");
    }
    const uint64_t *offsets = (const uint64_t *)(
        (const unsigned char *)index_mapping + sizeof(struct index_header)
    );

    double permutation_started = monotonic_seconds();
    uint32_t *permutation = build_permutation(header.record_count, seed);
    double permutation_seconds = monotonic_seconds() - permutation_started;
    fprintf(
        stderr,
        "{\"event\":\"permuted\",\"records\":%" PRIu64
        ",\"seconds\":%.6f}\n",
        header.record_count,
        permutation_seconds
    );
    fflush(stderr);

    int output_fd = open(
        output_path,
        O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
        0600
    );
    if (output_fd < 0) {
        fail_errno("open output");
    }
    if (posix_fallocate(output_fd, 0, (off_t)header.output_size) != 0) {
        fail_message("failed to preallocate output");
    }
    if (lseek(output_fd, 0, SEEK_SET) < 0) {
        fail_errno("seek output");
    }

    double second_pass_started = monotonic_seconds();
    uint64_t peak_batch_bytes = 0;
    uint64_t output_bytes = shuffle_to_output(
        shuffled_source_fd,
        output_fd,
        offsets,
        permutation,
        &header,
        thread_count,
        prefetch_records,
        prefetch_bytes,
        &peak_batch_bytes
    );
    double second_pass_seconds = monotonic_seconds() - second_pass_started;
    if (output_bytes != header.output_size) {
        fail_message("output byte count does not match normalized output size");
    }
    if (close(output_fd) != 0) {
        fail_errno("close output");
    }
    if (
        munmap(
            permutation,
            (size_t)header.record_count * sizeof(uint32_t)
        ) != 0
    ) {
        fail_errno("munmap permutation");
    }
    if (munmap(index_mapping, expected_index_size) != 0) {
        fail_errno("munmap index");
    }
    if (close(index_fd) != 0) {
        fail_errno("close index");
    }
    if (close(shuffled_source_fd) != 0) {
        fail_errno("close staged source");
    }

    fprintf(
        stderr,
        "{\"event\":\"complete\",\"records\":%" PRIu64
        ",\"bytes\":%" PRIu64 ",\"seconds\":%.6f}\n",
        header.record_count,
        output_bytes,
        second_pass_seconds
    );
    fflush(stderr);
    printf(
        "{\"schema_version\":1,\"version\":\"%s\","
        "\"source_size_bytes\":%" PRIu64 ","
        "\"staged_source_size_bytes\":%" PRIu64 ","
        "\"output_size_bytes\":%" PRIu64 ","
        "\"record_count\":%" PRIu64 ","
        "\"offset_index_size_bytes\":%zu,"
        "\"permutation_size_bytes\":%" PRIu64 ","
        "\"seed\":%" PRIu64 ","
        "\"threads\":%zu,"
        "\"prefetch_records\":%zu,"
        "\"prefetch_bytes\":%zu,"
        "\"peak_batch_bytes\":%" PRIu64 ","
        "\"random_read_source\":\"container-local-staged-copy\","
        "\"first_pass_seconds\":%.6f,"
        "\"permutation_seconds\":%.6f,"
        "\"second_pass_seconds\":%.6f}\n",
        PROGRAM_VERSION,
        header.source_size,
        header.source_size,
        header.output_size,
        header.record_count,
        expected_index_size,
        header.record_count * sizeof(uint32_t),
        seed,
        thread_count,
        prefetch_records,
        prefetch_bytes,
        peak_batch_bytes,
        first_pass_seconds,
        permutation_seconds,
        second_pass_seconds
    );
    return EXIT_SUCCESS;
}
