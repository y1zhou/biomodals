"""Develop and benchmark deterministic AlphaFold 3 database sharding.

Upstream sources:

- <https://github.com/google-deepmind/alphafold3>
- <https://github.com/google-deepmind/alphafold3/blob/main/docs/performance.md>

This is an isolated, temporary development app. It does not mount AlphaFold
model weights or the production MSA cache, and it never imports
``alphafold3_app``. Commands are plan-only unless ``--submit`` is explicitly
supplied.

The first operation copies small BFD from the read-only production database
Volume, creates 64 deterministic SeqKit shards in
``AlphaFold3-msa-db-sharded``, validates them, and publishes ``manifest.json``
last. Duplicate full headers omitted by SeqKit's two-pass FASTA index are
recovered from its logged byte offsets before splitting. Benchmark evidence is
written to ``AlphaFold3-MSA-Benchmark-outputs``.

The production-candidate operations build seven fixed immutable profiles
without copying their monolithic sources into the sharded Volume. They stage
an ephemeral source copy, the shuffled FASTA, and a compact occurrence-offset
index under ``/tmp``. A pinned native two-pass helper preserves duplicate
headers, tees its sequential indexing pass into the local copy, then uses
bounded concurrent reads from local SSD. Only shards, validation evidence, and
a manifest-last publication remain on the sharded Volume.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import median
from threading import Event, Lock, Thread
from time import perf_counter, time
from typing import Any, BinaryIO, cast

import modal
import orjson

from biomodals.app.config import AppConfig
from biomodals.helper import patch_image_for_helper

CAMPAIGN_ID = "small-bfd-phase1-v2"
PROFILE_ID = "small-bfd-64-v1"
PROFILE_SCHEMA_VERSION = 1
SOURCE_DB_FILENAME = "bfd-first_non_consensus_sequences.fasta"
SOURCE_DB_VOLUME_NAME = "AlphaFold3-msa-db"
SHARDED_DB_VOLUME_NAME = "AlphaFold3-msa-db-sharded"
OUTPUT_VOLUME_NAME = "AlphaFold3-MSA-Benchmark-outputs"
DATABASE_ID = "small-bfd"
SHARD_COUNT = 64
SHARD_RANDOM_SEED = 23
SMALL_BFD_Z = 65_984_053
EXPECTED_RECOVERED_RECORDS = 55_187
EXPECTED_RECOVERED_RESIDUES = 24_934_582
SEQKIT_VERSION = "2.13.0"
DEFAULT_SEQKIT_THREADS = 8
MAX_SEQKIT_THREADS = 32
MAX_PROFILE_IMBALANCE = 0.05
PROFILE_RECIPE_VERSION = 2
RECOVERED_HEADER_NAMESPACE = "__AF3_RECOVERED_"
MAX_FASTA_HEADER_BYTES = 1024 * 1024
PROFILE_VALIDATION_RELPATHS = (
    "validation/source-stats.tsv",
    "validation/shard-stats.tsv",
    "validation/shard-summary.parquet",
    "validation/source-sum.tsv",
    "validation/shard-sum.tsv",
    "validation/seqkit-sum.json",
    "validation/shuffle-stderr.log",
    "validation/duplicate-recovery.jsonl",
)
HMMER_VERSION = "3.4"
JACKHMMER_PATCH_SHA256 = (
    "df9e3ae35ad1659921d96ebfca67a9616a7a467ddde2be18a56f9bd3edb38c41"
)
JACKHMMER_BINARY_PATH = "/hmmer/bin/jackhmmer"
JACKHMMER_N_ITER = 1
JACKHMMER_E_VALUE = 1e-4
JACKHMMER_MAX_SEQUENCES = 5_000
JACKHMMER_FILTER_F1 = 5e-4
JACKHMMER_FILTER_F2 = 5e-5
JACKHMMER_FILTER_F3 = 5e-7
SCIENTIFIC_COMPARISON_POLICY = "top-target-order-exact-modulo-evalue-bit-score-ties-v2"
RESOURCE_TRACE_INTERVAL_SECONDS = 1.0
MODAL_CPU_USD_PER_CORE_SECOND = 0.0000131
MODAL_MEMORY_USD_PER_GIB_SECOND = 0.00000222
MODAL_PRICING_OBSERVED_DATE = "2026-07-22"
MODAL_PRICING_URL = "https://modal.com/pricing"
JSON_OPTIONS = orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE
JSONL_OPTIONS = orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE
_FAI_DUPLICATE_WARNING = re.compile(
    rb'^\[fai warning\] ignoring duplicate sequence "(?P<name>.*)" '
    rb"at byte offset (?P<offset>[0-9]+)\r?\n?$"
)

PRODUCTION_PROFILE_SCHEMA_VERSION = 2
LEGACY_PRODUCTION_PROFILE_RECIPE_VERSION = 3
ORDINAL_SHUFFLER_RECIPE_VERSION = 4
COMPOSABLE_MULTISET_RECIPE_VERSION = 5
PRODUCTION_PROFILE_ROOT = "profiles"
PRODUCTION_PREPARATION_ROOT = "production-candidates/profile-builds"
RECORD_MULTISET_BENCHMARK_ROOT = "production-candidates/record-multiset-benchmarks"
PRODUCTION_PROFILE_CLAIM_DICT_NAME = "AlphaFold3-msa-profile-build-claims"
PRODUCTION_BUILD_TIMEOUT_SECONDS = 86_400
PRODUCTION_BUILD_MEMORY_MIB = (1024, 262_144)
PRODUCTION_PROFILE_STALE_SECONDS = PRODUCTION_BUILD_TIMEOUT_SECONDS + 900
PRODUCTION_SCRATCH_ROOT = Path(tempfile.gettempdir())
PRODUCTION_SCRATCH_HEADROOM_BYTES = 1024 * 1024 * 1024
ORDINAL_SHUFFLER_VERSION = "af3-fasta-two-pass-v2"
ORDINAL_SHUFFLER_PREFETCH_RECORDS = 65_536
ORDINAL_SHUFFLER_PREFETCH_BYTES = 256 * 1024 * 1024
RECORD_MULTISET_VERSION = "af3-fasta-record-multiset-v1"
RECORD_MULTISET_CANONICALIZATION = (
    "full-header-and-sequence-case-sensitive-line-ending-independent-v1"
)
RECORD_MULTISET_AGGREGATE = "sha256-lane-sum-xor-and-square-sum-with-counts-v1"
UNIPROT_V4_VALIDATION_BASELINE = {
    "source_seqkit_sum_seconds": 338.390180,
    "post_shard_stats_to_completion_seconds": 2242.917015,
    "complete_builder_seconds": 4695.084267,
}
SOURCE_POLICIES = ("keep", "compress", "delete")
LEGACY_PRODUCTION_VALIDATION_RELPATHS = (
    "validation/source-stats.tsv",
    "validation/shard-stats.tsv",
    "validation/shard-summary.parquet",
    "validation/source-sum.tsv",
    "validation/shard-sum.tsv",
    "validation/seqkit-sum.json",
    "validation/shuffle-stderr.log",
    "validation/duplicate-recovery.jsonl",
)
ORDINAL_PRODUCTION_VALIDATION_RELPATHS = (
    *LEGACY_PRODUCTION_VALIDATION_RELPATHS,
    "validation/shuffler-metrics.json",
)
PRODUCTION_VALIDATION_RELPATHS = (
    "validation/source-stats.tsv",
    "validation/shard-stats.tsv",
    "validation/shard-summary.parquet",
    "validation/record-multiset.json",
    "validation/shuffle-stderr.log",
    "validation/duplicate-recovery.jsonl",
    "validation/shuffler-metrics.json",
)
NHMMER_BINARY_PATH = "/hmmer/bin/nhmmer"
HMMALIGN_BINARY_PATH = "/hmmer/bin/hmmalign"
HMMBUILD_BINARY_PATH = "/hmmer/bin/hmmbuild"
PRODUCTION_SEARCH_N_CPU = 2
PRODUCTION_SEARCH_MAX_PARALLEL_SHARDS = 16
ORACLE_MONOLITH_N_CPU = 8


_ORDINAL_SHUFFLER_SOURCE = r"""
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
"""
ORDINAL_SHUFFLER_SOURCE_SHA256 = hashlib.sha256(
    _ORDINAL_SHUFFLER_SOURCE.encode("utf-8")
).hexdigest()

_RECORD_MULTISET_SOURCE = r"""
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
"""
RECORD_MULTISET_SOURCE_SHA256 = hashlib.sha256(
    _RECORD_MULTISET_SOURCE.encode("utf-8")
).hexdigest()


CONF = AppConfig(
    tags={"group": Path(__file__).parent.name},
    name="AlphaFold3-MSA-Benchmark",
    repo_url="https://github.com/y1zhou/alphafold3",
    repo_commit_hash="987ad1cb7d7028b6d35908cf63fe7d951d98d6b6",
    package_name="alphafold3",
    version="3.0.2",
    python_version="3.12",
    cuda_version="cu130",
    timeout=21_600,
)


@dataclass(frozen=True)
class AppInfo:
    """Fixed Volume mount points and profile-relative paths."""

    source_db_dir: str = f"/mnt/{SOURCE_DB_VOLUME_NAME}"
    sharded_db_dir: str = f"/mnt/{SHARDED_DB_VOLUME_NAME}"
    output_dir: str = f"/mnt/{OUTPUT_VOLUME_NAME}"
    profile_relpath: str = f"profiles/{PROFILE_ID}"
    preparation_relpath: str = f"benchmarks/{CAMPAIGN_ID}/preparation"


@dataclass(frozen=True)
class FaiDuplicateWarning:
    """One record omitted by SeqKit's full-header FASTA index."""

    sequence_name: bytes
    sequence_offset: int


@dataclass(frozen=True)
class DatabaseProfileSpec:
    """One code-owned immutable database-sharding specification."""

    database_id: str
    profile_id: str
    source_filename: str
    shard_count: int
    polymer: str
    expected_num_seqs: int | None
    expected_sum_len: int | None
    max_sequences: int

    @property
    def search_space_value(self) -> int | float:
        """Return the full-database HMMER search-space value."""
        if self.polymer == "protein":
            if self.expected_num_seqs is None:
                raise RuntimeError(
                    f"{self.database_id} lacks an expected sequence count"
                )
            return self.expected_num_seqs
        if self.polymer == "rna":
            if self.expected_sum_len is None:
                raise RuntimeError(
                    f"{self.database_id} lacks an expected residue count"
                )
            return self.expected_sum_len / 1_000_000
        raise RuntimeError(f"Unsupported polymer type: {self.polymer}")

    @property
    def search_space_unit(self) -> str:
        """Return the unit expected by the pinned HMMER wrapper."""
        if self.polymer == "protein":
            return "sequences"
        if self.polymer == "rna":
            return "megabases"
        raise RuntimeError(f"Unsupported polymer type: {self.polymer}")


DATABASE_PROFILE_SPECS = (
    DatabaseProfileSpec(
        database_id="small_bfd",
        profile_id="small-bfd-64-v2",
        source_filename="bfd-first_non_consensus_sequences.fasta",
        shard_count=64,
        polymer="protein",
        expected_num_seqs=65_984_053,
        expected_sum_len=None,
        max_sequences=5_000,
    ),
    DatabaseProfileSpec(
        database_id="mgnify",
        profile_id="mgnify-512-v1",
        source_filename="mgy_clusters_2022_05.fa",
        shard_count=512,
        polymer="protein",
        expected_num_seqs=623_796_864,
        expected_sum_len=None,
        max_sequences=5_000,
    ),
    DatabaseProfileSpec(
        database_id="uniprot",
        profile_id="uniprot-256-v1",
        source_filename="uniprot_all_2021_04.fa",
        shard_count=256,
        polymer="protein",
        expected_num_seqs=225_619_586,
        expected_sum_len=None,
        max_sequences=50_000,
    ),
    DatabaseProfileSpec(
        database_id="uniref90",
        profile_id="uniref90-128-v1",
        source_filename="uniref90_2022_05.fa",
        shard_count=128,
        polymer="protein",
        expected_num_seqs=153_742_194,
        expected_sum_len=None,
        max_sequences=10_000,
    ),
    DatabaseProfileSpec(
        database_id="ntrna",
        profile_id="nt-rna-256-v1",
        source_filename="nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta",
        shard_count=256,
        polymer="rna",
        expected_num_seqs=None,
        expected_sum_len=76_752_808_514,
        max_sequences=10_000,
    ),
    DatabaseProfileSpec(
        database_id="rfam",
        profile_id="rfam-16-v1",
        source_filename="rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta",
        shard_count=16,
        polymer="rna",
        expected_num_seqs=None,
        expected_sum_len=138_115_553,
        max_sequences=10_000,
    ),
    DatabaseProfileSpec(
        database_id="rnacentral",
        profile_id="rnacentral-64-v1",
        source_filename="rnacentral_active_seq_id_90_cov_80_linclust.fasta",
        shard_count=64,
        polymer="rna",
        expected_num_seqs=None,
        expected_sum_len=13_271_415_730,
        max_sequences=10_000,
    ),
)


APP_INFO = AppInfo()

SOURCE_MSA_DB_VOLUME = modal.Volume.from_name(
    SOURCE_DB_VOLUME_NAME,
    version=2,
)
SHARDED_MSA_DB_VOLUME = modal.Volume.from_name(
    SHARDED_DB_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
BENCHMARK_OUTPUT_VOLUME = modal.Volume.from_name(
    OUTPUT_VOLUME_NAME,
    create_if_missing=True,
    version=2,
)
PROFILE_BUILD_CLAIMS = modal.Dict.from_name(
    PRODUCTION_PROFILE_CLAIM_DICT_NAME,
    create_if_missing=True,
)


# Keep the benchmark on the exact AlphaFold/HMMER environment used by the
# production app. SeqKit is an additional preparation-only tool.
runtime_image = (
    modal.Image
    .micromamba(python_version=CONF.python_version)
    .apt_install(
        "git",
        "build-essential",
        "zstd",
        "zlib1g-dev",
        "wget",
    )
    .env(
        CONF.default_env
        | {
            "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
            "XLA_CLIENT_MEM_FRACTION": "0.95",
        }
    )
    .micromamba_install(
        f"seqkit={SEQKIT_VERSION}",
        channels=["conda-forge", "bioconda"],
    )
    .run_commands("seqkit version")
    .run_commands(
        " && ".join((
            f"git clone {CONF.repo_url} {CONF.git_clone_dir}",
            f"cd {CONF.git_clone_dir}",
            f"git checkout {CONF.repo_commit_hash}",
            "mkdir /hmmer_build",
            "wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz "
            "--directory-prefix /hmmer_build",
            "cd /hmmer_build",
            "echo 'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3 "
            "hmmer-3.4.tar.gz' | sha256sum --check",
            "tar zxf hmmer-3.4.tar.gz",
            "rm hmmer-3.4.tar.gz",
            "cd /hmmer_build",
            f"patch -p0 < {CONF.git_clone_dir}/docker/jackhmmer_seq_limit.patch",
            "cd /hmmer_build/hmmer-3.4",
            "./configure --prefix=/hmmer",
            "make -j",
            "make install",
            "cd /hmmer_build/hmmer-3.4/easel",
            "make install",
            "rm -rf /hmmer_build",
        ))
    )
    .workdir(str(CONF.git_clone_dir))
    .uv_pip_install(str(CONF.git_clone_dir))
    .run_commands("build_data")
    .env({"PATH": "/hmmer/bin:$PATH"})
    .pipe(patch_image_for_helper)
)

app = modal.App(CONF.name, image=runtime_image, tags=CONF.tags)

_CONTAINER_INSTANCE_ID = uuid.uuid4().hex
_CONTAINER_SAMPLE_COUNT = 0
_CONTAINER_SAMPLE_LOCK = Lock()


def _utc_now() -> str:
    """Return an RFC 3339-compatible UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _validate_seqkit_threads(seqkit_threads: int) -> int:
    """Validate the SeqKit concurrency argument."""
    if isinstance(seqkit_threads, bool) or not isinstance(seqkit_threads, int):
        raise TypeError("seqkit_threads must be an integer")
    if not 1 <= seqkit_threads <= MAX_SEQKIT_THREADS:
        raise ValueError(
            f"seqkit_threads must be between 1 and {MAX_SEQKIT_THREADS}, "
            f"got {seqkit_threads}"
        )
    return seqkit_threads


def _shard_filename(index: int) -> str:
    """Return the AlphaFold shard filename for a zero-based index."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("shard index must be an integer")
    if not 0 <= index < SHARD_COUNT:
        raise ValueError(f"shard index must be in [0, {SHARD_COUNT}), got {index}")
    return f"{SOURCE_DB_FILENAME}-{index:05d}-of-{SHARD_COUNT:05d}"


def _shard_names() -> tuple[str, ...]:
    """Return every expected shard filename in AlphaFold order."""
    return tuple(_shard_filename(index) for index in range(SHARD_COUNT))


def _duplicate_recovery_recipe() -> dict[str, object]:
    """Return the scientific recipe for restoring FAI-omitted records."""
    return {
        "warning_source": "seqkit-fai-sequence-byte-offset",
        "expected_records": EXPECTED_RECOVERED_RECORDS,
        "temporary_header_identity": "unique-uuid",
        "append_after_shuffle": True,
        "strip_after_split": True,
    }


def _json_bytes(value: object) -> bytes:
    """Serialize a JSON value deterministically."""
    return orjson.dumps(value, option=JSON_OPTIONS)


def _write_json_atomic(path: Path, value: object) -> None:
    """Atomically publish one small JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    """Atomically publish one immutable byte payload."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object, rejecting all other top-level values."""
    value = orjson.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _require_regular_file(path: Path) -> None:
    """Require a non-symlink regular file with at least one byte."""
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"Expected regular file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"Expected nonempty file: {path}")


def _sha256_file(
    path: Path,
    *,
    chunk_size: int = 16 * 1024 * 1024,
    forbidden_bytes: bytes | None = None,
) -> str:
    """Compute a digest and optionally reject a byte marker while streaming."""
    _require_regular_file(path)
    if forbidden_bytes == b"":
        raise ValueError("forbidden_bytes must be nonempty")
    digest = hashlib.sha256()
    overlap = b""
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            if forbidden_bytes is not None:
                searchable = overlap + chunk
                if forbidden_bytes in searchable:
                    raise ValueError(f"Forbidden byte marker remains in {path}")
                overlap_size = len(forbidden_bytes) - 1
                overlap = searchable[-overlap_size:] if overlap_size else b""
    return digest.hexdigest()


def _copy_file_with_sha256(source: Path, destination: Path) -> tuple[str, int]:
    """Stream one file between Volume mounts while hashing its source bytes."""
    _require_regular_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as source_handle, destination.open("xb") as dest_handle:
        while chunk := source_handle.read(16 * 1024 * 1024):
            dest_handle.write(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        dest_handle.flush()
        os.fsync(dest_handle.fileno())
    return digest.hexdigest(), byte_count


def _append_log(path: Path, message: str) -> None:
    """Append one timestamped line to a durable operation log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_utc_now()} {message}\n")


def _append_diagnostic_file(source_path: Path, log_path: Path) -> None:
    """Copy one command's raw diagnostics into the durable operation log."""
    _require_regular_file(source_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("rb") as source, log_path.open("ab") as log:
        shutil.copyfileobj(source, log, length=1024 * 1024)


def _parse_fai_duplicate_warnings(
    diagnostics_path: Path,
) -> tuple[FaiDuplicateWarning, ...]:
    """Parse SeqKit FAI duplicate warnings as ordered sequence offsets."""
    _require_regular_file(diagnostics_path)
    warnings: list[FaiDuplicateWarning] = []
    with diagnostics_path.open("rb") as diagnostics:
        for line_number, line in enumerate(diagnostics, start=1):
            if b"[fai warning]" not in line:
                continue
            match = _FAI_DUPLICATE_WARNING.fullmatch(line)
            if match is None:
                raise ValueError(
                    f"Malformed SeqKit FAI warning at {diagnostics_path}:{line_number}"
                )
            sequence_name = match.group("name")
            sequence_offset = int(match.group("offset"))
            if not sequence_name:
                raise ValueError(
                    f"Empty sequence name in FAI warning at line {line_number}"
                )
            if sequence_offset <= 0:
                raise ValueError(
                    f"Invalid FAI sequence offset at line {line_number}: "
                    f"{sequence_offset}"
                )
            if warnings and sequence_offset <= warnings[-1].sequence_offset:
                raise ValueError(
                    "SeqKit FAI warning offsets must be strictly increasing"
                )
            warnings.append(FaiDuplicateWarning(sequence_name, sequence_offset))
    return tuple(warnings)


def _read_fasta_header_before_sequence_offset(
    source: BinaryIO,
    sequence_offset: int,
) -> bytes:
    """Read the exact FASTA header preceding an FAI sequence-start offset."""
    file_size = os.fstat(source.fileno()).st_size
    if not 1 < sequence_offset <= file_size:
        raise ValueError(
            f"FAI sequence offset {sequence_offset} is outside the source file"
        )
    source.seek(sequence_offset - 1)
    if source.read(1) != b"\n":
        raise ValueError(
            f"FAI sequence offset {sequence_offset} does not follow a header line"
        )

    cursor = sequence_offset - 1
    header_chunks: list[bytes] = []
    scanned_bytes = 0
    header_line: bytes | None = None
    while cursor > 0 and scanned_bytes < MAX_FASTA_HEADER_BYTES:
        read_size = min(4096, cursor, MAX_FASTA_HEADER_BYTES - scanned_bytes)
        chunk_start = cursor - read_size
        source.seek(chunk_start)
        chunk = source.read(read_size)
        previous_newline = chunk.rfind(b"\n")
        if previous_newline >= 0:
            header_chunks.append(chunk[previous_newline + 1 :])
            header_line = b"".join(reversed(header_chunks))
            break
        header_chunks.append(chunk)
        scanned_bytes += read_size
        cursor = chunk_start
    if header_line is None and cursor == 0:
        header_line = b"".join(reversed(header_chunks))
    if header_line is None:
        raise ValueError(
            f"FASTA header before offset {sequence_offset} exceeds "
            f"{MAX_FASTA_HEADER_BYTES} bytes"
        )
    if header_line.endswith(b"\r"):
        header_line = header_line[:-1]
    if not header_line.startswith(b">") or len(header_line) == 1:
        raise ValueError(
            f"FAI sequence offset {sequence_offset} has no valid preceding header"
        )
    return header_line[1:]


def _recovery_header_pattern(temporary_namespace: str) -> str:
    """Return the anchored SeqKit regex for one generation's UUID prefixes."""
    expected = rf"{RECOVERED_HEADER_NAMESPACE}[0-9a-f]{{32}}_"
    if re.fullmatch(expected, temporary_namespace) is None:
        raise ValueError("Invalid temporary recovery namespace")
    return rf"^{temporary_namespace}[0-9a-f]{{32}}__"


def _append_recovered_fasta_records(
    source_path: Path,
    shuffled_path: Path,
    warnings: tuple[FaiDuplicateWarning, ...],
    report_path: Path,
    *,
    temporary_namespace: str,
) -> dict[str, int]:
    """Recover FAI-omitted records and append UUID-prefixed FASTA entries."""
    _require_regular_file(source_path)
    _require_regular_file(shuffled_path)
    _recovery_header_pattern(temporary_namespace)
    if not warnings:
        raise ValueError("SeqKit emitted no duplicate-record byte offsets")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    recovered_residues = 0
    temporary_uuids: set[str] = set()
    previous_offset = 0
    namespace_bytes = temporary_namespace.encode("ascii")
    with (
        source_path.open("rb") as source,
        shuffled_path.open("r+b") as shuffled,
        report_path.open("xb") as report,
    ):
        shuffled.seek(0, os.SEEK_END)
        shuffled_size = shuffled.tell()
        if shuffled_size > 0:
            shuffled.seek(-1, os.SEEK_END)
            if shuffled.read(1) != b"\n":
                shuffled.write(b"\n")
        shuffled.seek(0, os.SEEK_END)

        for warning in warnings:
            if warning.sequence_offset <= previous_offset:
                raise ValueError(
                    "SeqKit FAI warning offsets must be strictly increasing"
                )
            previous_offset = warning.sequence_offset
            original_header = _read_fasta_header_before_sequence_offset(
                source,
                warning.sequence_offset,
            )
            normalized_header = re.sub(rb"\t+", b" ", original_header)
            if normalized_header != warning.sequence_name:
                raise ValueError(
                    "FAI warning name does not match source header at byte offset "
                    f"{warning.sequence_offset}"
                )

            record_uuid = uuid.uuid4().hex
            if not re.fullmatch(r"[0-9a-f]{32}", record_uuid):
                raise RuntimeError(
                    "UUID generator returned an invalid hexadecimal UUID"
                )
            if record_uuid in temporary_uuids:
                raise RuntimeError("UUID generator returned a duplicate recovery UUID")
            temporary_uuids.add(record_uuid)
            temporary_prefix = namespace_bytes + record_uuid.encode("ascii") + b"__"
            shuffled.write(b">" + temporary_prefix + original_header + b"\n")

            sequence_digest = hashlib.sha256()
            sequence_length = 0
            sequence_ends_with_newline = True
            source.seek(warning.sequence_offset)
            while line := source.readline():
                if line.startswith(b">"):
                    break
                shuffled.write(line)
                sequence_bases = line.rstrip(b"\r\n")
                sequence_digest.update(sequence_bases)
                sequence_length += len(sequence_bases)
                sequence_ends_with_newline = line.endswith(b"\n")
            if not sequence_ends_with_newline:
                shuffled.write(b"\n")
            recovered_residues += sequence_length
            report.write(
                orjson.dumps(
                    {
                        "byte_offset": warning.sequence_offset,
                        "header_sha256": hashlib.sha256(original_header).hexdigest(),
                        "sequence_length": sequence_length,
                        "sequence_sha256": sequence_digest.hexdigest(),
                        "temporary_uuid": record_uuid,
                    },
                    option=JSONL_OPTIONS,
                )
            )

        shuffled.flush()
        os.fsync(shuffled.fileno())
        report.flush()
        os.fsync(report.fileno())

    return {
        "recovered_records": len(warnings),
        "recovered_residues": recovered_residues,
        "first_byte_offset": warnings[0].sequence_offset,
        "last_byte_offset": warnings[-1].sequence_offset,
    }


def _require_executable(name: str) -> str:
    """Resolve a fixed executable name to an absolute path."""
    executable = shutil.which(name)
    if executable is None:
        raise FileNotFoundError(f"Required executable is not installed: {name}")
    return str(Path(executable).resolve())


def _compile_ordinal_shuffler(
    scratch_root: Path,
    log_path: Path,
) -> Path:
    """Compile the pinned bounded-memory FASTA shuffler in local scratch."""
    compiler = _require_executable("cc")
    source_path = scratch_root / "af3-fasta-two-pass.c"
    executable_path = scratch_root / ORDINAL_SHUFFLER_VERSION
    source_path.write_text(_ORDINAL_SHUFFLER_SOURCE, encoding="utf-8")
    if _sha256_file(source_path) != ORDINAL_SHUFFLER_SOURCE_SHA256:
        raise RuntimeError("Native shuffler source digest changed while writing")
    compile_argv = [
        compiler,
        "-std=c11",
        "-O3",
        "-pthread",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(source_path),
        "-o",
        str(executable_path),
    ]
    _append_log(log_path, f"Running command: {shlex.join(compile_argv)}")
    with log_path.open("ab") as log:
        completed = subprocess.run(  # noqa: S603
            compile_argv,
            check=False,
            stdout=log,
            stderr=log,
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, compile_argv)
    _require_regular_file(executable_path)
    if not os.access(executable_path, os.X_OK):
        raise PermissionError(f"Native shuffler is not executable: {executable_path}")
    version = subprocess.run(  # noqa: S603
        [str(executable_path), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != ORDINAL_SHUFFLER_VERSION:
        raise RuntimeError(
            f"Expected native shuffler {ORDINAL_SHUFFLER_VERSION}, got {version!r}"
        )
    return executable_path


def _record_multiset_identity() -> dict[str, str]:
    """Return the fixed canonical-record multiset algorithm identity."""
    return {
        "version": RECORD_MULTISET_VERSION,
        "source_code_sha256": RECORD_MULTISET_SOURCE_SHA256,
        "canonicalization": RECORD_MULTISET_CANONICALIZATION,
        "digest": "sha256",
        "aggregate": RECORD_MULTISET_AGGREGATE,
    }


def _compile_record_multiset_helper(
    scratch_root: Path,
    log_path: Path,
) -> Path:
    """Compile the pinned composable FASTA-record validator in local scratch."""
    compiler = _require_executable("cc")
    source_path = scratch_root / "af3-fasta-record-multiset.c"
    executable_path = scratch_root / RECORD_MULTISET_VERSION
    source_path.write_text(_RECORD_MULTISET_SOURCE, encoding="utf-8")
    if _sha256_file(source_path) != RECORD_MULTISET_SOURCE_SHA256:
        raise RuntimeError("Record-multiset helper source digest changed while writing")
    compile_argv = [
        compiler,
        "-std=c11",
        "-O3",
        "-pthread",
        "-Wall",
        "-Wextra",
        "-Werror",
        str(source_path),
        "-o",
        str(executable_path),
    ]
    _append_log(log_path, f"Running command: {shlex.join(compile_argv)}")
    with log_path.open("ab") as log:
        completed = subprocess.run(  # noqa: S603
            compile_argv,
            check=False,
            stdout=log,
            stderr=log,
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, compile_argv)
    _require_regular_file(executable_path)
    if not os.access(executable_path, os.X_OK):
        raise PermissionError(
            f"Record-multiset helper is not executable: {executable_path}"
        )
    version = subprocess.run(  # noqa: S603
        [str(executable_path), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != RECORD_MULTISET_VERSION:
        raise RuntimeError(
            f"Expected validator {RECORD_MULTISET_VERSION}, got {version!r}"
        )
    return executable_path


def _record_multiset_signature(report: dict[str, Any]) -> dict[str, object]:
    """Validate one helper report and return its composable signature."""
    expected_strings = _record_multiset_identity()
    for field in ("version", "canonicalization", "digest", "aggregate"):
        if report.get(field) != expected_strings[field]:
            raise ValueError(f"Unexpected record-multiset {field}")
    for field in ("files", "threads", "records", "header_bytes", "sequence_bytes"):
        value = report.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Record-multiset {field} must be an integer")
        if value < (1 if field in {"files", "threads", "records"} else 0):
            raise ValueError(f"Record-multiset {field} is outside its range")
    if report["threads"] > report["files"]:
        raise ValueError("Record-multiset threads exceed input files")
    signature: dict[str, object] = {
        "records": report["records"],
        "header_bytes": report["header_bytes"],
        "sequence_bytes": report["sequence_bytes"],
    }
    for field in (
        "sum_sha256_lanes",
        "xor_sha256_lanes",
        "sum_square_sha256_lanes",
    ):
        values = report.get(field)
        if (
            not isinstance(values, list)
            or len(values) != 4
            or any(
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{16}", value) is None
                for value in values
            )
        ):
            raise ValueError(f"Invalid record-multiset {field}")
        signature[field] = values
    return signature


def _run_record_multiset_helper(
    executable: Path,
    input_paths: tuple[Path, ...],
    output_path: Path,
    log_path: Path,
    *,
    threads: int,
) -> dict[str, object]:
    """Scan one file set and return a validated composable multiset report."""
    if not input_paths:
        raise ValueError("Record-multiset validation requires at least one input")
    selected_threads = min(_validate_seqkit_threads(threads), len(input_paths))
    input_bytes = 0
    for path in input_paths:
        _require_regular_file(path)
        input_bytes += path.stat().st_size
    argv = [
        str(executable),
        str(selected_threads),
        str(output_path),
        *(str(path) for path in input_paths),
    ]
    start_message = (
        "Running record-multiset helper with "
        f"{selected_threads} threads over {len(input_paths)} files "
        f"({input_bytes} bytes)"
    )
    _append_log(log_path, start_message)
    print(f"🧬 validator {start_message}", flush=True)
    started = perf_counter()
    with log_path.open("ab") as log:
        completed = subprocess.run(  # noqa: S603
            argv,
            check=False,
            stdout=log,
            stderr=log,
        )
    elapsed = perf_counter() - started
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, argv)
    _require_regular_file(output_path)
    report = _load_json_object(output_path)
    if report.get("files") != len(input_paths):
        raise ValueError("Record-multiset helper reported the wrong file count")
    if report.get("threads") != selected_threads:
        raise ValueError("Record-multiset helper reported the wrong thread count")
    _record_multiset_signature(report)
    completed_message = (
        "Completed record-multiset helper in "
        f"{elapsed:.6f}s at "
        f"{input_bytes / elapsed if elapsed else 0.0:.3f} bytes/s"
    )
    _append_log(log_path, completed_message)
    print(f"🧬 validator {completed_message}", flush=True)
    return {
        "input_bytes": input_bytes,
        "wall_seconds": elapsed,
        "throughput_bytes_per_second": (input_bytes / elapsed if elapsed else 0.0),
        "report": report,
    }


def _run_record_multiset_validation(
    source_path: Path,
    shard_paths: tuple[Path, ...],
    scratch_root: Path,
    output_path: Path,
    log_path: Path,
    *,
    threads: int,
) -> dict[str, object]:
    """Compare canonical full-record multisets with composable shard scans."""
    executable = _compile_record_multiset_helper(scratch_root, log_path)
    source_result = _run_record_multiset_helper(
        executable,
        (source_path,),
        scratch_root / "source-record-multiset.json",
        log_path,
        threads=1,
    )
    shard_result = _run_record_multiset_helper(
        executable,
        shard_paths,
        scratch_root / "shard-record-multiset.json",
        log_path,
        threads=threads,
    )
    source_report = source_result.get("report")
    shard_report = shard_result.get("report")
    if not isinstance(source_report, dict) or not isinstance(shard_report, dict):
        raise TypeError("Record-multiset helper result lost its report")
    source_signature = _record_multiset_signature(cast(dict[str, Any], source_report))
    shard_signature = _record_multiset_signature(cast(dict[str, Any], shard_report))
    source_signature_sha256 = hashlib.sha256(_json_bytes(source_signature)).hexdigest()
    shard_signature_sha256 = hashlib.sha256(_json_bytes(shard_signature)).hexdigest()
    match = source_signature == shard_signature
    result: dict[str, object] = {
        "match": match,
        "algorithm": _record_multiset_identity(),
        "source_signature_sha256": source_signature_sha256,
        "shard_signature_sha256": shard_signature_sha256,
        "source_signature": source_signature,
        "shard_signature": shard_signature,
        "source": source_result,
        "shards": shard_result,
    }
    if match:
        result["signature_sha256"] = source_signature_sha256
        result["signature"] = source_signature
    _write_json_atomic(output_path, result)
    if not match:
        raise ValueError("Canonical source and shard record multisets differ")
    return result


def _required_ordinal_shuffler_scratch_bytes(
    source_size: int,
    record_count: int,
) -> int:
    """Return local bytes needed for the staged source, shuffle, and index."""
    if (
        isinstance(source_size, bool)
        or not isinstance(source_size, int)
        or source_size <= 0
    ):
        raise ValueError("source_size must be a positive integer")
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count <= 0
    ):
        raise ValueError("record_count must be a positive integer")
    index_size = 48 + (record_count + 1) * 8
    return (
        source_size + source_size + 1 + index_size + PRODUCTION_SCRATCH_HEADROOM_BYTES
    )


def _run_ordinal_two_pass_shuffle(
    source_path: Path,
    shuffled_path: Path,
    scratch_root: Path,
    diagnostics_path: Path,
    metrics_path: Path,
    log_path: Path,
    *,
    expected_records: int,
    worker_threads: int,
) -> dict[str, Any]:
    """Shuffle occurrences with a compact index and ordered concurrent reads."""
    if (
        isinstance(expected_records, bool)
        or not isinstance(expected_records, int)
        or expected_records <= 0
    ):
        raise ValueError("expected_records must be a positive integer")
    threads = _validate_seqkit_threads(worker_threads)
    _require_regular_file(source_path)
    executable = _compile_ordinal_shuffler(scratch_root, log_path)
    staged_source_path = scratch_root / "source.fasta"
    index_path = scratch_root / "occurrence-offsets.bin"
    argv = [
        str(executable),
        "--source",
        str(source_path),
        "--staged-source",
        str(staged_source_path),
        "--output",
        str(shuffled_path),
        "--index",
        str(index_path),
        "--expected-records",
        str(expected_records),
        "--seed",
        str(SHARD_RANDOM_SEED),
        "--threads",
        str(threads),
        "--prefetch-records",
        str(ORDINAL_SHUFFLER_PREFETCH_RECORDS),
        "--prefetch-bytes",
        str(ORDINAL_SHUFFLER_PREFETCH_BYTES),
    ]
    _append_log(log_path, f"Running command: {shlex.join(argv)}")
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics_path.open("xb") as diagnostics:
        process = subprocess.Popen(  # noqa: S603
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("Native shuffler did not expose output streams")
        for line in iter(process.stderr.readline, b""):
            diagnostics.write(line)
            diagnostics.flush()
            print(f"🧬 shuffler {line.decode(errors='replace')}", end="", flush=True)
        process.stderr.close()
        metrics_bytes = process.stdout.read()
        process.stdout.close()
        returncode = process.wait()
    _append_diagnostic_file(diagnostics_path, log_path)
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, argv)
    try:
        metrics = orjson.loads(metrics_bytes)
    except orjson.JSONDecodeError as exc:
        raise ValueError("Native shuffler returned invalid metrics JSON") from exc
    if not isinstance(metrics, dict):
        raise ValueError("Native shuffler metrics must be a JSON object")
    source_size = source_path.stat().st_size
    with source_path.open("rb") as source:
        source.seek(-1, os.SEEK_END)
        output_size = source_size + (source.read(1) != b"\n")
    expected_metrics = {
        "schema_version": 1,
        "version": ORDINAL_SHUFFLER_VERSION,
        "record_count": expected_records,
        "source_size_bytes": source_size,
        "staged_source_size_bytes": source_size,
        "output_size_bytes": output_size,
        "seed": SHARD_RANDOM_SEED,
        "threads": threads,
        "prefetch_records": ORDINAL_SHUFFLER_PREFETCH_RECORDS,
        "prefetch_bytes": ORDINAL_SHUFFLER_PREFETCH_BYTES,
        "random_read_source": "container-local-staged-copy",
    }
    for key, expected in expected_metrics.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"Native shuffler metric {key!r} is {metrics.get(key)!r}, "
                f"expected {expected!r}"
            )
    if metrics.get("offset_index_size_bytes") != 48 + (expected_records + 1) * 8:
        raise ValueError("Native shuffler offset index has an unexpected size")
    if metrics.get("permutation_size_bytes") != expected_records * 4:
        raise ValueError("Native shuffler permutation has an unexpected size")
    for key in (
        "peak_batch_bytes",
        "first_pass_seconds",
        "permutation_seconds",
        "second_pass_seconds",
    ):
        value = metrics.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            raise ValueError(f"Native shuffler metric {key!r} is invalid")
    _require_regular_file(shuffled_path)
    if shuffled_path.stat().st_size != output_size:
        raise ValueError("Native shuffler output size is not normalized")
    _require_regular_file(staged_source_path)
    if staged_source_path.stat().st_size != source_size:
        raise ValueError("Native shuffler staged source size does not match source")
    first_pass_seconds = float(metrics["first_pass_seconds"])
    second_pass_seconds = float(metrics["second_pass_seconds"])
    published_metrics = metrics | {
        "source_code_sha256": ORDINAL_SHUFFLER_SOURCE_SHA256,
        "index_identity": "uint64-source-occurrence-offsets-v1",
        "permutation_identity": "splitmix64-fisher-yates-u32-v1",
        "staging_identity": "first-pass-tee-to-container-local-v1",
        "read_identity": "bounded-concurrent-local-pread-ordered-write-v2",
        "first_pass_bytes_per_second": (
            source_size / first_pass_seconds if first_pass_seconds > 0 else None
        ),
        "second_pass_bytes_per_second": (
            output_size / second_pass_seconds if second_pass_seconds > 0 else None
        ),
    }
    _write_json_atomic(metrics_path, published_metrics)
    return published_metrics


def _run_to_file(argv: list[str], output_path: Path, log_path: Path) -> None:
    """Run a fixed argv command with separate data and diagnostic streams."""
    _append_log(log_path, f"Running command: {shlex.join(argv)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as output, log_path.open("ab") as log:
        completed = subprocess.run(  # noqa: S603
            argv,
            check=False,
            stdout=output,
            stderr=log,
        )
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, argv)


def _run_shuffle_split(
    source_path: Path,
    raw_shard_dir: Path,
    shard_dir: Path,
    validation_dir: Path,
    log_path: Path,
    *,
    seqkit_threads: int,
) -> dict[str, int | str]:
    """Shuffle, recover FAI-omitted records, split, and restore headers."""
    seqkit = _require_executable("seqkit")
    shuffled_path = raw_shard_dir.parent / ".shuffled.fasta"
    shuffle_diagnostics_path = validation_dir / "shuffle-stderr.log"
    recovery_report_path = validation_dir / "duplicate-recovery.jsonl"
    shuffle_argv = [
        seqkit,
        "shuffle",
        "-j",
        str(seqkit_threads),
        "--two-pass",
        "--update-faidx",
        "--rand-seed",
        str(SHARD_RANDOM_SEED),
        str(source_path),
    ]
    _append_log(log_path, f"Running command: {shlex.join(shuffle_argv)}")
    validation_dir.mkdir(parents=True, exist_ok=True)
    with (
        shuffled_path.open("xb") as shuffled,
        shuffle_diagnostics_path.open("xb") as diagnostics,
    ):
        shuffle_process = subprocess.run(  # noqa: S603
            shuffle_argv,
            check=False,
            stdout=shuffled,
            stderr=diagnostics,
        )
    _append_diagnostic_file(shuffle_diagnostics_path, log_path)
    if shuffle_process.returncode != 0:
        raise subprocess.CalledProcessError(shuffle_process.returncode, shuffle_argv)

    warnings = _parse_fai_duplicate_warnings(shuffle_diagnostics_path)
    if len(warnings) != EXPECTED_RECOVERED_RECORDS:
        raise ValueError(
            "Unexpected number of SeqKit FAI duplicate warnings: "
            f"{len(warnings)} != {EXPECTED_RECOVERED_RECORDS}"
        )
    temporary_namespace = f"{RECOVERED_HEADER_NAMESPACE}{uuid.uuid4().hex}_"
    recovery_metrics = _append_recovered_fasta_records(
        source_path,
        shuffled_path,
        warnings,
        recovery_report_path,
        temporary_namespace=temporary_namespace,
    )
    if recovery_metrics["recovered_residues"] != EXPECTED_RECOVERED_RESIDUES:
        raise ValueError(
            "Recovered duplicate residue count does not match the failed-run "
            f"deficit: {recovery_metrics['recovered_residues']} != "
            f"{EXPECTED_RECOVERED_RESIDUES}"
        )
    _append_log(
        log_path,
        "Recovered "
        f"{recovery_metrics['recovered_records']} FAI-omitted records "
        f"({recovery_metrics['recovered_residues']} residues) from byte offsets "
        f"{recovery_metrics['first_byte_offset']} through "
        f"{recovery_metrics['last_byte_offset']}",
    )

    split_argv = [
        seqkit,
        "split2",
        "-j",
        str(seqkit_threads),
        "--by-part",
        str(SHARD_COUNT),
        "--out-dir",
        str(raw_shard_dir),
        "--force",
        "--out-prefix",
        "part_",
        str(shuffled_path),
    ]
    _append_log(log_path, f"Running command: {shlex.join(split_argv)}")
    raw_shard_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        split_process = subprocess.run(  # noqa: S603
            split_argv,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=log,
        )
    if split_process.returncode != 0:
        raise subprocess.CalledProcessError(split_process.returncode, split_argv)

    raw_shards = sorted(
        path
        for path in raw_shard_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    )
    if len(raw_shards) != SHARD_COUNT:
        raise ValueError(f"Expected {SHARD_COUNT} raw shards, found {len(raw_shards)}")
    shard_dir.mkdir(parents=True, exist_ok=True)
    recovery_pattern = _recovery_header_pattern(temporary_namespace)
    for index, raw_shard in enumerate(raw_shards):
        if raw_shard.stat().st_size <= 0:
            raise ValueError(f"SeqKit produced empty shard: {raw_shard}")
        _run_to_file(
            [
                seqkit,
                "replace",
                "-j",
                str(seqkit_threads),
                "--pattern",
                recovery_pattern,
                "--replacement",
                "",
                str(raw_shard),
            ],
            shard_dir / _shard_filename(index),
            log_path,
        )

    for raw_shard in raw_shards:
        raw_shard.unlink()
    raw_shard_dir.rmdir()
    shuffled_path.unlink()
    Path(f"{source_path}.seqkit.fai").unlink(missing_ok=True)
    return recovery_metrics | {
        "temporary_namespace": temporary_namespace,
        "temporary_header_pattern": recovery_pattern,
    }


def _run_aggregate_seqkit_sum(
    shard_paths: tuple[Path, ...],
    output_path: Path,
    log_path: Path,
    *,
    seqkit_threads: int,
) -> None:
    """Stream every shard into one order-independent SeqKit checksum."""
    cat = _require_executable("cat")
    seqkit = _require_executable("seqkit")
    cat_argv = [cat, *(str(path) for path in shard_paths)]
    sum_argv = [seqkit, "sum", "-j", str(seqkit_threads), "--all", "-"]
    _append_log(log_path, f"Running command: {shlex.join(cat_argv)}")
    _append_log(log_path, f"Piping into: {shlex.join(sum_argv)}")
    with output_path.open("xb") as output, log_path.open("ab") as log:
        cat_process = subprocess.Popen(  # noqa: S603
            cat_argv,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        if cat_process.stdout is None:
            cat_process.kill()
            raise RuntimeError("cat did not expose stdout")
        try:
            sum_process = subprocess.run(  # noqa: S603
                sum_argv,
                check=False,
                stdin=cat_process.stdout,
                stdout=output,
                stderr=log,
            )
        finally:
            cat_process.stdout.close()
        cat_returncode = cat_process.wait()
    if cat_returncode != 0:
        raise subprocess.CalledProcessError(cat_returncode, cat_argv)
    if sum_process.returncode != 0:
        raise subprocess.CalledProcessError(sum_process.returncode, sum_argv)


def _seqkit_sum_digest(path: Path) -> str:
    """Extract the digest token from one SeqKit sum output file."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise ValueError(f"Expected one SeqKit sum row in {path}, got {len(lines)}")
    fields = lines[0].split("\t")
    if not fields or not fields[0].startswith("seqkit."):
        raise ValueError(f"Invalid SeqKit sum output in {path}: {lines[0]!r}")
    return fields[0]


def _validate_profile_statistics(
    source_stats_path: Path,
    shard_stats_path: Path,
    shard_summary_path: Path,
) -> dict[str, int | float]:
    """Validate SeqKit statistics and persist the normalized shard table."""
    import polars as pl

    source_stats = pl.read_csv(source_stats_path, separator="\t")
    shard_stats = pl.read_csv(shard_stats_path, separator="\t")
    required_columns = {"file", "num_seqs", "sum_len"}
    if not required_columns.issubset(source_stats.columns):
        raise ValueError(f"Source stats missing columns: {required_columns}")
    if not required_columns.issubset(shard_stats.columns):
        raise ValueError(f"Shard stats missing columns: {required_columns}")
    if source_stats.height != 1:
        raise ValueError(f"Expected one source stats row, got {source_stats.height}")
    if shard_stats.height != SHARD_COUNT:
        raise ValueError(
            f"Expected {SHARD_COUNT} shard stats rows, got {shard_stats.height}"
        )

    shard_stats = shard_stats.with_columns(
        pl
        .col("file")
        .cast(pl.String)
        .str.replace_all(r"\\", "/")
        .str.split("/")
        .list.last()
        .alias("basename")
    ).sort("basename")
    expected_names = list(_shard_names())
    actual_names = shard_stats.get_column("basename").to_list()
    if actual_names != expected_names:
        raise ValueError("SeqKit stats shard names do not match the profile")

    source_num_seqs = int(source_stats.item(0, "num_seqs"))
    source_sum_len = int(source_stats.item(0, "sum_len"))
    shard_num_seqs = int(shard_stats.get_column("num_seqs").sum())
    shard_sum_len = int(shard_stats.get_column("sum_len").sum())
    if source_num_seqs != SMALL_BFD_Z:
        raise ValueError(
            f"Expected small-BFD Z={SMALL_BFD_Z}, measured {source_num_seqs}"
        )
    if shard_num_seqs != source_num_seqs:
        raise ValueError(
            f"Shard sequence count {shard_num_seqs} != source {source_num_seqs}"
        )
    if shard_sum_len != source_sum_len:
        raise ValueError(
            f"Shard residue count {shard_sum_len} != source {source_sum_len}"
        )

    mean_sum_len = source_sum_len / SHARD_COUNT
    maximum_imbalance = max(
        abs(int(value) - mean_sum_len) / mean_sum_len
        for value in shard_stats.get_column("sum_len")
    )
    if maximum_imbalance > MAX_PROFILE_IMBALANCE:
        raise ValueError(
            "Shard residue imbalance exceeds "
            f"{MAX_PROFILE_IMBALANCE:.0%}: {maximum_imbalance:.3%}"
        )

    shard_summary_path.parent.mkdir(parents=True, exist_ok=True)
    shard_stats.write_parquet(shard_summary_path)
    return {
        "num_seqs": source_num_seqs,
        "sum_len": source_sum_len,
        "maximum_residue_imbalance": maximum_imbalance,
    }


def _artifact_record(
    path: Path,
    profile_root: Path,
    *,
    forbidden_bytes: bytes | None = None,
) -> dict[str, str | int]:
    """Build one manifest artifact record below the profile root."""
    _require_regular_file(path)
    resolved_root = profile_root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"Artifact escapes profile root: {path}")
    return {
        "path": resolved_path.relative_to(resolved_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path, forbidden_bytes=forbidden_bytes),
    }


def _validate_published_profile(
    profile_root: Path,
    *,
    verify_digests: bool,
) -> dict[str, Any]:
    """Validate a published profile manifest and its declared artifacts."""
    manifest_path = profile_root / "manifest.json"
    _require_regular_file(manifest_path)
    manifest = _load_json_object(manifest_path)
    source, shards, validation_artifacts = _validate_profile_manifest(manifest)

    records = [source, *shards, *validation_artifacts]
    for record in records:
        relative = str(record["path"])
        artifact_path = (profile_root / relative).resolve()
        if not artifact_path.is_relative_to(profile_root.resolve()):
            raise ValueError(f"Profile artifact escapes root: {relative}")
        _require_regular_file(artifact_path)
        if artifact_path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Profile artifact size mismatch: {relative}")
        if verify_digests and _sha256_file(artifact_path) != record["sha256"]:
            raise ValueError(f"Profile artifact digest mismatch: {relative}")
    return manifest


def _validate_profile_manifest(
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate profile metadata without requiring a mounted filesystem."""
    if manifest.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError("Unexpected profile manifest schema version")
    if manifest.get("profile_id") != PROFILE_ID:
        raise ValueError("Unexpected profile ID")
    if manifest.get("database_id") != DATABASE_ID:
        raise ValueError("Unexpected database ID")
    if manifest.get("shard_count") != SHARD_COUNT:
        raise ValueError("Unexpected shard count")
    if manifest.get("z_value") != SMALL_BFD_Z:
        raise ValueError("Unexpected small-BFD Z value")

    source = manifest.get("source")
    shards = manifest.get("shards")
    recipe = manifest.get("recipe")
    validation = manifest.get("validation")
    if not isinstance(source, dict):
        raise ValueError("Profile manifest source must be an object")
    if source.get("path") != f"source/{SOURCE_DB_FILENAME}":
        raise ValueError("Profile manifest source path is invalid")
    if not isinstance(shards, list) or len(shards) != SHARD_COUNT:
        raise ValueError(f"Profile manifest must declare {SHARD_COUNT} shards")
    if not isinstance(recipe, dict):
        raise ValueError("Profile manifest recipe must be an object")
    if recipe.get("version") != PROFILE_RECIPE_VERSION:
        raise ValueError("Unexpected profile recipe version")
    if recipe.get("seqkit_version") != SEQKIT_VERSION:
        raise ValueError("Unexpected profile SeqKit version")
    try:
        _validate_seqkit_threads(recipe.get("seqkit_threads"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid profile SeqKit thread count") from exc
    if recipe.get("random_seed") != SHARD_RANDOM_SEED:
        raise ValueError("Unexpected profile shuffle seed")
    if recipe.get("shuffle") != ["--two-pass", "--update-faidx"]:
        raise ValueError("Unexpected profile shuffle recipe")
    if recipe.get("duplicate_recovery") != _duplicate_recovery_recipe():
        raise ValueError("Unexpected profile duplicate-recovery recipe")
    if recipe.get("split") != ["--by-part", SHARD_COUNT]:
        raise ValueError("Unexpected profile split recipe")
    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("Profile manifest does not declare passed validation")
    if validation.get("recovered_records") != EXPECTED_RECOVERED_RECORDS:
        raise ValueError("Unexpected recovered duplicate-record count")
    if validation.get("recovered_residues") != EXPECTED_RECOVERED_RESIDUES:
        raise ValueError("Unexpected recovered duplicate-residue count")
    if validation.get("temporary_recovery_prefix_absent") is not True:
        raise ValueError("Profile may retain temporary recovery prefixes")
    validation_artifacts = validation.get("artifacts")
    if not isinstance(validation_artifacts, list):
        raise ValueError("Profile manifest validation artifacts must be a list")

    expected_paths = [f"shards/{name}" for name in _shard_names()]
    actual_paths: list[str] = []
    records = [source, *shards, *validation_artifacts]
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Profile artifact record must be an object")
        relative = record.get("path")
        size_bytes = record.get("size_bytes")
        digest = record.get("sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Profile artifact path must be relative")
        if ".." in PurePosixPath(relative).parts:
            raise ValueError(f"Profile artifact path escapes root: {relative}")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise ValueError(f"Invalid artifact size: {relative}")
        if size_bytes <= 0:
            raise ValueError(f"Profile artifact is empty: {relative}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid artifact SHA-256: {relative}")
        if relative.startswith("shards/"):
            actual_paths.append(relative)
    if actual_paths != expected_paths:
        raise ValueError("Profile manifest shard order or names are invalid")
    actual_validation_paths = [str(record["path"]) for record in validation_artifacts]
    if actual_validation_paths != list(PROFILE_VALIDATION_RELPATHS):
        raise ValueError("Profile manifest validation artifact paths are invalid")
    return source, shards, validation_artifacts


def _build_prepare_plan(seqkit_threads: int) -> dict[str, object]:
    """Build the side-effect-free profile-preparation plan."""
    threads = _validate_seqkit_threads(seqkit_threads)
    return {
        "campaign_id": CAMPAIGN_ID,
        "operation": "prepare",
        "remote_calls": 1,
        "resources": {
            "cpu": [0.125, 32.125],
            "memory_mib": [1024, 131_072],
            "timeout_seconds": CONF.timeout,
        },
        "source": {
            "volume": SOURCE_DB_VOLUME_NAME,
            "path": SOURCE_DB_FILENAME,
            "mount": "read-only",
        },
        "destination": {
            "volume": SHARDED_DB_VOLUME_NAME,
            "profile": APP_INFO.profile_relpath,
        },
        "seqkit": {
            "version": SEQKIT_VERSION,
            "threads": threads,
            "random_seed": SHARD_RANDOM_SEED,
            "shards": SHARD_COUNT,
            "duplicate_recovery": _duplicate_recovery_recipe(),
        },
        "existing_profile_policy": "validate-and-reuse",
    }


def _prepare_profile(seqkit_threads: int) -> dict[str, object]:
    """Build, validate, and publish the small-BFD profile."""
    threads = _validate_seqkit_threads(seqkit_threads)
    source_root = Path(APP_INFO.source_db_dir)
    sharded_root = Path(APP_INFO.sharded_db_dir)
    output_root = Path(APP_INFO.output_dir)
    source_path = source_root / SOURCE_DB_FILENAME
    profile_root = sharded_root / APP_INFO.profile_relpath
    evidence_root = output_root / APP_INFO.preparation_relpath
    log_path = evidence_root / "run.log"
    evidence_root.mkdir(parents=True, exist_ok=True)
    _append_log(log_path, f"Preparing profile {PROFILE_ID}")

    SHARDED_MSA_DB_VOLUME.reload()
    BENCHMARK_OUTPUT_VOLUME.reload()
    _ensure_campaign_plan_mounted()
    BENCHMARK_OUTPUT_VOLUME.commit()
    if (profile_root / "manifest.json").is_file():
        manifest = _validate_published_profile(profile_root, verify_digests=True)
        result = {
            "status": "reused",
            "profile_path": str(profile_root),
            "manifest_sha256": _sha256_file(profile_root / "manifest.json"),
        }
        _write_json_atomic(evidence_root / "metrics.json", result)
        BENCHMARK_OUTPUT_VOLUME.commit()
        _write_json_atomic(
            evidence_root / "done.json",
            result | {"completed_at": _utc_now(), "profile_id": manifest["profile_id"]},
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        return result

    if profile_root.exists():
        orphan_root = sharded_root / ".orphaned"
        orphan_root.mkdir(parents=True, exist_ok=True)
        orphan_path = orphan_root / f"{PROFILE_ID}-{uuid.uuid4().hex}"
        profile_root.replace(orphan_path)
        SHARDED_MSA_DB_VOLUME.commit()
        _append_log(log_path, f"Preserved incomplete profile at {orphan_path}")

    generation_id = uuid.uuid4().hex
    staging_root = sharded_root / ".staging" / f"{PROFILE_ID}-{generation_id}"
    staging_source_dir = staging_root / "source"
    raw_shard_dir = staging_root / ".raw-shards"
    shard_dir = staging_root / "shards"
    validation_dir = staging_root / "validation"
    staging_source_dir.mkdir(parents=True)
    shard_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)

    copied_source = staging_source_dir / SOURCE_DB_FILENAME
    source_sha256, source_size = _copy_file_with_sha256(source_path, copied_source)
    copied_sha256 = _sha256_file(copied_source)
    if copied_sha256 != source_sha256:
        raise ValueError("Copied small-BFD SHA-256 does not match source")
    if copied_source.stat().st_size != source_size:
        raise ValueError("Copied small-BFD byte size does not match source")
    SHARDED_MSA_DB_VOLUME.commit()

    seqkit = _require_executable("seqkit")
    version_output = subprocess.run(  # noqa: S603
        [seqkit, "version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if SEQKIT_VERSION not in version_output:
        raise RuntimeError(
            f"Expected SeqKit {SEQKIT_VERSION}, observed {version_output!r}"
        )
    _append_log(log_path, f"Using {version_output}")

    recovery_metrics = _run_shuffle_split(
        copied_source,
        raw_shard_dir,
        shard_dir,
        validation_dir,
        log_path,
        seqkit_threads=threads,
    )
    shuffle_log_path = validation_dir / "shuffle-stderr.log"
    recovery_report_path = validation_dir / "duplicate-recovery.jsonl"
    evidence_shuffle_path = evidence_root / f"{generation_id}-shuffle-stderr.log"
    evidence_recovery_path = evidence_root / f"{generation_id}-duplicate-recovery.jsonl"
    shuffle_log_sha256, shuffle_log_size = _copy_file_with_sha256(
        shuffle_log_path,
        evidence_shuffle_path,
    )
    recovery_report_sha256, recovery_report_size = _copy_file_with_sha256(
        recovery_report_path,
        evidence_recovery_path,
    )
    _write_json_atomic(
        evidence_root / "recovery.json",
        recovery_metrics
        | {
            "generation_id": generation_id,
            "shuffle_diagnostics": {
                "path": evidence_shuffle_path.name,
                "sha256": shuffle_log_sha256,
                "size_bytes": shuffle_log_size,
            },
            "recovery_report": {
                "path": evidence_recovery_path.name,
                "sha256": recovery_report_sha256,
                "size_bytes": recovery_report_size,
            },
        },
    )
    BENCHMARK_OUTPUT_VOLUME.commit()
    shard_paths = tuple(shard_dir / name for name in _shard_names())

    source_stats_path = validation_dir / "source-stats.tsv"
    shard_stats_path = validation_dir / "shard-stats.tsv"
    shard_summary_path = validation_dir / "shard-summary.parquet"
    source_sum_path = validation_dir / "source-sum.tsv"
    shard_sum_path = validation_dir / "shard-sum.tsv"
    _run_to_file(
        [
            seqkit,
            "stats",
            "-j",
            str(threads),
            "--all",
            "--tabular",
            str(copied_source),
        ],
        source_stats_path,
        log_path,
    )
    _run_to_file(
        [
            seqkit,
            "stats",
            "-j",
            str(threads),
            "--all",
            "--tabular",
            *(str(path) for path in shard_paths),
        ],
        shard_stats_path,
        log_path,
    )
    _run_to_file(
        [
            seqkit,
            "sum",
            "-j",
            str(threads),
            "--all",
            str(copied_source),
        ],
        source_sum_path,
        log_path,
    )
    _run_aggregate_seqkit_sum(
        shard_paths,
        shard_sum_path,
        log_path,
        seqkit_threads=threads,
    )
    source_sum = _seqkit_sum_digest(source_sum_path)
    shard_sum = _seqkit_sum_digest(shard_sum_path)
    if source_sum != shard_sum:
        raise ValueError("Aggregate shard SeqKit sum does not match source")
    statistics = _validate_profile_statistics(
        source_stats_path,
        shard_stats_path,
        shard_summary_path,
    )
    seqkit_sum_report_path = validation_dir / "seqkit-sum.json"
    _write_json_atomic(
        seqkit_sum_report_path,
        {
            "source": source_sum,
            "aggregate_shards": shard_sum,
            "match": True,
            "seqkit_version": SEQKIT_VERSION,
        },
    )

    source_record = _artifact_record(copied_source, staging_root)
    temporary_namespace = recovery_metrics.get("temporary_namespace")
    if not isinstance(temporary_namespace, str):
        raise RuntimeError("Duplicate recovery did not return its temporary namespace")
    forbidden_recovery_header = b">" + temporary_namespace.encode("ascii")
    shard_records = [
        _artifact_record(
            shard_path,
            staging_root,
            forbidden_bytes=forbidden_recovery_header,
        )
        for shard_path in shard_paths
    ]
    validation_records = [
        _artifact_record(staging_root / relative_path, staging_root)
        for relative_path in PROFILE_VALIDATION_RELPATHS
    ]
    manifest: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_id": PROFILE_ID,
        "database_id": DATABASE_ID,
        "created_at": _utc_now(),
        "generation_id": generation_id,
        "source_volume": SOURCE_DB_VOLUME_NAME,
        "source": source_record,
        "shard_count": SHARD_COUNT,
        "shard_prefix": f"shards/{SOURCE_DB_FILENAME}",
        "shards": shard_records,
        "z_value": SMALL_BFD_Z,
        "recipe": {
            "version": PROFILE_RECIPE_VERSION,
            "seqkit_version": SEQKIT_VERSION,
            "seqkit_threads": threads,
            "random_seed": SHARD_RANDOM_SEED,
            "shuffle": ["--two-pass", "--update-faidx"],
            "duplicate_recovery": _duplicate_recovery_recipe(),
            "split": ["--by-part", SHARD_COUNT],
        },
        "validation": {
            "passed": True,
            "source_sha256_matches_copy": True,
            "seqkit_sum": source_sum,
            "num_seqs": statistics["num_seqs"],
            "sum_len": statistics["sum_len"],
            "maximum_residue_imbalance": statistics["maximum_residue_imbalance"],
            "maximum_allowed_residue_imbalance": MAX_PROFILE_IMBALANCE,
            "recovered_records": recovery_metrics["recovered_records"],
            "recovered_residues": recovery_metrics["recovered_residues"],
            "first_recovered_byte_offset": recovery_metrics["first_byte_offset"],
            "last_recovered_byte_offset": recovery_metrics["last_byte_offset"],
            "temporary_recovery_prefix_absent": True,
            "artifacts": validation_records,
        },
    }

    _write_json_atomic(staging_root / "manifest.json", manifest)
    SHARDED_MSA_DB_VOLUME.commit()
    _validate_published_profile(staging_root, verify_digests=False)

    publication_status = "published"
    profile_root.parent.mkdir(parents=True, exist_ok=True)
    if profile_root.exists():
        try:
            existing_manifest = _validate_published_profile(
                profile_root,
                verify_digests=False,
            )
        except (FileNotFoundError, ValueError):
            orphan_root = sharded_root / ".orphaned"
            orphan_root.mkdir(parents=True, exist_ok=True)
            profile_root.replace(orphan_root / f"{PROFILE_ID}-{uuid.uuid4().hex}")
        else:
            if _profile_scientific_identity(existing_manifest) != (
                _profile_scientific_identity(manifest)
            ):
                raise RuntimeError(
                    "A different valid profile was published concurrently"
                )
            duplicate_root = sharded_root / ".orphaned"
            duplicate_root.mkdir(parents=True, exist_ok=True)
            staging_root.replace(
                duplicate_root / f"{PROFILE_ID}-duplicate-{generation_id}"
            )
            publication_status = "reused-concurrent"
    if publication_status == "published":
        staging_root.replace(profile_root)
    SHARDED_MSA_DB_VOLUME.commit()
    _validate_published_profile(profile_root, verify_digests=False)

    result = {
        "status": publication_status,
        "profile_path": str(profile_root),
        "manifest_sha256": _sha256_file(profile_root / "manifest.json"),
        "source_size_bytes": source_size,
        "source_sha256": source_sha256,
        "num_seqs": statistics["num_seqs"],
        "sum_len": statistics["sum_len"],
        "maximum_residue_imbalance": statistics["maximum_residue_imbalance"],
        "recovered_records": recovery_metrics["recovered_records"],
        "recovered_residues": recovery_metrics["recovered_residues"],
    }
    _append_log(log_path, f"Published profile {PROFILE_ID}")
    _write_json_atomic(evidence_root / "metrics.json", result)
    BENCHMARK_OUTPUT_VOLUME.commit()
    _write_json_atomic(
        evidence_root / "done.json",
        result | {"completed_at": _utc_now(), "profile_id": PROFILE_ID},
    )
    BENCHMARK_OUTPUT_VOLUME.commit()
    return result


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 131_072),
    timeout=CONF.timeout,
    max_containers=1,
    volumes={
        APP_INFO.source_db_dir: SOURCE_MSA_DB_VOLUME.with_mount_options(read_only=True),
        APP_INFO.sharded_db_dir: SHARDED_MSA_DB_VOLUME,
        APP_INFO.output_dir: BENCHMARK_OUTPUT_VOLUME,
    },
)
def prepare_small_bfd_profile(
    seqkit_threads: int = DEFAULT_SEQKIT_THREADS,
) -> dict[str, object]:
    """Prepare and validate the immutable 64-shard small-BFD profile.

    Args:
        seqkit_threads: SeqKit thread count, from 1 through 32.

    Returns:
        Primitive publication status and profile provenance.
    """
    try:
        return _prepare_profile(seqkit_threads)
    except Exception as exc:
        evidence_root = Path(APP_INFO.output_dir) / APP_INFO.preparation_relpath
        evidence_root.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(
            evidence_root / "failure.json",
            {
                "failed_at": _utc_now(),
                "profile_id": PROFILE_ID,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        raise


def _database_profile_spec(database_id: str) -> DatabaseProfileSpec:
    """Resolve one fixed database ID without accepting free-form paths."""
    if not isinstance(database_id, str):
        raise TypeError("database_id must be a string")
    for spec in DATABASE_PROFILE_SPECS:
        if spec.database_id == database_id:
            return spec
    choices = ", ".join(spec.database_id for spec in DATABASE_PROFILE_SPECS)
    raise ValueError(f"Unknown database_id {database_id!r}; expected one of {choices}")


def _validate_source_policy(source_policy: str) -> str:
    """Validate the post-publication source-retirement policy."""
    if not isinstance(source_policy, str):
        raise TypeError("source_policy must be a string")
    if source_policy not in SOURCE_POLICIES:
        choices = ", ".join(SOURCE_POLICIES)
        raise ValueError(
            f"Unknown source_policy {source_policy!r}; expected one of {choices}"
        )
    return source_policy


def _production_shard_filename(
    spec: DatabaseProfileSpec,
    index: int,
) -> str:
    """Return one fixed AlphaFold-compatible shard filename."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("shard index must be an integer")
    if not 0 <= index < spec.shard_count:
        raise ValueError(f"shard index must be in [0, {spec.shard_count}), got {index}")
    return f"{spec.source_filename}-{index:05d}-of-{spec.shard_count:05d}"


def _production_shard_names(
    spec: DatabaseProfileSpec,
) -> tuple[str, ...]:
    """Return every expected production-candidate shard name."""
    return tuple(
        _production_shard_filename(spec, index) for index in range(spec.shard_count)
    )


def _production_profile_root(
    sharded_root: Path,
    spec: DatabaseProfileSpec,
) -> Path:
    """Return the fixed profile root below the sharded Volume."""
    return sharded_root / PRODUCTION_PROFILE_ROOT / spec.profile_id


def _production_profile_plan(
    database_id: str,
    seqkit_threads: int,
    source_policy: str,
) -> dict[str, object]:
    """Build a side-effect-free production-candidate profile plan."""
    spec = _database_profile_spec(database_id)
    threads = _validate_seqkit_threads(seqkit_threads)
    policy = _validate_source_policy(source_policy)
    return {
        "operation": "build-profile",
        "remote_calls": 1,
        "database": {
            "database_id": spec.database_id,
            "profile_id": spec.profile_id,
            "polymer": spec.polymer,
            "source_filename": spec.source_filename,
            "shard_count": spec.shard_count,
            "expected_num_seqs": spec.expected_num_seqs,
            "expected_sum_len": spec.expected_sum_len,
            "search_space_value": spec.search_space_value,
            "search_space_unit": spec.search_space_unit,
        },
        "resources": {
            "cpu": [0.125, 32.125],
            "memory_mib": list(PRODUCTION_BUILD_MEMORY_MIB),
            "ephemeral_disk_mib": "platform-default",
            "timeout_seconds": PRODUCTION_BUILD_TIMEOUT_SECONDS,
        },
        "source": {
            "volume": SOURCE_DB_VOLUME_NAME,
            "path": spec.source_filename,
            "policy_after_validation": policy,
        },
        "destination": {
            "volume": SHARDED_DB_VOLUME_NAME,
            "profile": (f"{PRODUCTION_PROFILE_ROOT}/{spec.profile_id}"),
            "persistent_payloads": [
                "shards",
                "validation",
                "manifest.json",
            ],
        },
        "scratch": {
            "staged_source": str(PRODUCTION_SCRATCH_ROOT),
            "shuffle": str(PRODUCTION_SCRATCH_ROOT),
            "occurrence_index": str(PRODUCTION_SCRATCH_ROOT),
            "permutation": "memory",
            "raw_shards": SHARDED_DB_VOLUME_NAME,
        },
        "shuffle": {
            "version": ORDINAL_SHUFFLER_VERSION,
            "source_code_sha256": ORDINAL_SHUFFLER_SOURCE_SHA256,
            "passes": 2,
            "record_identity": "source-occurrence",
            "source_staging": "first-pass-tee-to-container-local-v1",
            "random_read_source": "container-local-staged-copy",
            "random_seed": SHARD_RANDOM_SEED,
            "permutation": "splitmix64-fisher-yates-u32-v1",
            "worker_threads": threads,
            "prefetch_records": ORDINAL_SHUFFLER_PREFETCH_RECORDS,
            "prefetch_bytes": ORDINAL_SHUFFLER_PREFETCH_BYTES,
            "ordered_output": True,
        },
        "seqkit": {
            "version": SEQKIT_VERSION,
            "threads": threads,
            "operations": ["stats", "split2"],
        },
        "record_multiset": {
            **_record_multiset_identity(),
            "source_threads": 1,
            "shard_threads": threads,
            "source_and_shards_compared": True,
        },
        "existing_profile_policy": "validate-and-reuse",
    }


def _record_multiset_benchmark_plan(
    seqkit_threads: int,
) -> dict[str, object]:
    """Plan the read-only recipe-v5 validator control on published UniProt."""
    threads = _validate_seqkit_threads(seqkit_threads)
    spec = _database_profile_spec("uniprot")
    return {
        "operation": "benchmark-validator",
        "remote_calls": 1,
        "database": {
            "database_id": spec.database_id,
            "profile_id": spec.profile_id,
            "source_filename": spec.source_filename,
            "shard_count": spec.shard_count,
        },
        "validator": {
            **_record_multiset_identity(),
            "source_threads": 1,
            "shard_threads": threads,
        },
        "inputs": {
            "source_volume": SOURCE_DB_VOLUME_NAME,
            "sharded_volume": SHARDED_DB_VOLUME_NAME,
            "mounts": "read-only",
            "source_and_shards_mutated": False,
        },
        "output_volume": OUTPUT_VOLUME_NAME,
        "baseline": UNIPROT_V4_VALIDATION_BASELINE,
        "resources": {
            "cpu": [0.125, 32.125],
            "memory_mib": list(PRODUCTION_BUILD_MEMORY_MIB),
            "timeout_seconds": CONF.timeout,
        },
    }


def _recover_profile_duplicate_records(
    source_path: Path,
    shuffled_path: Path,
    warnings: tuple[FaiDuplicateWarning, ...],
    report_path: Path,
    *,
    temporary_namespace: str,
) -> dict[str, int | str | None]:
    """Recover all FAI omissions while keeping a non-empty audit artifact."""
    detail_path = report_path.parent / f".{report_path.name}.details"
    if warnings:
        recovered = _append_recovered_fasta_records(
            source_path,
            shuffled_path,
            warnings,
            detail_path,
            temporary_namespace=temporary_namespace,
        )
        recovered_records = recovered["recovered_records"]
        recovered_residues = recovered["recovered_residues"]
        first_byte_offset: int | None = recovered["first_byte_offset"]
        last_byte_offset: int | None = recovered["last_byte_offset"]
    else:
        recovered_records = 0
        recovered_residues = 0
        first_byte_offset = None
        last_byte_offset = None
    metrics = {
        "recovered_records": recovered_records,
        "recovered_residues": recovered_residues,
        "first_byte_offset": first_byte_offset,
        "last_byte_offset": last_byte_offset,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("xb") as report:
        report.write(
            orjson.dumps(
                {
                    "kind": "summary",
                    **metrics,
                    "warning_source": "seqkit-fai-sequence-byte-offset",
                },
                option=JSONL_OPTIONS,
            )
        )
        if detail_path.is_file():
            with detail_path.open("rb") as details:
                shutil.copyfileobj(details, report, length=1024 * 1024)
        report.flush()
        os.fsync(report.fileno())
    detail_path.unlink(missing_ok=True)
    return metrics | {
        "temporary_namespace": temporary_namespace,
        "temporary_header_pattern": _recovery_header_pattern(temporary_namespace),
    }


def _run_production_shuffle_split(
    spec: DatabaseProfileSpec,
    source_path: Path,
    scratch_root: Path,
    raw_shard_dir: Path,
    shard_dir: Path,
    validation_dir: Path,
    log_path: Path,
    *,
    expected_records: int,
    seqkit_threads: int,
) -> dict[str, Any]:
    """Shuffle occurrences locally and split the validated result to the Volume."""
    seqkit = _require_executable("seqkit")
    shuffled_path = scratch_root / "shuffled.fasta"
    shuffle_diagnostics_path = validation_dir / "shuffle-stderr.log"
    shuffler_metrics_path = validation_dir / "shuffler-metrics.json"
    recovery_report_path = validation_dir / "duplicate-recovery.jsonl"
    validation_dir.mkdir(parents=True, exist_ok=True)
    shuffler_metrics = _run_ordinal_two_pass_shuffle(
        source_path,
        shuffled_path,
        scratch_root,
        shuffle_diagnostics_path,
        shuffler_metrics_path,
        log_path,
        expected_records=expected_records,
        worker_threads=seqkit_threads,
    )
    recovery_metrics: dict[str, int | str | None] = {
        "recovered_records": 0,
        "recovered_residues": 0,
        "first_byte_offset": None,
        "last_byte_offset": None,
        "temporary_namespace": None,
        "temporary_header_pattern": None,
    }
    with recovery_report_path.open("xb") as report:
        report.write(
            orjson.dumps(
                {
                    "kind": "summary",
                    **recovery_metrics,
                    "warning_source": None,
                    "record_identity": "source-occurrence",
                    "fai_duplicate_omission_possible": False,
                },
                option=JSONL_OPTIONS,
            )
        )
        report.flush()
        os.fsync(report.fileno())
    _append_log(
        log_path,
        "Preserved every FASTA record by source occurrence; FAI duplicate "
        "recovery is not applicable",
    )

    split_argv = [
        seqkit,
        "split2",
        "-j",
        str(seqkit_threads),
        "--by-part",
        str(spec.shard_count),
        "--out-dir",
        str(raw_shard_dir),
        "--force",
        "--out-prefix",
        "part_",
        str(shuffled_path),
    ]
    _append_log(log_path, f"Running command: {shlex.join(split_argv)}")
    raw_shard_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        split_process = subprocess.run(  # noqa: S603
            split_argv,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=log,
        )
    if split_process.returncode != 0:
        raise subprocess.CalledProcessError(
            split_process.returncode,
            split_argv,
        )

    raw_shards = sorted(
        path
        for path in raw_shard_dir.iterdir()
        if path.is_file() and not path.is_symlink()
    )
    if len(raw_shards) != spec.shard_count:
        raise ValueError(
            f"Expected {spec.shard_count} raw shards, found {len(raw_shards)}"
        )

    shuffled_path.unlink()
    shard_dir.mkdir(parents=True, exist_ok=True)
    for index, raw_shard in enumerate(raw_shards):
        if raw_shard.stat().st_size <= 0:
            raise ValueError(f"SeqKit produced empty shard: {raw_shard}")
        final_shard = shard_dir / _production_shard_filename(spec, index)
        raw_shard.replace(final_shard)
        _require_regular_file(final_shard)
    raw_shard_dir.rmdir()
    return recovery_metrics | {"shuffler": shuffler_metrics}


def _validate_production_profile_statistics(
    spec: DatabaseProfileSpec,
    source_stats_path: Path,
    shard_stats_path: Path,
    shard_summary_path: Path,
) -> dict[str, int | float]:
    """Validate full source/shard statistics for one fixed specification."""
    import polars as pl

    source_stats = pl.read_csv(source_stats_path, separator="\t")
    shard_stats = pl.read_csv(shard_stats_path, separator="\t")
    required_columns = {"file", "num_seqs", "sum_len"}
    if not required_columns.issubset(source_stats.columns):
        raise ValueError(f"Source stats missing columns: {required_columns}")
    if not required_columns.issubset(shard_stats.columns):
        raise ValueError(f"Shard stats missing columns: {required_columns}")
    if source_stats.height != 1:
        raise ValueError(f"Expected one source stats row, got {source_stats.height}")
    if shard_stats.height != spec.shard_count:
        raise ValueError(
            f"Expected {spec.shard_count} shard stats rows, got {shard_stats.height}"
        )

    shard_stats = shard_stats.with_columns(
        pl
        .col("file")
        .cast(pl.String)
        .str.replace_all(r"\\", "/")
        .str.split("/")
        .list.last()
        .alias("basename")
    ).sort("basename")
    if shard_stats.get_column("basename").to_list() != list(
        _production_shard_names(spec)
    ):
        raise ValueError("SeqKit stats shard names do not match the profile")

    source_num_seqs = int(source_stats.item(0, "num_seqs"))
    source_sum_len = int(source_stats.item(0, "sum_len"))
    shard_num_seqs = int(shard_stats.get_column("num_seqs").sum())
    shard_sum_len = int(shard_stats.get_column("sum_len").sum())
    if spec.expected_num_seqs is not None and source_num_seqs != spec.expected_num_seqs:
        raise ValueError(
            f"{spec.database_id} sequence count {source_num_seqs} does not "
            f"match expected {spec.expected_num_seqs}"
        )
    if spec.expected_sum_len is not None and source_sum_len != spec.expected_sum_len:
        raise ValueError(
            f"{spec.database_id} residue count {source_sum_len} does not "
            f"match expected {spec.expected_sum_len}"
        )
    if shard_num_seqs != source_num_seqs:
        raise ValueError(
            f"Shard sequence count {shard_num_seqs} != source {source_num_seqs}"
        )
    if shard_sum_len != source_sum_len:
        raise ValueError(
            f"Shard residue count {shard_sum_len} != source {source_sum_len}"
        )

    mean_sum_len = source_sum_len / spec.shard_count
    maximum_imbalance = max(
        abs(int(value) - mean_sum_len) / mean_sum_len
        for value in shard_stats.get_column("sum_len")
    )
    if maximum_imbalance > MAX_PROFILE_IMBALANCE:
        raise ValueError(
            "Shard residue imbalance exceeds "
            f"{MAX_PROFILE_IMBALANCE:.0%}: {maximum_imbalance:.3%}"
        )

    shard_summary_path.parent.mkdir(parents=True, exist_ok=True)
    shard_stats.write_parquet(shard_summary_path)
    return {
        "num_seqs": source_num_seqs,
        "sum_len": source_sum_len,
        "maximum_residue_imbalance": maximum_imbalance,
    }


def _validate_production_profile_recipe(
    recipe: dict[str, Any],
    spec: DatabaseProfileSpec,
) -> tuple[int, tuple[str, ...]]:
    """Validate one supported immutable sharding recipe."""
    if recipe.get("seqkit_version") != SEQKIT_VERSION:
        raise ValueError("Unexpected production profile SeqKit version")
    if recipe.get("random_seed") != SHARD_RANDOM_SEED:
        raise ValueError("Unexpected production profile shuffle seed")
    if recipe.get("split") != ["--by-part", spec.shard_count]:
        raise ValueError("Unexpected production profile split recipe")
    raw_seqkit_threads = recipe.get("seqkit_threads")
    if isinstance(raw_seqkit_threads, bool) or not isinstance(
        raw_seqkit_threads,
        int,
    ):
        raise ValueError("Invalid production profile SeqKit threads")
    try:
        seqkit_threads = _validate_seqkit_threads(raw_seqkit_threads)
    except ValueError as exc:
        raise ValueError("Invalid production profile SeqKit threads") from exc

    recipe_version = recipe.get("version")
    if recipe_version == LEGACY_PRODUCTION_PROFILE_RECIPE_VERSION:
        if recipe.get("shuffle") != [
            "--two-pass",
            "--update-faidx",
            "--tmp-dir=/tmp",
        ]:
            raise ValueError("Unexpected legacy production shuffle recipe")
        if recipe.get("duplicate_recovery") != {
            "warning_source": "seqkit-fai-sequence-byte-offset",
            "temporary_header_identity": "generation-unique-uuid",
            "append_after_shuffle": True,
            "strip_after_split": True,
        }:
            raise ValueError("Unexpected legacy duplicate-recovery recipe")
        return recipe_version, LEGACY_PRODUCTION_VALIDATION_RELPATHS

    if recipe_version not in {
        ORDINAL_SHUFFLER_RECIPE_VERSION,
        COMPOSABLE_MULTISET_RECIPE_VERSION,
    }:
        raise ValueError("Unexpected production profile recipe version")
    if recipe.get("shuffle") != [
        "two-pass",
        "first-pass-stage-local-source",
        "source-occurrence-offset-index",
        "splitmix64-fisher-yates-u32",
        "bounded-concurrent-local-pread",
        "ordered-write",
    ]:
        raise ValueError("Unexpected occurrence-indexed shuffle recipe")
    if recipe.get("shuffler") != {
        "version": ORDINAL_SHUFFLER_VERSION,
        "source_code_sha256": ORDINAL_SHUFFLER_SOURCE_SHA256,
        "record_identity": "source-occurrence",
        "offset_index": "uint64-source-occurrence-offsets-v1",
        "permutation": "splitmix64-fisher-yates-u32-v1",
        "staging": "first-pass-tee-to-container-local-v1",
        "read": "bounded-concurrent-local-pread-ordered-write-v2",
        "ordered_output": True,
    }:
        raise ValueError("Unexpected native production shuffler identity")
    if recipe.get("execution") != {
        "worker_threads": seqkit_threads,
        "prefetch_records": ORDINAL_SHUFFLER_PREFETCH_RECORDS,
        "prefetch_bytes": ORDINAL_SHUFFLER_PREFETCH_BYTES,
    }:
        raise ValueError("Unexpected native production shuffler execution plan")
    if recipe.get("duplicate_recovery") != {
        "warning_source": None,
        "record_identity": "source-occurrence",
        "append_after_shuffle": False,
        "strip_after_split": False,
    }:
        raise ValueError("Unexpected occurrence-indexed duplicate policy")
    if recipe_version == ORDINAL_SHUFFLER_RECIPE_VERSION:
        return recipe_version, ORDINAL_PRODUCTION_VALIDATION_RELPATHS
    if recipe.get("record_multiset") != (
        _record_multiset_identity() | {"shard_threads": seqkit_threads}
    ):
        raise ValueError("Unexpected composable record-multiset validator")
    return recipe_version, PRODUCTION_VALIDATION_RELPATHS


def _validate_production_profile_manifest(
    manifest: dict[str, Any],
    spec: DatabaseProfileSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate one production-candidate manifest without filesystem access."""
    if manifest.get("schema_version") != PRODUCTION_PROFILE_SCHEMA_VERSION:
        raise ValueError("Unexpected production profile schema version")
    if manifest.get("profile_id") != spec.profile_id:
        raise ValueError("Unexpected production profile ID")
    if manifest.get("database_id") != spec.database_id:
        raise ValueError("Unexpected production database ID")
    if manifest.get("polymer") != spec.polymer:
        raise ValueError("Unexpected production profile polymer")
    if manifest.get("shard_count") != spec.shard_count:
        raise ValueError("Unexpected production profile shard count")
    if manifest.get("shard_prefix") != f"shards/{spec.source_filename}":
        raise ValueError("Unexpected production profile shard prefix")
    if manifest.get("search_space_value") != spec.search_space_value:
        raise ValueError("Unexpected production profile search-space value")
    if manifest.get("search_space_unit") != spec.search_space_unit:
        raise ValueError("Unexpected production profile search-space unit")

    source = manifest.get("source")
    shards = manifest.get("shards")
    validation = manifest.get("validation")
    recipe = manifest.get("recipe")
    compatibility = manifest.get("compatibility")
    if not isinstance(source, dict):
        raise ValueError("Production profile source must be an object")
    if source.get("volume") != SOURCE_DB_VOLUME_NAME:
        raise ValueError("Production profile source Volume is invalid")
    if source.get("path") != spec.source_filename:
        raise ValueError("Production profile source path is invalid")
    if not isinstance(source.get("size_bytes"), int) or source["size_bytes"] <= 0:
        raise ValueError("Production profile source size is invalid")
    if not isinstance(source.get("sha256"), str) or len(source["sha256"]) != 64:
        raise ValueError("Production profile source SHA-256 is invalid")
    if (
        not isinstance(source.get("num_seqs"), int)
        or source["num_seqs"] <= 0
        or not isinstance(source.get("sum_len"), int)
        or source["sum_len"] <= 0
    ):
        raise ValueError("Production profile source statistics are invalid")
    if (
        spec.expected_num_seqs is not None
        and source["num_seqs"] != spec.expected_num_seqs
    ):
        raise ValueError("Production profile source sequence count is invalid")
    if spec.expected_sum_len is not None and source["sum_len"] != spec.expected_sum_len:
        raise ValueError("Production profile source residue count is invalid")

    if not isinstance(shards, list) or len(shards) != spec.shard_count:
        raise ValueError(f"Production profile must declare {spec.shard_count} shards")
    if not isinstance(recipe, dict):
        raise ValueError("Production profile recipe must be an object")
    if compatibility != {
        "alphafold_repository": CONF.repo_url,
        "alphafold_commit": CONF.repo_commit_hash,
        "hmmer_version": HMMER_VERSION,
        "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
    }:
        raise ValueError("Unexpected production profile compatibility pin")
    recipe_version, expected_validation_relpaths = _validate_production_profile_recipe(
        recipe, spec
    )

    if not isinstance(validation, dict) or validation.get("passed") is not True:
        raise ValueError("Production profile does not declare passed validation")
    if validation.get("temporary_recovery_prefix_absent") is not True:
        raise ValueError("Production profile may retain recovery prefixes")
    if validation.get("num_seqs") != source["num_seqs"]:
        raise ValueError("Production profile validation sequence count is invalid")
    if validation.get("sum_len") != source["sum_len"]:
        raise ValueError("Production profile validation residue count is invalid")
    if recipe_version in {
        ORDINAL_SHUFFLER_RECIPE_VERSION,
        COMPOSABLE_MULTISET_RECIPE_VERSION,
    }:
        if validation.get("record_occurrences_preserved") is not True:
            raise ValueError("Production profile does not preserve record occurrences")
        if (
            validation.get("recovered_records") != 0
            or validation.get("recovered_residues") != 0
            or validation.get("first_recovered_byte_offset") is not None
            or validation.get("last_recovered_byte_offset") is not None
        ):
            raise ValueError("Occurrence-indexed profile declares FAI recovery")
    if recipe_version == COMPOSABLE_MULTISET_RECIPE_VERSION:
        if validation.get("canonical_record_multiset_match") is not True:
            raise ValueError("Canonical source and shard record multisets differ")
        signature_sha256 = validation.get("record_multiset_signature_sha256")
        if (
            not isinstance(signature_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", signature_sha256) is None
        ):
            raise ValueError("Invalid canonical record-multiset signature")
        if "seqkit_sum" in validation:
            raise ValueError(
                "Composable-multiset profile unexpectedly declares SeqKit sum"
            )
    validation_artifacts = validation.get("artifacts")
    if not isinstance(validation_artifacts, list):
        raise ValueError("Production validation artifacts must be a list")

    expected_shard_paths = [f"shards/{name}" for name in _production_shard_names(spec)]
    actual_shard_paths: list[str] = []
    for record in [*shards, *validation_artifacts]:
        if not isinstance(record, dict):
            raise ValueError("Production artifact record must be an object")
        relative = record.get("path")
        size_bytes = record.get("size_bytes")
        digest = record.get("sha256")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Production artifact path must be relative")
        if ".." in PurePosixPath(relative).parts:
            raise ValueError(f"Production artifact escapes root: {relative}")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
        ):
            raise ValueError(f"Production artifact is empty: {relative}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid production artifact digest: {relative}")
        if relative.startswith("shards/"):
            actual_shard_paths.append(relative)
    if actual_shard_paths != expected_shard_paths:
        raise ValueError("Production profile shard order or names are invalid")
    if [str(record["path"]) for record in validation_artifacts] != list(
        expected_validation_relpaths
    ):
        raise ValueError("Production validation artifact paths are invalid")
    return source, shards, validation_artifacts


def _validate_published_production_profile(
    profile_root: Path,
    spec: DatabaseProfileSpec,
    *,
    verify_digests: bool,
) -> dict[str, Any]:
    """Deeply validate a manifest-last production-candidate publication."""
    manifest_path = profile_root / "manifest.json"
    _require_regular_file(manifest_path)
    if (profile_root / "source").exists():
        raise ValueError("Production profile must not contain a source copy")
    manifest = _load_json_object(manifest_path)
    _, shards, validation_artifacts = _validate_production_profile_manifest(
        manifest,
        spec,
    )
    resolved_root = profile_root.resolve()
    for record in [*shards, *validation_artifacts]:
        relative = str(record["path"])
        artifact_path = (profile_root / relative).resolve()
        if not artifact_path.is_relative_to(resolved_root):
            raise ValueError(f"Production artifact escapes root: {relative}")
        _require_regular_file(artifact_path)
        if artifact_path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"Production artifact size mismatch: {relative}")
        if verify_digests and _sha256_file(artifact_path) != record["sha256"]:
            raise ValueError(f"Production artifact digest mismatch: {relative}")
    return manifest


def _profile_claim_key(spec: DatabaseProfileSpec) -> str:
    """Return the one mutable election-slot key for a fixed Profile ID."""
    return f"active:{spec.profile_id}"


def _profile_owner_key(spec: DatabaseProfileSpec, generation_id: str) -> str:
    """Return one append-only claim-owner record key."""
    return f"owner:{spec.profile_id}:{generation_id}"


def _profile_status_key(spec: DatabaseProfileSpec, generation_id: str) -> str:
    """Return one append-only claim-terminal record key."""
    return f"status:{spec.profile_id}:{generation_id}"


def _acquire_profile_build_claim(
    spec: DatabaseProfileSpec,
    generation_id: str,
) -> dict[str, object]:
    """Elect one builder, failing immediately on a non-stale conflict."""
    owner = {
        "profile_id": spec.profile_id,
        "database_id": spec.database_id,
        "generation_id": generation_id,
        "container_id": _CONTAINER_INSTANCE_ID,
        "hostname": socket.gethostname(),
        "started_at": _utc_now(),
        "started_at_epoch_seconds": time(),
        "maximum_age_seconds": PRODUCTION_PROFILE_STALE_SECONDS,
    }
    active_key = _profile_claim_key(spec)
    if PROFILE_BUILD_CLAIMS.put(active_key, owner, skip_if_exists=True):
        if not PROFILE_BUILD_CLAIMS.put(
            _profile_owner_key(spec, generation_id),
            owner,
            skip_if_exists=True,
        ):
            PROFILE_BUILD_CLAIMS.pop(active_key, None)
            raise RuntimeError("Profile generation owner key already exists")
        return owner

    active = PROFILE_BUILD_CLAIMS.get(active_key, None)
    if not isinstance(active, dict):
        raise RuntimeError(f"Profile {spec.profile_id} has an invalid active claim")
    started_at = active.get("started_at_epoch_seconds")
    if not isinstance(started_at, int | float):
        raise RuntimeError(f"Profile {spec.profile_id} has an invalid claim time")
    age_seconds = time() - float(started_at)
    if age_seconds <= PRODUCTION_PROFILE_STALE_SECONDS:
        raise RuntimeError(
            f"Profile {spec.profile_id} is already being built by generation "
            f"{active.get('generation_id')!r}"
        )

    removed = PROFILE_BUILD_CLAIMS.pop(active_key, None)
    if removed != active:
        raise RuntimeError(
            f"Profile {spec.profile_id} claim changed during stale recovery"
        )
    stale_generation = active.get("generation_id")
    if isinstance(stale_generation, str):
        PROFILE_BUILD_CLAIMS.put(
            _profile_status_key(spec, stale_generation),
            {
                "status": "abandoned",
                "abandoned_at": _utc_now(),
                "age_seconds": age_seconds,
                "replaced_by_generation_id": generation_id,
            },
            skip_if_exists=True,
        )
    if not PROFILE_BUILD_CLAIMS.put(active_key, owner, skip_if_exists=True):
        raise RuntimeError(f"Profile {spec.profile_id} claim was acquired concurrently")
    if not PROFILE_BUILD_CLAIMS.put(
        _profile_owner_key(spec, generation_id),
        owner,
        skip_if_exists=True,
    ):
        PROFILE_BUILD_CLAIMS.pop(active_key, None)
        raise RuntimeError("Profile generation owner key already exists")
    return owner


def _finish_profile_build_claim(
    spec: DatabaseProfileSpec,
    generation_id: str,
    *,
    status: str,
    detail: dict[str, object],
) -> None:
    """Append terminal claim status and release this generation's slot."""
    PROFILE_BUILD_CLAIMS.put(
        _profile_status_key(spec, generation_id),
        {
            "status": status,
            "finished_at": _utc_now(),
            **detail,
        },
        skip_if_exists=True,
    )
    active_key = _profile_claim_key(spec)
    active = PROFILE_BUILD_CLAIMS.get(active_key, None)
    if isinstance(active, dict) and active.get("generation_id") == generation_id:
        PROFILE_BUILD_CLAIMS.pop(active_key, None)


def _hash_decompressed_zstd(
    archive_path: Path,
    log_path: Path,
) -> tuple[str, int]:
    """Stream one zstd archive through SHA-256 without materializing it."""
    zstd = _require_executable("zstd")
    argv = [zstd, "--quiet", "--decompress", "--stdout", str(archive_path)]
    _append_log(log_path, f"Running command: {shlex.join(argv)}")
    with log_path.open("ab") as log:
        process = subprocess.Popen(  # noqa: S603
            argv,
            stdout=subprocess.PIPE,
            stderr=log,
        )
        if process.stdout is None:
            process.kill()
            raise RuntimeError("zstd did not expose decompressed stdout")
        digest = hashlib.sha256()
        size_bytes = 0
        while chunk := process.stdout.read(8 * 1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
        process.stdout.close()
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, argv)
    return digest.hexdigest(), size_bytes


def _apply_source_policy(
    spec: DatabaseProfileSpec,
    manifest: dict[str, Any],
    source_policy: str,
    log_path: Path,
    *,
    seqkit_threads: int,
) -> dict[str, object]:
    """Retire a source only after a valid profile publication exists."""
    policy = _validate_source_policy(source_policy)
    source_path = Path(APP_INFO.source_db_dir) / spec.source_filename
    archive_path = source_path.with_name(f"{source_path.name}.zst")
    if policy == "keep":
        return {
            "source_policy": policy,
            "source_status": "kept" if source_path.is_file() else "already-retired",
        }

    source_record = manifest.get("source")
    if not isinstance(source_record, dict):
        raise ValueError("Validated manifest lost its source record")
    expected_sha256 = source_record.get("sha256")
    expected_size = source_record.get("size_bytes")
    if not isinstance(expected_sha256, str) or not isinstance(expected_size, int):
        raise ValueError("Validated manifest source identity is invalid")

    if not source_path.is_file():
        if policy == "compress" and archive_path.is_file():
            archive_sha256, archive_size = _hash_decompressed_zstd(
                archive_path,
                log_path,
            )
            if (archive_sha256, archive_size) != (
                expected_sha256,
                expected_size,
            ):
                raise ValueError(
                    f"Existing archive does not reproduce {spec.source_filename}"
                )
            return {
                "source_policy": policy,
                "source_status": "already-compressed",
                "archive_path": str(archive_path),
            }
        if policy == "delete":
            return {
                "source_policy": policy,
                "source_status": "already-deleted",
            }
        raise FileNotFoundError(f"Source FASTA is missing: {source_path}")

    if policy == "delete":
        if (
            source_path.stat().st_size != expected_size
            or _sha256_file(source_path) != expected_sha256
        ):
            raise ValueError(
                f"Refusing to delete changed source {spec.source_filename}"
            )
        source_path.unlink()
        SOURCE_MSA_DB_VOLUME.commit()
        return {
            "source_policy": policy,
            "source_status": "deleted",
        }

    if archive_path.is_file():
        archive_sha256, archive_size = _hash_decompressed_zstd(
            archive_path,
            log_path,
        )
        if (archive_sha256, archive_size) != (expected_sha256, expected_size):
            raise ValueError(
                f"Existing archive does not reproduce {spec.source_filename}"
            )
    else:
        zstd = _require_executable("zstd")
        temporary_archive = archive_path.with_name(
            f".{archive_path.name}.{uuid.uuid4().hex}.tmp"
        )
        argv = [
            zstd,
            f"-T{seqkit_threads}",
            "--quiet",
            "--stdout",
            str(source_path),
        ]
        _append_log(log_path, f"Running command: {shlex.join(argv)}")
        try:
            with temporary_archive.open("xb") as archive, log_path.open("ab") as log:
                completed = subprocess.run(  # noqa: S603
                    argv,
                    check=False,
                    stdout=archive,
                    stderr=log,
                )
                archive.flush()
                os.fsync(archive.fileno())
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, argv)
            archive_sha256, archive_size = _hash_decompressed_zstd(
                temporary_archive,
                log_path,
            )
            if (archive_sha256, archive_size) != (
                expected_sha256,
                expected_size,
            ):
                raise ValueError(
                    f"Compressed archive does not reproduce {spec.source_filename}"
                )
            temporary_archive.replace(archive_path)
            SOURCE_MSA_DB_VOLUME.commit()
        finally:
            temporary_archive.unlink(missing_ok=True)

    source_path.unlink()
    SOURCE_MSA_DB_VOLUME.commit()
    return {
        "source_policy": policy,
        "source_status": "compressed",
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
    }


def _build_production_profile(
    database_id: str,
    seqkit_threads: int,
    source_policy: str,
) -> dict[str, object]:
    """Build, publish, deeply validate, and optionally retire one source."""
    spec = _database_profile_spec(database_id)
    threads = _validate_seqkit_threads(seqkit_threads)
    policy = _validate_source_policy(source_policy)
    generation_id = uuid.uuid4().hex
    source_root = Path(APP_INFO.source_db_dir)
    sharded_root = Path(APP_INFO.sharded_db_dir)
    output_root = Path(APP_INFO.output_dir)
    source_path = source_root / spec.source_filename
    profile_root = _production_profile_root(sharded_root, spec)
    evidence_root = (
        output_root / PRODUCTION_PREPARATION_ROOT / spec.profile_id / generation_id
    )
    log_path = evidence_root / "run.log"
    evidence_root.mkdir(parents=True, exist_ok=True)
    _append_log(log_path, f"Preparing production profile {spec.profile_id}")

    SOURCE_MSA_DB_VOLUME.reload()
    SHARDED_MSA_DB_VOLUME.reload()
    BENCHMARK_OUTPUT_VOLUME.reload()
    if (profile_root / "manifest.json").is_file():
        manifest = _validate_published_production_profile(
            profile_root,
            spec,
            verify_digests=True,
        )
        source_result = _apply_source_policy(
            spec,
            manifest,
            policy,
            log_path,
            seqkit_threads=threads,
        )
        result = {
            "status": "reused",
            "database_id": spec.database_id,
            "profile_id": spec.profile_id,
            "generation_id": generation_id,
            "profile_path": str(profile_root),
            "manifest_sha256": _sha256_file(profile_root / "manifest.json"),
            **source_result,
        }
        _write_json_atomic(evidence_root / "metrics.json", result)
        BENCHMARK_OUTPUT_VOLUME.commit()
        _write_json_atomic(
            evidence_root / "done.json",
            result | {"completed_at": _utc_now()},
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        return result

    claim = _acquire_profile_build_claim(spec, generation_id)
    _write_json_atomic(evidence_root / "claim.json", claim)
    BENCHMARK_OUTPUT_VOLUME.commit()
    staging_root = sharded_root / ".staging" / f"{spec.profile_id}-{generation_id}"
    raw_shard_dir = staging_root / ".raw-shards"
    shard_dir = staging_root / "shards"
    validation_dir = staging_root / "validation"
    payload_moved = False
    manifest_published = False
    try:
        if profile_root.exists():
            orphan_root = sharded_root / ".orphaned"
            orphan_root.mkdir(parents=True, exist_ok=True)
            profile_root.replace(orphan_root / f"{spec.profile_id}-{uuid.uuid4().hex}")
            SHARDED_MSA_DB_VOLUME.commit()

        if not source_path.is_file():
            archive_path = source_path.with_name(f"{source_path.name}.zst")
            if archive_path.is_file():
                raise FileNotFoundError(
                    f"{source_path} is archived as {archive_path}. Restore the "
                    "plain FASTA manually in a Modal Sandbox before rebuilding."
                )
            raise FileNotFoundError(f"Source FASTA is missing: {source_path}")
        _require_regular_file(source_path)
        source_size = source_path.stat().st_size

        shard_dir.mkdir(parents=True)
        validation_dir.mkdir(parents=True)
        seqkit = _require_executable("seqkit")
        version_output = subprocess.run(  # noqa: S603
            [seqkit, "version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if SEQKIT_VERSION not in version_output:
            raise RuntimeError(
                f"Expected SeqKit {SEQKIT_VERSION}, observed {version_output!r}"
            )
        _append_log(log_path, f"Using {version_output}")

        source_stats_path = validation_dir / "source-stats.tsv"
        shard_stats_path = validation_dir / "shard-stats.tsv"
        shard_summary_path = validation_dir / "shard-summary.parquet"
        record_multiset_path = validation_dir / "record-multiset.json"
        _run_to_file(
            [
                seqkit,
                "stats",
                "-j",
                str(threads),
                "--all",
                "--tabular",
                str(source_path),
            ],
            source_stats_path,
            log_path,
        )

        import polars as pl

        source_stats = pl.read_csv(source_stats_path, separator="\t")
        if source_stats.height != 1:
            raise ValueError(
                f"Expected one source stats row, got {source_stats.height}"
            )
        source_num_seqs = int(source_stats.item(0, "num_seqs"))
        source_sum_len = int(source_stats.item(0, "sum_len"))
        if (
            spec.expected_num_seqs is not None
            and source_num_seqs != spec.expected_num_seqs
        ):
            raise ValueError(
                f"{spec.database_id} sequence count {source_num_seqs} does not "
                f"match expected {spec.expected_num_seqs}"
            )
        if (
            spec.expected_sum_len is not None
            and source_sum_len != spec.expected_sum_len
        ):
            raise ValueError(
                f"{spec.database_id} residue count {source_sum_len} does not "
                f"match expected {spec.expected_sum_len}"
            )
        scratch_free = shutil.disk_usage(PRODUCTION_SCRATCH_ROOT).free
        scratch_required = _required_ordinal_shuffler_scratch_bytes(
            source_size,
            source_num_seqs,
        )
        if scratch_free < scratch_required:
            raise OSError(
                f"Insufficient /tmp space for {spec.database_id}: need at least "
                f"{scratch_required} bytes, found {scratch_free}"
            )
        _append_log(
            log_path,
            f"Reserved scratch budget {scratch_required} bytes for local source "
            "staging, shuffled FASTA, and occurrence index",
        )
        source_sha256 = _sha256_file(source_path)

        with tempfile.TemporaryDirectory(
            prefix=f"af3-{spec.database_id}-",
            dir=PRODUCTION_SCRATCH_ROOT,
        ) as scratch_dir:
            scratch_root = Path(scratch_dir)
            recovery_metrics = _run_production_shuffle_split(
                spec,
                source_path,
                scratch_root,
                raw_shard_dir,
                shard_dir,
                validation_dir,
                log_path,
                expected_records=source_num_seqs,
                seqkit_threads=threads,
            )
            shard_paths = tuple(
                shard_dir / name for name in _production_shard_names(spec)
            )
            record_multiset = _run_record_multiset_validation(
                source_path,
                shard_paths,
                scratch_root,
                record_multiset_path,
                log_path,
                threads=threads,
            )

        _run_to_file(
            [
                seqkit,
                "stats",
                "-j",
                str(threads),
                "--all",
                "--tabular",
                *(str(path) for path in shard_paths),
            ],
            shard_stats_path,
            log_path,
        )
        statistics = _validate_production_profile_statistics(
            spec,
            source_stats_path,
            shard_stats_path,
            shard_summary_path,
        )
        multiset_signature = record_multiset.get("signature")
        if not isinstance(multiset_signature, dict):
            raise TypeError("Record-multiset validation lost its signature")
        if multiset_signature.get("records") != statistics["num_seqs"]:
            raise ValueError("Record-multiset count does not match SeqKit statistics")
        if multiset_signature.get("sequence_bytes") != statistics["sum_len"]:
            raise ValueError(
                "Record-multiset sequence bytes do not match SeqKit statistics"
            )
        multiset_signature_sha256 = record_multiset.get("signature_sha256")
        if (
            not isinstance(multiset_signature_sha256, str)
            or len(multiset_signature_sha256) != 64
        ):
            raise ValueError("Record-multiset signature SHA-256 is invalid")

        if recovery_metrics.get("temporary_namespace") is not None:
            raise RuntimeError("Occurrence shuffling must not create recovery headers")
        shard_records = [
            _artifact_record(shard_path, staging_root) for shard_path in shard_paths
        ]
        validation_records = [
            _artifact_record(staging_root / relative, staging_root)
            for relative in PRODUCTION_VALIDATION_RELPATHS
        ]
        manifest: dict[str, object] = {
            "schema_version": PRODUCTION_PROFILE_SCHEMA_VERSION,
            "profile_id": spec.profile_id,
            "database_id": spec.database_id,
            "polymer": spec.polymer,
            "created_at": _utc_now(),
            "generation_id": generation_id,
            "source": {
                "volume": SOURCE_DB_VOLUME_NAME,
                "path": spec.source_filename,
                "size_bytes": source_size,
                "sha256": source_sha256,
                "num_seqs": statistics["num_seqs"],
                "sum_len": statistics["sum_len"],
            },
            "shard_count": spec.shard_count,
            "shard_prefix": f"shards/{spec.source_filename}",
            "shards": shard_records,
            "search_space_value": (
                statistics["num_seqs"]
                if spec.polymer == "protein"
                else statistics["sum_len"] / 1_000_000
            ),
            "search_space_unit": spec.search_space_unit,
            "compatibility": {
                "alphafold_repository": CONF.repo_url,
                "alphafold_commit": CONF.repo_commit_hash,
                "hmmer_version": HMMER_VERSION,
                "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
            },
            "recipe": {
                "version": COMPOSABLE_MULTISET_RECIPE_VERSION,
                "seqkit_version": SEQKIT_VERSION,
                "seqkit_threads": threads,
                "random_seed": SHARD_RANDOM_SEED,
                "shuffle": [
                    "two-pass",
                    "first-pass-stage-local-source",
                    "source-occurrence-offset-index",
                    "splitmix64-fisher-yates-u32",
                    "bounded-concurrent-local-pread",
                    "ordered-write",
                ],
                "shuffler": {
                    "version": ORDINAL_SHUFFLER_VERSION,
                    "source_code_sha256": ORDINAL_SHUFFLER_SOURCE_SHA256,
                    "record_identity": "source-occurrence",
                    "offset_index": "uint64-source-occurrence-offsets-v1",
                    "permutation": "splitmix64-fisher-yates-u32-v1",
                    "staging": "first-pass-tee-to-container-local-v1",
                    "read": "bounded-concurrent-local-pread-ordered-write-v2",
                    "ordered_output": True,
                },
                "execution": {
                    "worker_threads": threads,
                    "prefetch_records": ORDINAL_SHUFFLER_PREFETCH_RECORDS,
                    "prefetch_bytes": ORDINAL_SHUFFLER_PREFETCH_BYTES,
                },
                "duplicate_recovery": {
                    "warning_source": None,
                    "record_identity": "source-occurrence",
                    "append_after_shuffle": False,
                    "strip_after_split": False,
                },
                "record_multiset": _record_multiset_identity()
                | {"shard_threads": threads},
                "split": ["--by-part", spec.shard_count],
            },
            "validation": {
                "passed": True,
                "num_seqs": statistics["num_seqs"],
                "sum_len": statistics["sum_len"],
                "maximum_residue_imbalance": (statistics["maximum_residue_imbalance"]),
                "maximum_allowed_residue_imbalance": MAX_PROFILE_IMBALANCE,
                "recovered_records": recovery_metrics["recovered_records"],
                "recovered_residues": recovery_metrics["recovered_residues"],
                "first_recovered_byte_offset": recovery_metrics["first_byte_offset"],
                "last_recovered_byte_offset": recovery_metrics["last_byte_offset"],
                "temporary_recovery_prefix_absent": True,
                "record_occurrences_preserved": True,
                "canonical_record_multiset_match": True,
                "record_multiset_signature_sha256": (multiset_signature_sha256),
                "artifacts": validation_records,
            },
        }
        _validate_production_profile_manifest(
            manifest,
            spec,
        )

        SHARDED_MSA_DB_VOLUME.commit()
        profile_root.parent.mkdir(parents=True, exist_ok=True)
        staging_root.replace(profile_root)
        payload_moved = True
        SHARDED_MSA_DB_VOLUME.commit()
        _write_json_atomic(profile_root / "manifest.json", manifest)
        SHARDED_MSA_DB_VOLUME.commit()
        manifest_published = True
        published_manifest = _validate_published_production_profile(
            profile_root,
            spec,
            verify_digests=True,
        )
        source_result = _apply_source_policy(
            spec,
            published_manifest,
            policy,
            log_path,
            seqkit_threads=threads,
        )
        result = {
            "status": "published",
            "database_id": spec.database_id,
            "profile_id": spec.profile_id,
            "generation_id": generation_id,
            "profile_path": str(profile_root),
            "manifest_sha256": _sha256_file(profile_root / "manifest.json"),
            "source_size_bytes": source_size,
            "source_sha256": source_sha256,
            "num_seqs": statistics["num_seqs"],
            "sum_len": statistics["sum_len"],
            "search_space_value": manifest["search_space_value"],
            "search_space_unit": spec.search_space_unit,
            "maximum_residue_imbalance": (statistics["maximum_residue_imbalance"]),
            "recovered_records": recovery_metrics["recovered_records"],
            "recovered_residues": recovery_metrics["recovered_residues"],
            **source_result,
        }
        _write_json_atomic(evidence_root / "metrics.json", result)
        BENCHMARK_OUTPUT_VOLUME.commit()
        _write_json_atomic(
            evidence_root / "done.json",
            result | {"completed_at": _utc_now()},
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        _finish_profile_build_claim(
            spec,
            generation_id,
            status="complete",
            detail={"manifest_sha256": result["manifest_sha256"]},
        )
        return result
    except Exception as exc:
        if not manifest_published:
            if payload_moved and profile_root.exists():
                shutil.rmtree(profile_root)
            elif staging_root.exists():
                shutil.rmtree(staging_root)
            SHARDED_MSA_DB_VOLUME.commit()
        failure = {
            "failed_at": _utc_now(),
            "database_id": spec.database_id,
            "profile_id": spec.profile_id,
            "generation_id": generation_id,
            "profile_published": manifest_published,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json_atomic(evidence_root / "failure.json", failure)
        BENCHMARK_OUTPUT_VOLUME.commit()
        _finish_profile_build_claim(
            spec,
            generation_id,
            status="failed",
            detail={
                "error_type": type(exc).__name__,
                "profile_published": manifest_published,
            },
        )
        raise


@app.function(
    cpu=(0.125, 32.125),
    memory=PRODUCTION_BUILD_MEMORY_MIB,
    timeout=PRODUCTION_BUILD_TIMEOUT_SECONDS,
    max_containers=len(DATABASE_PROFILE_SPECS),
    volumes={
        APP_INFO.source_db_dir: SOURCE_MSA_DB_VOLUME,
        APP_INFO.sharded_db_dir: SHARDED_MSA_DB_VOLUME,
        APP_INFO.output_dir: BENCHMARK_OUTPUT_VOLUME,
    },
)
def build_sharded_database(
    database_id: str,
    seqkit_threads: int = DEFAULT_SEQKIT_THREADS,
    source_policy: str = "keep",
) -> dict[str, object]:
    """Build one fixed, immutable production-candidate database profile."""
    return _build_production_profile(
        database_id,
        seqkit_threads,
        source_policy,
    )


def _benchmark_published_uniprot_record_multiset(
    seqkit_threads: int,
) -> dict[str, object]:
    """Compare recipe-v5 validation with the measured recipe-v4 UniProt run."""
    threads = _validate_seqkit_threads(seqkit_threads)
    spec = _database_profile_spec("uniprot")
    generation_id = uuid.uuid4().hex
    source_root = Path(APP_INFO.source_db_dir)
    sharded_root = Path(APP_INFO.sharded_db_dir)
    output_root = Path(APP_INFO.output_dir)
    source_path = source_root / spec.source_filename
    profile_root = _production_profile_root(sharded_root, spec)
    evidence_root = (
        output_root / RECORD_MULTISET_BENCHMARK_ROOT / spec.profile_id / generation_id
    )
    log_path = evidence_root / "run.log"
    evidence_root.mkdir(parents=True, exist_ok=True)
    _append_log(
        log_path,
        f"Benchmarking recipe-v5 record validation on {spec.profile_id}",
    )
    try:
        SOURCE_MSA_DB_VOLUME.reload()
        SHARDED_MSA_DB_VOLUME.reload()
        BENCHMARK_OUTPUT_VOLUME.reload()
        manifest = _validate_published_production_profile(
            profile_root,
            spec,
            verify_digests=False,
        )
        manifest_sha256 = _sha256_file(profile_root / "manifest.json")
        source_record = manifest.get("source")
        shard_records = manifest.get("shards")
        if not isinstance(source_record, dict) or not isinstance(shard_records, list):
            raise TypeError("Published UniProt manifest lost its artifact records")
        _require_regular_file(source_path)
        source_size = source_path.stat().st_size
        if source_record.get("size_bytes") != source_size:
            raise ValueError("Published UniProt source size no longer matches")
        shard_paths = tuple(
            profile_root / "shards" / name for name in _production_shard_names(spec)
        )
        declared_shard_bytes = 0
        for record in shard_records:
            if not isinstance(record, dict):
                raise TypeError("Published UniProt shard record is invalid")
            size_bytes = record.get("size_bytes")
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
                raise TypeError("Published UniProt shard size is invalid")
            declared_shard_bytes += size_bytes

        with tempfile.TemporaryDirectory(
            prefix="af3-record-multiset-benchmark-",
            dir=PRODUCTION_SCRATCH_ROOT,
        ) as scratch_dir:
            validation = _run_record_multiset_validation(
                source_path,
                shard_paths,
                Path(scratch_dir),
                evidence_root / "record-multiset.json",
                log_path,
                threads=threads,
            )

        signature = validation.get("signature")
        source_result = validation.get("source")
        shard_result = validation.get("shards")
        if (
            not isinstance(signature, dict)
            or not isinstance(source_result, dict)
            or not isinstance(shard_result, dict)
        ):
            raise TypeError("Record-multiset benchmark lost validator metrics")
        signature = cast(dict[str, Any], signature)
        source_result = cast(dict[str, Any], source_result)
        shard_result = cast(dict[str, Any], shard_result)
        if signature.get("records") != source_record.get("num_seqs"):
            raise ValueError("Validator record count differs from v4 manifest")
        if signature.get("sequence_bytes") != source_record.get("sum_len"):
            raise ValueError("Validator residue count differs from v4 manifest")
        if source_result.get("input_bytes") != source_size:
            raise ValueError("Validator source byte count is invalid")
        if shard_result.get("input_bytes") != declared_shard_bytes:
            raise ValueError("Validator shard byte count is invalid")
        source_seconds = source_result.get("wall_seconds")
        shard_seconds = shard_result.get("wall_seconds")
        if (
            isinstance(source_seconds, bool)
            or not isinstance(source_seconds, int | float)
            or source_seconds <= 0
            or isinstance(shard_seconds, bool)
            or not isinstance(shard_seconds, int | float)
            or shard_seconds <= 0
        ):
            raise ValueError("Validator benchmark timing is invalid")
        combined_seconds = float(source_seconds) + float(shard_seconds)
        baseline = UNIPROT_V4_VALIDATION_BASELINE
        result: dict[str, object] = {
            "schema_version": 1,
            "status": "passed",
            "database_id": spec.database_id,
            "profile_id": spec.profile_id,
            "generation_id": generation_id,
            "profile_manifest_sha256": manifest_sha256,
            "validator": _record_multiset_identity(),
            "threads": threads,
            "source_size_bytes": source_size,
            "shard_size_bytes": declared_shard_bytes,
            "records": signature["records"],
            "sequence_bytes": signature["sequence_bytes"],
            "signature_sha256": validation["signature_sha256"],
            "source_wall_seconds": float(source_seconds),
            "shard_wall_seconds": float(shard_seconds),
            "combined_wall_seconds": combined_seconds,
            "source_bytes_per_second": source_result["throughput_bytes_per_second"],
            "shard_bytes_per_second": shard_result["throughput_bytes_per_second"],
            "baseline": baseline,
            "comparisons": {
                "source_seqkit_sum_over_new_source_ratio": (
                    baseline["source_seqkit_sum_seconds"] / float(source_seconds)
                ),
                "historical_post_stats_window_over_new_combined_ratio": (
                    baseline["post_shard_stats_to_completion_seconds"]
                    / combined_seconds
                ),
            },
            "source_and_shards_mutated": False,
            "evidence_path": str(evidence_root),
            "completed_at": _utc_now(),
        }
        _write_json_atomic(evidence_root / "metrics.json", result)
        BENCHMARK_OUTPUT_VOLUME.commit()
        _write_json_atomic(
            evidence_root / "done.json",
            {
                "schema_version": 1,
                "status": "complete",
                "generation_id": generation_id,
                "signature_sha256": validation["signature_sha256"],
                "completed_at": _utc_now(),
            },
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        return result
    except Exception as exc:
        _append_log(log_path, f"Failed with {type(exc).__name__}: {exc}")
        _write_json_atomic(
            evidence_root / "failure.json",
            {
                "failed_at": _utc_now(),
                "database_id": spec.database_id,
                "profile_id": spec.profile_id,
                "generation_id": generation_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        raise


@app.function(
    cpu=(0.125, 32.125),
    memory=PRODUCTION_BUILD_MEMORY_MIB,
    timeout=CONF.timeout,
    max_containers=1,
    volumes={
        APP_INFO.source_db_dir: SOURCE_MSA_DB_VOLUME.with_mount_options(read_only=True),
        APP_INFO.sharded_db_dir: SHARDED_MSA_DB_VOLUME.with_mount_options(
            read_only=True
        ),
        APP_INFO.output_dir: BENCHMARK_OUTPUT_VOLUME,
    },
)
def benchmark_record_multiset_validator(
    seqkit_threads: int = DEFAULT_SEQKIT_THREADS,
) -> dict[str, object]:
    """Benchmark recipe-v5 validation on the published recipe-v4 UniProt."""
    return _benchmark_published_uniprot_record_multiset(seqkit_threads)


SCAN_BUFFER_SIZE = 8 * 1024 * 1024
DONE_SCHEMA_VERSION = 1
SCAN_PASS_NAMES = ("first-pass", "immediate-repeat")


@dataclass(frozen=True)
class ScanCase:
    """One immutable Volume scan topology."""

    case_id: str
    layout: str
    containers: int
    readers_per_container: int

    def as_dict(self) -> dict[str, str | int]:
        """Return a primitive plan representation."""
        return {
            "case_id": self.case_id,
            "layout": self.layout,
            "containers": self.containers,
            "readers_per_container": self.readers_per_container,
            "aggregate_readers": self.containers * self.readers_per_container,
        }


SCAN_CASES = (
    ScanCase("V0", "monolith", 1, 1),
    ScanCase("V1", "shards", 1, 1),
    ScanCase("V2", "shards", 1, 2),
    ScanCase("V3", "shards", 1, 4),
    ScanCase("V4", "shards", 1, 8),
    ScanCase("V5", "shards", 1, 16),
    ScanCase("V6", "shards", 2, 8),
    ScanCase("V7", "shards", 4, 4),
    ScanCase("V8", "shards", 4, 16),
)


def _scan_case(case_id: str) -> ScanCase:
    """Resolve one fixed case ID, rejecting arbitrary path-like input."""
    for case in SCAN_CASES:
        if case.case_id == case_id:
            return case
    choices = ", ".join(case.case_id for case in SCAN_CASES)
    raise ValueError(f"Unknown scan case {case_id!r}; expected one of {choices}")


def _scan_case_paths(case: ScanCase) -> tuple[str, ...]:
    """Return profile-relative files read by a scan case."""
    if case.layout == "monolith":
        return (f"source/{SOURCE_DB_FILENAME}",)
    if case.layout == "shards":
        return tuple(f"shards/{name}" for name in _shard_names())
    raise ValueError(f"Unsupported scan layout: {case.layout}")


def _scan_partition_paths(case: ScanCase, partition_index: int) -> tuple[str, ...]:
    """Assign a disjoint deterministic subset of files to one container."""
    if isinstance(partition_index, bool) or not isinstance(partition_index, int):
        raise TypeError("partition_index must be an integer")
    if not 0 <= partition_index < case.containers:
        raise ValueError(
            f"partition_index must be in [0, {case.containers}), got {partition_index}"
        )
    return _scan_case_paths(case)[partition_index :: case.containers]


def _profile_artifact_map(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Index validated profile artifact records by relative path."""
    source, shards, _ = _validate_profile_manifest(manifest)
    return {str(record["path"]): record for record in [source, *shards]}


def _sha256_bytes(data: bytes) -> str:
    """Return the SHA-256 digest of a small in-memory artifact."""
    return hashlib.sha256(data).hexdigest()


def _scan_partition_identity(
    manifest_sha256: str,
    manifest: dict[str, Any],
    case: ScanCase,
    partition_index: int,
) -> str:
    """Hash the result-affecting identity of one scan partition."""
    records = _profile_artifact_map(manifest)
    files = [
        {
            "path": path,
            "size_bytes": records[path]["size_bytes"],
            "sha256": records[path]["sha256"],
        }
        for path in _scan_partition_paths(case, partition_index)
    ]
    identity = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": manifest_sha256,
        "case": case.as_dict(),
        "partition_index": partition_index,
        "passes": list(SCAN_PASS_NAMES),
        "buffer_size": SCAN_BUFFER_SIZE,
        "files": files,
    }
    return _sha256_bytes(_json_bytes(identity))


def _scan_partition_relpath(case_id: str, partition_index: int) -> str:
    """Return the evidence path for one fixed scan partition."""
    case = _scan_case(case_id)
    if not 0 <= partition_index < case.containers:
        raise ValueError("partition index outside case topology")
    return (
        f"benchmarks/{CAMPAIGN_ID}/storage-scans/{case.case_id}/"
        f"partition-{partition_index:02d}"
    )


def _validate_done_marker(
    artifact_root: Path,
    *,
    expected_identity: str,
) -> dict[str, Any]:
    """Validate a local mounted completion marker and every small artifact."""
    marker_path = artifact_root / "done.json"
    _require_regular_file(marker_path)
    marker = _load_json_object(marker_path)
    if marker.get("schema_version") != DONE_SCHEMA_VERSION:
        raise ValueError("Unexpected completion marker schema")
    if marker.get("status") != "complete":
        raise ValueError("Completion marker is not complete")
    if marker.get("identity") != expected_identity:
        raise ValueError("Completion marker identity mismatch")
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("Completion marker has no artifacts")
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("Completion artifact record must be an object")
        relative = record.get("path")
        if not isinstance(relative, str):
            raise ValueError("Completion artifact path must be a string")
        relative_path = PurePosixPath(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Completion artifact path escapes its root")
        artifact = (artifact_root / relative).resolve()
        if not artifact.is_relative_to(artifact_root.resolve()):
            raise ValueError("Completion artifact escapes its root")
        _require_regular_file(artifact)
        if artifact.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"Completion artifact size mismatch: {relative}")
        if _sha256_file(artifact) != record.get("sha256"):
            raise ValueError(f"Completion artifact digest mismatch: {relative}")
    return marker


def _scan_one_file(path: Path, relative_path: str) -> dict[str, str | int | float]:
    """Read one complete file and report exact bytes and elapsed time."""
    _require_regular_file(path)
    expected_bytes = path.stat().st_size
    byte_count = 0
    buffer = bytearray(SCAN_BUFFER_SIZE)
    started_at = _utc_now()
    started = perf_counter()
    with path.open("rb", buffering=0) as handle:
        while read_size := handle.readinto(buffer):
            byte_count += read_size
    elapsed = perf_counter() - started
    finished_at = _utc_now()
    if byte_count != expected_bytes:
        raise OSError(
            f"Short scan for {relative_path}: read {byte_count}, expected {expected_bytes}"
        )
    return {
        "path": relative_path,
        "started_at": started_at,
        "finished_at": finished_at,
        "bytes": byte_count,
        "wall_seconds": elapsed,
        "throughput_bytes_per_second": byte_count / elapsed if elapsed else 0.0,
    }


def _scan_files(
    profile_root: Path,
    relative_paths: tuple[str, ...],
    *,
    readers: int,
) -> list[dict[str, object]]:
    """Read an assignment twice with one persistent local thread pool."""
    if readers < 1:
        raise ValueError("readers must be positive")
    absolute_paths = tuple(profile_root / path for path in relative_paths)
    pass_metrics: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=readers) as executor:
        for pass_name in SCAN_PASS_NAMES:
            started = perf_counter()
            files = list(executor.map(_scan_one_file, absolute_paths, relative_paths))
            elapsed = perf_counter() - started
            byte_count = sum(int(file["bytes"]) for file in files)
            pass_metrics.append({
                "pass": pass_name,
                "bytes": byte_count,
                "wall_seconds": elapsed,
                "throughput_bytes_per_second": (
                    byte_count / elapsed if elapsed else 0.0
                ),
                "files": files,
            })
    return pass_metrics


def _container_metadata() -> dict[str, object]:
    """Collect portable container placement and CPU-affinity evidence."""
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    load_average: list[float] | None = None
    if hasattr(os, "getloadavg"):
        load_average = list(os.getloadavg())
    return {
        "hostname": socket.gethostname(),
        "cpu_affinity": affinity,
        "cpu_count": os.cpu_count(),
        "load_average_at_finish": load_average,
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "modal_cloud_provider": os.environ.get("MODAL_CLOUD_PROVIDER"),
        "modal_region": os.environ.get("MODAL_REGION"),
    }


def _container_sample_metadata() -> dict[str, object]:
    """Record whether this interpreter already ran a benchmark sample."""
    global _CONTAINER_SAMPLE_COUNT  # noqa: PLW0603

    with _CONTAINER_SAMPLE_LOCK:
        _CONTAINER_SAMPLE_COUNT += 1
        sample_ordinal = _CONTAINER_SAMPLE_COUNT
    return _container_metadata() | {
        "container_instance_id": _CONTAINER_INSTANCE_ID,
        "container_sample_ordinal": sample_ordinal,
        "container_reused_for_sample": sample_ordinal > 1,
    }


def _run_scan_partition(case_id: str, partition_index: int) -> dict[str, object]:
    """Execute one two-pass Volume scan assignment."""
    case = _scan_case(case_id)
    relative_paths = _scan_partition_paths(case, partition_index)
    profile_root = Path(APP_INFO.sharded_db_dir) / APP_INFO.profile_relpath
    final_output_root = Path(APP_INFO.output_dir) / _scan_partition_relpath(
        case_id, partition_index
    )
    SHARDED_MSA_DB_VOLUME.reload()
    BENCHMARK_OUTPUT_VOLUME.reload()
    manifest = _validate_published_profile(profile_root, verify_digests=False)
    manifest_path = profile_root / "manifest.json"
    manifest_sha256 = _sha256_file(manifest_path)
    identity = _scan_partition_identity(
        manifest_sha256,
        manifest,
        case,
        partition_index,
    )
    try:
        _validate_done_marker(final_output_root, expected_identity=identity)
    except (FileNotFoundError, ValueError):
        pass
    else:
        metrics = _load_json_object(final_output_root / "metrics.json")
        return metrics | {"status": "reused"}

    output_root = (
        final_output_root.parent
        / ".staging"
        / f"partition-{partition_index:02d}-{uuid.uuid4().hex}"
    )
    output_root.mkdir(parents=True)
    log_path = output_root / "run.log"
    _append_log(
        log_path,
        f"Starting {case.case_id} partition {partition_index:02d} "
        f"with {case.readers_per_container} readers",
    )
    started = perf_counter()
    passes = _scan_files(
        profile_root,
        relative_paths,
        readers=case.readers_per_container,
    )
    sample_wall_seconds = perf_counter() - started
    records = _profile_artifact_map(manifest)
    expected_bytes = sum(int(records[path]["size_bytes"]) for path in relative_paths)
    for result in passes:
        observed_bytes = result.get("bytes")
        if isinstance(observed_bytes, bool) or not isinstance(observed_bytes, int):
            raise ValueError("A Volume scan pass has an invalid byte count")
        if observed_bytes != expected_bytes:
            raise ValueError("A Volume scan pass did not read its complete assignment")
    metadata = _container_metadata()
    metrics: dict[str, object] = {
        "status": "published",
        "campaign_id": CAMPAIGN_ID,
        "case_id": case.case_id,
        "layout": case.layout,
        "containers": case.containers,
        "readers_per_container": case.readers_per_container,
        "partition_index": partition_index,
        "partition_count": case.containers,
        "identity": identity,
        "profile_manifest_sha256": manifest_sha256,
        "relative_paths": list(relative_paths),
        "expected_bytes_per_pass": expected_bytes,
        "passes": passes,
        "sample_wall_seconds": sample_wall_seconds,
        "container": metadata,
    }
    _append_log(
        log_path,
        f"Completed {case.case_id} partition {partition_index:02d}",
    )
    metrics_path = output_root / "metrics.json"
    _write_json_atomic(metrics_path, metrics)
    BENCHMARK_OUTPUT_VOLUME.commit()
    marker = {
        "schema_version": DONE_SCHEMA_VERSION,
        "status": "complete",
        "identity": identity,
        "completed_at": _utc_now(),
        "artifacts": [
            _artifact_record(metrics_path, output_root),
            _artifact_record(log_path, output_root),
        ],
    }
    _write_json_atomic(output_root / "done.json", marker)
    BENCHMARK_OUTPUT_VOLUME.commit()
    if final_output_root.exists():
        try:
            _validate_done_marker(final_output_root, expected_identity=identity)
        except (FileNotFoundError, ValueError):
            orphan_root = final_output_root.parent / ".orphaned"
            orphan_root.mkdir(parents=True, exist_ok=True)
            final_output_root.replace(
                orphan_root / f"partition-{partition_index:02d}-{uuid.uuid4().hex}"
            )
        else:
            duplicate_root = final_output_root.parent / ".orphaned"
            duplicate_root.mkdir(parents=True, exist_ok=True)
            output_root.replace(
                duplicate_root
                / f"partition-{partition_index:02d}-duplicate-{uuid.uuid4().hex}"
            )
            BENCHMARK_OUTPUT_VOLUME.commit()
            existing = _load_json_object(final_output_root / "metrics.json")
            return existing | {"status": "reused-concurrent"}
    output_root.replace(final_output_root)
    BENCHMARK_OUTPUT_VOLUME.commit()
    return metrics


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 131_072),
    timeout=CONF.timeout,
    max_containers=4,
    volumes={
        APP_INFO.sharded_db_dir: SHARDED_MSA_DB_VOLUME.with_mount_options(
            read_only=True
        ),
        APP_INFO.output_dir: BENCHMARK_OUTPUT_VOLUME,
    },
)
def scan_volume_partition(case_id: str, partition_index: int) -> dict[str, object]:
    """Read one disjoint part of a fixed Volume scan case twice.

    Args:
        case_id: Fixed scan case ID from V0 through V8.
        partition_index: Zero-based container partition within the case.

    Returns:
        Primitive per-file, per-pass, and container measurements.
    """
    return _run_scan_partition(case_id, partition_index)


def _build_scan_plan() -> dict[str, object]:
    """Build the complete side-effect-free Volume scan plan."""
    function_inputs = sum(case.containers for case in SCAN_CASES)
    return {
        "campaign_id": CAMPAIGN_ID,
        "operation": "scan",
        "cases": [case.as_dict() for case in SCAN_CASES],
        "case_count": len(SCAN_CASES),
        "remote_function_inputs": function_inputs,
        "passes_per_input": list(SCAN_PASS_NAMES),
        "full_dataset_reads_per_case": len(SCAN_PASS_NAMES),
        "execution": "cases-sequential-partitions-concurrent",
        "resources_per_container": {
            "cpu": [0.125, 32.125],
            "memory_mib": [1024, 131_072],
            "timeout_seconds": CONF.timeout,
        },
    }


def _read_volume_bytes(
    volume: modal.Volume,
    relative_path: str,
    *,
    maximum_bytes: int = 32 * 1024 * 1024,
) -> bytes:
    """Read one bounded artifact through Modal's local Volume client."""
    data = bytearray()
    for chunk in volume.read_file(relative_path):
        data.extend(chunk)
        if len(data) > maximum_bytes:
            raise ValueError(f"Volume artifact exceeds byte limit: {relative_path}")
    return bytes(data)


def _read_volume_json(
    volume: modal.Volume,
    relative_path: str,
) -> dict[str, Any]:
    """Read one bounded JSON object through the local Volume client."""
    value = orjson.loads(_read_volume_bytes(volume, relative_path))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in Volume path {relative_path}")
    return value


def _upload_volume_bytes(
    volume: modal.Volume,
    relative_path: str,
    data: bytes,
) -> None:
    """Upload one small complete artifact through the local Volume client."""
    with volume.batch_upload(force=True) as batch:
        batch.put_file(io.BytesIO(data), relative_path)


def _volume_artifact_record(relative_path: str, data: bytes) -> dict[str, object]:
    """Build a marker record for a client-uploaded artifact."""
    return {
        "path": relative_path,
        "size_bytes": len(data),
        "sha256": _sha256_bytes(data),
    }


def _client_done_marker_valid(
    volume: modal.Volume,
    marker_relative_path: str,
    *,
    expected_identity: str,
) -> bool:
    """Validate a completion marker and every artifact through client reads."""
    try:
        marker = _read_volume_json(volume, marker_relative_path)
        if marker.get("schema_version") != DONE_SCHEMA_VERSION:
            return False
        if marker.get("status") != "complete":
            return False
        if marker.get("identity") != expected_identity:
            return False
        artifacts = marker.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return False
        marker_parent = PurePosixPath(marker_relative_path).parent
        for record in artifacts:
            if not isinstance(record, dict):
                return False
            relative = record.get("path")
            if not isinstance(relative, str):
                return False
            path = PurePosixPath(relative)
            if path.is_absolute() or ".." in path.parts:
                return False
            size_bytes = record.get("size_bytes")
            digest = record.get("sha256")
            if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
                return False
            if not isinstance(digest, str) or len(digest) != 64:
                return False
            artifact_path = (marker_parent / path).as_posix().lstrip("/")
            artifact = _read_volume_bytes(volume, artifact_path)
            if len(artifact) != size_bytes:
                return False
            if _sha256_bytes(artifact) != digest:
                return False
    except (FileNotFoundError, ValueError, orjson.JSONDecodeError):
        return False
    return True


def _scan_results_parquet(results: list[dict[str, Any]]) -> bytes:
    """Serialize one flat row per scan partition and pass."""
    import polars as pl

    rows: list[dict[str, object]] = []
    for result in results:
        container = result.get("container")
        if not isinstance(container, dict):
            raise ValueError("Scan result is missing container metadata")
        passes = result.get("passes")
        if not isinstance(passes, list) or len(passes) != len(SCAN_PASS_NAMES):
            raise ValueError("Scan result has an invalid pass set")
        for pass_result in passes:
            if not isinstance(pass_result, dict):
                raise ValueError("Scan pass result must be an object")
            rows.append({
                "campaign_id": CAMPAIGN_ID,
                "sample_kind": "storage-scan",
                "case_id": result["case_id"],
                "layout": result["layout"],
                "partition_index": result["partition_index"],
                "partition_count": result["partition_count"],
                "readers_per_container": result["readers_per_container"],
                "pass": pass_result["pass"],
                "bytes": pass_result["bytes"],
                "wall_seconds": pass_result["wall_seconds"],
                "throughput_bytes_per_second": pass_result[
                    "throughput_bytes_per_second"
                ],
                "sample_wall_seconds": result["sample_wall_seconds"],
                "remote_call_wall_seconds": result.get("remote_call_wall_seconds"),
                "container_hostname": container.get("hostname"),
                "result_path": result["result_path"],
            })
    buffer = io.BytesIO()
    pl.DataFrame(rows).sort(["case_id", "partition_index", "pass"]).write_parquet(
        buffer
    )
    return buffer.getvalue()


def _scan_case_pass_summaries(
    results: list[dict[str, Any]],
) -> list[dict[str, object]]:
    """Aggregate concurrent partition throughput for each scan pass."""
    summaries: list[dict[str, object]] = []
    for pass_name in SCAN_PASS_NAMES:
        partition_passes: list[dict[str, Any]] = []
        for result in results:
            passes = result.get("passes")
            if not isinstance(passes, list):
                raise ValueError("Scan result has no pass list")
            matches = [
                item
                for item in passes
                if isinstance(item, dict) and item.get("pass") == pass_name
            ]
            if len(matches) != 1:
                raise ValueError(f"Scan result has invalid {pass_name} evidence")
            partition_passes.append(matches[0])
        byte_count = sum(int(item["bytes"]) for item in partition_passes)
        critical_wall_seconds = max(
            float(item["wall_seconds"]) for item in partition_passes
        )
        summaries.append({
            "pass": pass_name,
            "bytes": byte_count,
            "critical_wall_seconds": critical_wall_seconds,
            "aggregate_throughput_bytes_per_second": (
                byte_count / critical_wall_seconds if critical_wall_seconds else 0.0
            ),
        })
    return summaries


def _scan_operation_identity(manifest_sha256: str) -> str:
    """Hash the immutable campaign scan plan."""
    return _sha256_bytes(
        _json_bytes({
            "campaign_id": CAMPAIGN_ID,
            "profile_manifest_sha256": manifest_sha256,
            "scan_plan": _build_scan_plan(),
        })
    )


def _submit_scan_matrix() -> dict[str, object]:
    """Submit missing scan partitions case by case and publish their index."""
    _ensure_campaign_plan_client()
    profile_manifest_relpath = f"{APP_INFO.profile_relpath}/manifest.json"
    manifest_bytes = _read_volume_bytes(
        SHARDED_MSA_DB_VOLUME,
        profile_manifest_relpath,
    )
    manifest = orjson.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Profile manifest must be a JSON object")
    _validate_profile_manifest(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    storage_root = f"benchmarks/{CAMPAIGN_ID}/storage-scans"
    operation_identity = _scan_operation_identity(manifest_sha256)
    operation_marker_path = f"{storage_root}/done.json"
    operation_complete = _client_done_marker_valid(
        BENCHMARK_OUTPUT_VOLUME,
        operation_marker_path,
        expected_identity=operation_identity,
    )
    partitions_complete = all(
        _client_done_marker_valid(
            BENCHMARK_OUTPUT_VOLUME,
            f"{_scan_partition_relpath(case.case_id, partition_index)}/done.json",
            expected_identity=_scan_partition_identity(
                manifest_sha256,
                manifest,
                case,
                partition_index,
            ),
        )
        for case in SCAN_CASES
        for partition_index in range(case.containers)
    )
    if operation_complete and partitions_complete:
        _publish_campaign_progress(
            stage="storage scan",
            status="scan complete; search benchmarks pending",
            details=[
                f"The complete V0-V8 scan index is available under `{storage_root}/`.",
                "No search benchmark result is included yet.",
            ],
        )
        return {
            "status": "reused",
            "operation": "scan",
            "remote_function_inputs_submitted": 0,
            "results_path": f"{storage_root}/results.parquet",
        }

    all_results: list[dict[str, Any]] = []
    case_artifacts: list[dict[str, object]] = []
    submitted_inputs = 0
    for case in SCAN_CASES:
        expected_identities = {
            partition_index: _scan_partition_identity(
                manifest_sha256,
                manifest,
                case,
                partition_index,
            )
            for partition_index in range(case.containers)
        }
        missing_partitions = [
            partition_index
            for partition_index, identity in expected_identities.items()
            if not _client_done_marker_valid(
                BENCHMARK_OUTPUT_VOLUME,
                f"{_scan_partition_relpath(case.case_id, partition_index)}/done.json",
                expected_identity=identity,
            )
        ]
        submitted_inputs += len(missing_partitions)
        started = perf_counter()
        new_results: list[dict[str, Any]] = []
        if len(missing_partitions) == 1:
            partition_index = missing_partitions[0]
            new_results.append(
                scan_volume_partition.remote(
                    case_id=case.case_id,
                    partition_index=partition_index,
                )
            )
        elif missing_partitions:
            inputs = [
                (case.case_id, partition_index)
                for partition_index in missing_partitions
            ]
            new_results.extend(scan_volume_partition.starmap(inputs))
        remote_call_wall_seconds = perf_counter() - started

        results_by_partition = {
            int(result["partition_index"]): result for result in new_results
        }
        for partition_index in range(case.containers):
            if partition_index not in results_by_partition:
                result_path = (
                    f"{_scan_partition_relpath(case.case_id, partition_index)}/"
                    "metrics.json"
                )
                results_by_partition[partition_index] = _read_volume_json(
                    BENCHMARK_OUTPUT_VOLUME,
                    result_path,
                )
            result = results_by_partition[partition_index]
            result["remote_call_wall_seconds"] = remote_call_wall_seconds
            result["result_path"] = (
                f"{_scan_partition_relpath(case.case_id, partition_index)}/metrics.json"
            )
            all_results.append(result)

        if len(missing_partitions) == case.containers and case.containers > 1:
            task_ids = {
                result["container"].get("modal_task_id")
                for result in results_by_partition.values()
            }
            if (
                not all(isinstance(task_id, str) and task_id for task_id in task_ids)
                or len(task_ids) != case.containers
            ):
                raise RuntimeError(
                    f"{case.case_id} used {len(task_ids)} Modal task IDs, "
                    f"expected {case.containers}"
                )

        case_summary = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "case": case.as_dict(),
            "profile_manifest_sha256": manifest_sha256,
            "partition_identities": {
                str(partition_index): identity
                for partition_index, identity in expected_identities.items()
            },
            "pass_summaries": _scan_case_pass_summaries(
                list(results_by_partition.values())
            ),
            "remote_call_wall_seconds": remote_call_wall_seconds,
            "submitted_partitions": missing_partitions,
            "completed_at": _utc_now(),
        }
        case_summary_bytes = _json_bytes(case_summary)
        case_summary_path = f"{case.case_id}/case-summary.json"
        _upload_volume_bytes(
            BENCHMARK_OUTPUT_VOLUME,
            f"{storage_root}/{case_summary_path}",
            case_summary_bytes,
        )
        case_artifacts.append(
            _volume_artifact_record(case_summary_path, case_summary_bytes)
        )

    results_bytes = _scan_results_parquet(all_results)
    results_relative_path = "results.parquet"
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{storage_root}/{results_relative_path}",
        results_bytes,
    )
    operation_marker = {
        "schema_version": DONE_SCHEMA_VERSION,
        "status": "complete",
        "identity": operation_identity,
        "completed_at": _utc_now(),
        "artifacts": [
            *case_artifacts,
            _volume_artifact_record(results_relative_path, results_bytes),
        ],
    }
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        operation_marker_path,
        _json_bytes(operation_marker),
    )
    _publish_campaign_progress(
        stage="storage scan",
        status="scan complete; search benchmarks pending",
        details=[
            f"Completed all {len(SCAN_CASES)} V0-V8 scan cases.",
            f"Submitted {submitted_inputs} remote partition inputs.",
        ],
    )
    return {
        "status": "published",
        "operation": "scan",
        "remote_function_inputs_submitted": submitted_inputs,
        "case_count": len(SCAN_CASES),
        "result_rows": len(all_results) * len(SCAN_PASS_NAMES),
        "results_path": f"{storage_root}/{results_relative_path}",
    }


PEMBROLIZUMAB_VH_SEQUENCE = (
    "QVQLVQSGVEVKKPGASVKVSCKASGYTFTNYYMYWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRV"
    "TLTTDSSTTTAYMELKSLQFDDTAVYYCARRDYRFDMGFDYWGQGTTVTVSS"
)
PEMBROLIZUMAB_VH_SHA256 = (
    "5d92fab232244fa55131fc3b8d31b34990aa778623cdd906d58cf920dbdaf28f"
)
RNA_ORACLE_SEQUENCE = "GGCCCGAUAGCUCAGUCGGUAGAGC"
ECOLI_K12_GROEL_SEQUENCE = (
    "MAAKDVKFGNDARVKMLRGVNVLADAVKVTLGPKGRNVVLDKSFGAPTITKDGVSVAREIELEDKFENMG"
    "AQMVKEVASKANDAAGDGTTTATVLAQAIITEGLKAVAAGMNPMDLKRGIDKAVTAAVEELKALSVPCSD"
    "SKAIAQVGTISANSDETVGKLIAEAMDKVGKEGVITVEDGTGLQDELDVVEGMQFDRGYLSPYFINKPET"
    "GAVELESPFILLADKKISNIREMLPVLEAVAKAGKPLLIIAEDVEGEALATLVVNTMRGIVKVAAVKAPG"
    "FGDRRKAMLQDIATLTGGTVISEEIGMELEKATLEDLGQAKRVVINKDTTTIIDGVGEEAAIQGRVAQIR"
    "QQIEEATSDYDREKLQERVAKLAGGVAVIKVGAATEVEMKEKKARVEDALHATRAAVEEGVVAGGGVALI"
    "RVASKLADLRGQNEDQNVGIKVALRAMEAPLRQIVLNCGEEPSVVANTVKGGDGNYGYNAATEEYGNMID"
    "MGILDPTKVTRSALQYAASVAGLMITTECMVTDLPKNDAADLGAAGGMGGMGGMGGMM"
)
ECOLI_K12_GROEL_SHA256 = (
    "40544c6fee0f15b6fe78d6ab7e5e27d8080224fe28dc0d6ca6f2e9a790dd24d4"
)
SMOKE_CASE_IDS = ("B0", "B1", "S3")
MATRIX_CASE_IDS = ("B0", "B1", "S0", "S1", "S2", "S3", "S4", "S5")
FOCUSED_SWEEP_REUSED_CASE_IDS = ("B1", "S3")
FOCUSED_SWEEP_NEW_CASE_IDS = ("S1", "S2", "S4", "S5", "S6")
FOCUSED_SWEEP_CASE_IDS = ("B1", "S1", "S2", "S3", "S4", "S5", "S6")
SCREENING_BLOCK_ORDERS = (
    ("B1", "S3", "S0", "B0", "S5", "S2", "S4", "S1"),
    ("S3", "S0", "B1", "S5", "S2", "B0", "S4", "S1"),
    ("S1", "S4", "B0", "S2", "S5", "B1", "S3", "S0"),
)


@dataclass(frozen=True)
class SearchQuery:
    """One immutable benchmark query sequence."""

    query_id: str
    role: str
    sequence: str
    sequence_sha256: str

    def as_dict(self) -> dict[str, str | int]:
        """Return query provenance without duplicating the full sequence."""
        return {
            "query_id": self.query_id,
            "role": self.role,
            "length": len(self.sequence),
            "sequence_sha256": self.sequence_sha256,
        }


@dataclass(frozen=True)
class SearchCase:
    """One scientific layout and operational Jackhmmer topology."""

    case_id: str
    layout: str
    jackhmmer_n_cpu: int
    active_shards: int
    z_value: int | None

    def as_dict(self) -> dict[str, str | int | None]:
        """Return a primitive plan representation."""
        return {
            "case_id": self.case_id,
            "layout": self.layout,
            "jackhmmer_n_cpu": self.jackhmmer_n_cpu,
            "active_shards": self.active_shards,
            "aggregate_cpu_slots": self.jackhmmer_n_cpu * self.active_shards,
            "z_value": self.z_value,
            "dom_z_value": self.z_value,
        }


SCREENING_QUERY = SearchQuery(
    query_id="pembrolizumab-vh",
    role="screening",
    sequence=PEMBROLIZUMAB_VH_SEQUENCE,
    sequence_sha256=PEMBROLIZUMAB_VH_SHA256,
)
STRESS_QUERY = SearchQuery(
    query_id="ecoli-k12-groel",
    role="stress",
    sequence=ECOLI_K12_GROEL_SEQUENCE,
    sequence_sha256=ECOLI_K12_GROEL_SHA256,
)
SEARCH_CASES = (
    SearchCase("B0", "monolith", 8, 1, None),
    SearchCase("B1", "monolith", 8, 1, SMALL_BFD_Z),
    SearchCase("S0", "shards", 8, 1, SMALL_BFD_Z),
    SearchCase("S1", "shards", 2, 4, SMALL_BFD_Z),
    SearchCase("S2", "shards", 2, 8, SMALL_BFD_Z),
    SearchCase("S3", "shards", 2, 16, SMALL_BFD_Z),
    SearchCase("S4", "shards", 4, 8, SMALL_BFD_Z),
    SearchCase("S5", "shards", 8, 4, SMALL_BFD_Z),
    SearchCase("S6", "shards", 1, 32, SMALL_BFD_Z),
)


def _search_case(case_id: str) -> SearchCase:
    """Resolve one fixed benchmark search case."""
    for case in SEARCH_CASES:
        if case.case_id == case_id:
            return case
    choices = ", ".join(case.case_id for case in SEARCH_CASES)
    raise ValueError(f"Unknown search case {case_id!r}; expected one of {choices}")


def _search_query(query_id: str) -> SearchQuery:
    """Resolve one fixed query and validate its embedded digest."""
    queries = (SCREENING_QUERY, STRESS_QUERY)
    for query in queries:
        if query.query_id == query_id:
            measured_sha256 = _sha256_bytes(query.sequence.encode())
            if measured_sha256 != query.sequence_sha256:
                raise RuntimeError(f"Embedded {query.query_id} SHA-256 is invalid")
            return query
    choices = ", ".join(query.query_id for query in queries)
    raise ValueError(f"Unknown search query {query_id!r}; expected one of {choices}")


def _focused_sweep_sample_id(case_id: str) -> str:
    """Return the fixed reused or new sample ID for one sweep case."""
    _search_case(case_id)
    if case_id in FOCUSED_SWEEP_REUSED_CASE_IDS:
        return f"smoke-{case_id.lower()}"
    if case_id in FOCUSED_SWEEP_NEW_CASE_IDS:
        return f"sweep-{case_id.lower()}"
    choices = ", ".join(FOCUSED_SWEEP_CASE_IDS)
    raise ValueError(f"Focused sweep case must be one of {choices}")


def _validate_sample_id(sample_id: str) -> str:
    """Reject sample identifiers that cannot be safe path components."""
    if not isinstance(sample_id, str) or not 1 <= len(sample_id) <= 64:
        raise ValueError("sample_id must contain between 1 and 64 characters")
    allowed = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")
    if sample_id[0] == "-" or any(character not in allowed for character in sample_id):
        raise ValueError("sample_id must use lowercase letters, digits, and hyphens")
    return sample_id


def _scientific_search_config(case: SearchCase) -> dict[str, object]:
    """Return only result-affecting Jackhmmer configuration."""
    return {
        "alphafold_commit": CONF.repo_commit_hash,
        "hmmer_version": HMMER_VERSION,
        "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
        "database_layout": case.layout,
        "n_iter": JACKHMMER_N_ITER,
        "e_value": JACKHMMER_E_VALUE,
        "z_value": case.z_value,
        "dom_z_value": case.z_value,
        "max_sequences": JACKHMMER_MAX_SEQUENCES,
        "filter_f1": JACKHMMER_FILTER_F1,
        "filter_f2": JACKHMMER_FILTER_F2,
        "filter_f3": JACKHMMER_FILTER_F3,
        "seq_limit_patch": True,
    }


def _profile_scientific_identity(manifest: dict[str, Any]) -> str:
    """Hash profile content and recipe while excluding operational thread count."""
    source, shards, _ = _validate_profile_manifest(manifest)
    recipe = manifest.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Profile manifest recipe must be an object")
    return _sha256_bytes(
        _json_bytes({
            "schema_version": manifest["schema_version"],
            "profile_id": manifest["profile_id"],
            "database_id": manifest["database_id"],
            "source": source,
            "shards": shards,
            "shard_count": manifest["shard_count"],
            "z_value": manifest["z_value"],
            "recipe": {
                "version": recipe.get("version"),
                "seqkit_version": recipe.get("seqkit_version"),
                "random_seed": recipe.get("random_seed"),
                "shuffle": recipe.get("shuffle"),
                "duplicate_recovery": recipe.get("duplicate_recovery"),
                "split": recipe.get("split"),
            },
        })
    )


def _search_identity(
    profile_scientific_identity: str,
    query: SearchQuery,
    case: SearchCase,
) -> str:
    """Hash query, database, and result-affecting settings only."""
    return _sha256_bytes(
        _json_bytes({
            "schema_version": 1,
            "database_id": DATABASE_ID,
            "profile_id": PROFILE_ID,
            "profile_scientific_identity": profile_scientific_identity,
            "query": query.as_dict(),
            "scientific_config": _scientific_search_config(case),
        })
    )


def _search_sample_identity(
    profile_scientific_identity: str,
    query: SearchQuery,
    case: SearchCase,
    sample_id: str,
) -> str:
    """Hash both scientific inputs and operational sample settings."""
    validated_sample_id = _validate_sample_id(sample_id)
    return _sha256_bytes(
        _json_bytes({
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "query": query.as_dict(),
            "search_identity": _search_identity(
                profile_scientific_identity,
                query,
                case,
            ),
            "sample_id": validated_sample_id,
            "operational_config": {
                "case_id": case.case_id,
                "jackhmmer_n_cpu": case.jackhmmer_n_cpu,
                "active_shards": case.active_shards,
                "resource_trace_interval_seconds": RESOURCE_TRACE_INTERVAL_SECONDS,
            },
        })
    )


def _search_sample_relpath(
    query: SearchQuery,
    search_identity: str,
    sample_id: str,
) -> str:
    """Return the common sequence-addressed raw-MSA sample path."""
    validated_sample_id = _validate_sample_id(sample_id)
    if len(search_identity) != 64:
        raise ValueError("search_identity must be a SHA-256 digest")
    sequence_sha256 = query.sequence_sha256
    return (
        f"{sequence_sha256[:2]}/{sequence_sha256}/raw-msa/{DATABASE_ID}/"
        f"{search_identity}/samples/{validated_sample_id}"
    )


def _build_search_plan(mode: str) -> dict[str, object]:
    """Build a side-effect-free fixed search plan."""
    common: dict[str, object] = {
        "campaign_id": CAMPAIGN_ID,
        "operation": "search",
        "mode": mode,
        "execution": "sequential",
        "resources_per_container": {
            "cpu": [0.125, 32.125],
            "memory_mib": [1024, 131_072],
            "timeout_seconds": CONF.timeout,
        },
        "cache_policy": "validate-done-marker-before-remote-call",
    }
    if mode == "smoke":
        cases = [_search_case(case_id) for case_id in SMOKE_CASE_IDS]
        return common | {
            "query": SCREENING_QUERY.as_dict(),
            "cases": [case.as_dict() for case in cases],
            "remote_function_inputs": len(cases),
            "counted_as_benchmark_samples": False,
            "oracle_case": "B1",
            "scientific_gate_case": "S3",
        }
    if mode == "sweep":
        return common | {
            "query": SCREENING_QUERY.as_dict(),
            "prerequisite": "completed passing smoke gate",
            "oracle_case": "B1",
            "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
            "reused_samples": [
                {
                    "case_id": case_id,
                    "sample_id": _focused_sweep_sample_id(case_id),
                }
                for case_id in FOCUSED_SWEEP_REUSED_CASE_IDS
            ],
            "new_samples": [
                {
                    "case": _search_case(case_id).as_dict(),
                    "sample_id": _focused_sweep_sample_id(case_id),
                }
                for case_id in FOCUSED_SWEEP_NEW_CASE_IDS
            ],
            "remote_function_inputs": len(FOCUSED_SWEEP_NEW_CASE_IDS),
            "total_case_results": len(FOCUSED_SWEEP_CASE_IDS),
            "runs_per_new_case": 1,
            "stress_samples": 0,
            "selection_policy": {
                "minimum_search_wall_improvement_vs_B1": 0.20,
                "cost_candidate_maximum_slowdown_vs_fastest": 0.15,
                "close_results_require_review_within": 0.15,
            },
        }
    if mode == "matrix":
        return common | {
            "screening_query": SCREENING_QUERY.as_dict(),
            "stress_query": STRESS_QUERY.as_dict(),
            "screening_block_orders": [list(order) for order in SCREENING_BLOCK_ORDERS],
            "screening_samples": 24,
            "conditional_stress_samples": 12,
            "maximum_remote_function_inputs": 36,
            "stress_cases": "B0, B1, and two promoted sharded layouts",
            "prerequisite": "completed passing smoke gate",
            "scientific_gate": {
                "oracle_case": "B1",
                "top_unique_hits_exact": 100,
                "minimum_full_unique_hit_jaccard": 0.99,
            },
            "performance_gate": {
                "minimum_search_wall_improvement_vs_B1": 0.20,
                "cost_candidate_maximum_slowdown_vs_fastest": 0.15,
                "maximum_three_sample_variation": 0.10,
            },
            "pricing": {
                "cpu_usd_per_core_second": MODAL_CPU_USD_PER_CORE_SECOND,
                "memory_usd_per_gib_second": MODAL_MEMORY_USD_PER_GIB_SECOND,
                "observed_date": MODAL_PRICING_OBSERVED_DATE,
                "source": MODAL_PRICING_URL,
            },
        }
    raise ValueError("search mode must be 'smoke', 'sweep', or 'matrix'")


def _campaign_plan_bytes() -> bytes:
    """Serialize the immutable plan shared by every campaign operation."""
    return _json_bytes({
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "profile_id": PROFILE_ID,
        "prepare": _build_prepare_plan(DEFAULT_SEQKIT_THREADS),
        "storage_scan": _build_scan_plan(),
        "search_smoke": _build_search_plan("smoke"),
        "search_matrix": _build_search_plan("matrix"),
    })


def _ensure_campaign_plan_mounted() -> None:
    """Create or validate the immutable campaign plan through its mount."""
    plan_path = Path(APP_INFO.output_dir) / "benchmarks" / CAMPAIGN_ID / "plan.json"
    expected = _campaign_plan_bytes()
    if plan_path.is_file():
        if plan_path.read_bytes() != expected:
            raise ValueError("Existing campaign plan differs from this app")
        return
    _write_bytes_exclusive(plan_path, expected)


def _ensure_campaign_plan_client() -> None:
    """Create or validate the immutable campaign plan through the client."""
    relative_path = f"benchmarks/{CAMPAIGN_ID}/plan.json"
    expected = _campaign_plan_bytes()
    try:
        existing = _read_volume_bytes(BENCHMARK_OUTPUT_VOLUME, relative_path)
    except FileNotFoundError:
        _upload_volume_bytes(BENCHMARK_OUTPUT_VOLUME, relative_path, expected)
    else:
        if existing != expected:
            raise ValueError("Existing campaign plan differs from this app")


def _parse_a3m_records(a3m: str) -> list[tuple[str, str]]:
    """Parse the small, truncated merged A3M without importing AlphaFold."""
    records: list[tuple[str, str]] = []
    description: str | None = None
    sequence_parts: list[str] = []
    for raw_line in a3m.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if description is not None:
                if not sequence_parts:
                    raise ValueError(f"A3M record {description!r} has no sequence")
                records.append(("".join(sequence_parts), description))
            description = line[1:]
            if not description:
                raise ValueError("A3M record has an empty description")
            sequence_parts = []
        else:
            if description is None:
                raise ValueError("A3M sequence appears before its description")
            sequence_parts.append(line)
    if description is not None:
        if not sequence_parts:
            raise ValueError(f"A3M record {description!r} has no sequence")
        records.append(("".join(sequence_parts), description))
    if not records:
        raise ValueError("Merged A3M is empty")
    return records


def _normalize_a3m_sequence(sequence: str) -> str:
    """Remove A3M insertions and dot gaps for exact aligned-hit comparison."""
    return "".join(
        character
        for character in sequence
        if not character.islower() and character != "."
    )


def _tblout_index(
    raw_tblouts: list[tuple[str, str]],
) -> tuple[dict[str, dict[str, object]], dict[str, list[dict[str, object]]]]:
    """Parse raw tblouts using the same target-name key as AlphaFold."""
    latest_by_target: dict[str, dict[str, object]] = {}
    occurrences: dict[str, list[dict[str, object]]] = {}
    for source, tblout in raw_tblouts:
        for line_number, line in enumerate(tblout.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 6:
                raise ValueError(
                    f"Invalid tblout line {line_number} from {source}: {line!r}"
                )
            entry: dict[str, object] = {
                "target_id": fields[0],
                "e_value_text": fields[4],
                "bit_score_text": fields[5],
                "e_value": float(fields[4]),
                "bit_score": float(fields[5]),
                "source": source,
                "line": line,
            }
            target_id = fields[0]
            occurrences.setdefault(target_id, []).append(entry)
            # This deliberately mirrors AlphaFold's last-tblout-line-wins map.
            latest_by_target[target_id] = entry
    return latest_by_target, occurrences


def _normalized_hit_rows(
    merged_a3m: str,
    raw_tblouts: list[tuple[str, str]],
) -> list[dict[str, object]]:
    """Create one normalized evidence row per non-query merged A3M record."""
    records = _parse_a3m_records(merged_a3m)
    latest_by_target, occurrences = _tblout_index(raw_tblouts)
    rows: list[dict[str, object]] = []
    for ordinal, (sequence, description) in enumerate(records[1:], start=1):
        target_id = description.partition(" ")[0].partition("/")[0]
        entry = latest_by_target.get(target_id)
        if entry is None:
            raise ValueError(f"Merged A3M target has no tblout row: {target_id}")
        target_occurrences = occurrences[target_id]
        occurrence_sources = [str(item["source"]) for item in target_occurrences]
        normalized_sequence = _normalize_a3m_sequence(sequence)
        rows.append({
            "ordinal": ordinal,
            "target_id": target_id,
            "description": description,
            "aligned_sequence": sequence,
            "normalized_sequence": normalized_sequence,
            "normalized_sequence_sha256": _sha256_bytes(normalized_sequence.encode()),
            "e_value": entry["e_value"],
            "e_value_text": entry["e_value_text"],
            "bit_score": entry["bit_score"],
            "bit_score_text": entry["bit_score_text"],
            "tblout_source": entry["source"],
            "tblout_line": entry["line"],
            "raw_occurrence_count": len(target_occurrences),
            "raw_occurrence_sources": ",".join(occurrence_sources),
            "cross_shard_duplicate": len(set(occurrence_sources)) > 1,
        })
    return rows


def _normalized_hits_parquet(rows: list[dict[str, object]]) -> bytes:
    """Serialize normalized hit evidence with a stable schema."""
    import polars as pl

    schema = {
        "ordinal": pl.Int64,
        "target_id": pl.String,
        "description": pl.String,
        "aligned_sequence": pl.String,
        "normalized_sequence": pl.String,
        "normalized_sequence_sha256": pl.String,
        "e_value": pl.Float64,
        "e_value_text": pl.String,
        "bit_score": pl.Float64,
        "bit_score_text": pl.String,
        "tblout_source": pl.String,
        "tblout_line": pl.String,
        "raw_occurrence_count": pl.Int64,
        "raw_occurrence_sources": pl.String,
        "cross_shard_duplicate": pl.Boolean,
    }
    table = pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)
    buffer = io.BytesIO()
    table.write_parquet(buffer)
    return buffer.getvalue()


def _write_bytes_exclusive(path: Path, data: bytes) -> None:
    """Write one evidence artifact without replacing an existing sample."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _read_optional_integer(path: Path) -> int | None:
    """Read a Linux counter if exposed by the current cgroup."""
    try:
        value = path.read_text(encoding="utf-8").strip()
        return None if value == "max" else int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        return None


def _cgroup_cpu_stats() -> dict[str, int | None]:
    """Read cumulative cgroup CPU use and throttling counters."""
    stats: dict[str, int | None] = {
        "usage_usec": None,
        "nr_periods": None,
        "nr_throttled": None,
        "throttled_usec": None,
    }
    try:
        lines = Path("/sys/fs/cgroup/cpu.stat").read_text(encoding="utf-8")
        for line in lines.splitlines():
            key, _, value = line.partition(" ")
            if key in stats:
                stats[key] = int(value)
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    if stats["usage_usec"] is None:
        usage_ns = _read_optional_integer(Path("/sys/fs/cgroup/cpuacct/cpuacct.usage"))
        stats["usage_usec"] = None if usage_ns is None else usage_ns // 1_000
    return stats


def _mark_resource_phase(
    phase_state: dict[str, Any],
    phase: str,
    started: float,
) -> None:
    """Record an exact phase boundary for the next trace observation."""
    phase_state["current"] = phase
    events = phase_state.setdefault("events", [])
    if not isinstance(events, list):
        raise TypeError("Resource phase events must be a list")
    events.append({
        "phase": phase,
        "observed_at": _utc_now(),
        "elapsed_seconds": perf_counter() - started,
    })


def _resource_snapshot(
    started: float,
    phase_state: dict[str, Any],
    context: dict[str, object],
) -> dict[str, object]:
    """Capture one portable one-second resource observation."""
    import resource

    load_average: list[float] | None = None
    if hasattr(os, "getloadavg"):
        load_average = list(os.getloadavg())
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_stats = _cgroup_cpu_stats()
    events = phase_state.get("events")
    if not isinstance(events, list):
        events = []
    return context | {
        "observed_at": _utc_now(),
        "elapsed_seconds": perf_counter() - started,
        "phase": phase_state.get("current"),
        "phase_events": list(events),
        "cpu_usage_usec": cpu_stats["usage_usec"],
        "cpu_nr_periods": cpu_stats["nr_periods"],
        "cpu_nr_throttled": cpu_stats["nr_throttled"],
        "cpu_throttled_usec": cpu_stats["throttled_usec"],
        "memory_current_bytes": _read_optional_integer(
            Path("/sys/fs/cgroup/memory.current")
        ),
        "memory_peak_bytes": _read_optional_integer(Path("/sys/fs/cgroup/memory.peak")),
        "load_average": load_average,
        "cpu_affinity": affinity,
        "children_user_seconds": children.ru_utime,
        "children_system_seconds": children.ru_stime,
    }


def _trace_resources(
    trace_path: Path,
    stop: Event,
    started: float,
    phase_state: dict[str, Any],
    context: dict[str, object],
) -> None:
    """Persist cgroup measurements once a second until the sample finishes."""
    with trace_path.open("xb") as handle:
        while True:
            snapshot = _resource_snapshot(started, phase_state, context)
            handle.write(orjson.dumps(snapshot) + b"\n")
            handle.flush()
            if stop.wait(RESOURCE_TRACE_INTERVAL_SECONDS):
                snapshot = _resource_snapshot(started, phase_state, context)
                handle.write(orjson.dumps(snapshot) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                return


def _summarize_resource_trace(trace_path: Path) -> dict[str, int | float | None]:
    """Reduce the raw trace to actual CPU, throttling, and memory signals."""
    snapshots = [
        orjson.loads(line)
        for line in trace_path.read_bytes().splitlines()
        if line.strip()
    ]
    if not snapshots:
        raise ValueError("Resource trace is empty")
    cpu_rates: list[float] = []
    memory_gib_seconds = 0.0
    has_memory_integral = False
    for previous, current in zip(snapshots, snapshots[1:], strict=False):
        elapsed = float(current["elapsed_seconds"]) - float(previous["elapsed_seconds"])
        previous_cpu = previous.get("cpu_usage_usec")
        current_cpu = current.get("cpu_usage_usec")
        if (
            isinstance(previous_cpu, int)
            and isinstance(current_cpu, int)
            and elapsed > 0
        ):
            cpu_rates.append((current_cpu - previous_cpu) / 1_000_000 / elapsed)
        previous_memory = previous.get("memory_current_bytes")
        current_memory = current.get("memory_current_bytes")
        if (
            isinstance(previous_memory, int)
            and isinstance(current_memory, int)
            and elapsed > 0
        ):
            mean_memory_bytes = (previous_memory + current_memory) / 2
            memory_gib_seconds += mean_memory_bytes / (1024**3) * elapsed
            has_memory_integral = True

    first_cpu = snapshots[0].get("cpu_usage_usec")
    last_cpu = snapshots[-1].get("cpu_usage_usec")
    cpu_core_seconds: float | None = None
    if isinstance(first_cpu, int) and isinstance(last_cpu, int):
        cpu_core_seconds = (last_cpu - first_cpu) / 1_000_000
    memory_values = [
        value
        for snapshot in snapshots
        for value in (snapshot.get("memory_current_bytes"),)
        if isinstance(value, int)
    ]
    peak_values = [
        value
        for snapshot in snapshots
        for value in (snapshot.get("memory_peak_bytes"),)
        if isinstance(value, int)
    ]
    first_throttled = snapshots[0].get("cpu_throttled_usec")
    last_throttled = snapshots[-1].get("cpu_throttled_usec")
    throttled_seconds: float | None = None
    if isinstance(first_throttled, int) and isinstance(last_throttled, int):
        throttled_seconds = (last_throttled - first_throttled) / 1_000_000
    first_nr_throttled = snapshots[0].get("cpu_nr_throttled")
    last_nr_throttled = snapshots[-1].get("cpu_nr_throttled")
    throttled_periods: int | None = None
    if isinstance(first_nr_throttled, int) and isinstance(last_nr_throttled, int):
        throttled_periods = last_nr_throttled - first_nr_throttled
    first_child_user = snapshots[0].get("children_user_seconds")
    last_child_user = snapshots[-1].get("children_user_seconds")
    first_child_system = snapshots[0].get("children_system_seconds")
    last_child_system = snapshots[-1].get("children_system_seconds")
    child_cpu_values = (
        first_child_user,
        last_child_user,
        first_child_system,
        last_child_system,
    )
    child_cpu_seconds: float | None = None
    if all(isinstance(value, int | float) for value in child_cpu_values):
        child_cpu_seconds = (
            float(last_child_user)
            - float(first_child_user)
            + float(last_child_system)
            - float(first_child_system)
        )
    return {
        "observations": len(snapshots),
        "elapsed_seconds": float(snapshots[-1]["elapsed_seconds"]),
        "cpu_core_seconds": cpu_core_seconds,
        "child_process_cpu_seconds": child_cpu_seconds,
        "peak_interval_cpu_cores": max(cpu_rates) if cpu_rates else None,
        "cpu_throttled_periods": throttled_periods,
        "cpu_throttled_seconds": throttled_seconds,
        "memory_gib_seconds": memory_gib_seconds if has_memory_integral else None,
        "peak_memory_current_bytes": max(memory_values) if memory_values else None,
        "cgroup_memory_peak_bytes": max(peak_values) if peak_values else None,
    }


def _estimate_compute_cost(
    resource_summary: dict[str, int | float | None],
    sample_wall_seconds: float,
) -> dict[str, float | str]:
    """Estimate billed CPU and memory using Modal's published Function rates."""
    cpu_core_seconds = resource_summary.get("cpu_core_seconds")
    memory_gib_seconds = resource_summary.get("memory_gib_seconds")
    observed_cpu = (
        float(cpu_core_seconds) if isinstance(cpu_core_seconds, int | float) else 0.0
    )
    observed_memory = (
        float(memory_gib_seconds)
        if isinstance(memory_gib_seconds, int | float)
        else 0.0
    )
    billed_cpu = max(0.125 * sample_wall_seconds, observed_cpu)
    billed_memory = max(1.0 * sample_wall_seconds, observed_memory)
    cpu_cost = billed_cpu * MODAL_CPU_USD_PER_CORE_SECOND
    memory_cost = billed_memory * MODAL_MEMORY_USD_PER_GIB_SECOND
    return {
        "estimated_billed_cpu_core_seconds": billed_cpu,
        "estimated_billed_memory_gib_seconds": billed_memory,
        "estimated_cpu_cost_usd": cpu_cost,
        "estimated_memory_cost_usd": memory_cost,
        "estimated_compute_cost_usd": cpu_cost + memory_cost,
        "pricing_observed_date": MODAL_PRICING_OBSERVED_DATE,
        "pricing_source": MODAL_PRICING_URL,
    }


def _execute_jackhmmer_search(
    profile_root: Path,
    query: SearchQuery,
    case: SearchCase,
    phase_state: dict[str, Any],
    sample_started: float,
) -> tuple[str, list[tuple[str, str]], dict[str, object]]:
    """Run the pinned wrapper while retaining raw tblout and phase timings."""
    from importlib import import_module

    jackhmmer = import_module("alphafold3.data.tools.jackhmmer")

    if case.layout == "monolith":
        database_path = str(profile_root / "source" / SOURCE_DB_FILENAME)
    else:
        database_path = f"{profile_root / 'shards' / SOURCE_DB_FILENAME}@{SHARD_COUNT}"
    tool = jackhmmer.Jackhmmer(
        binary_path=JACKHMMER_BINARY_PATH,
        database_path=database_path,
        n_cpu=case.jackhmmer_n_cpu,
        n_iter=JACKHMMER_N_ITER,
        e_value=JACKHMMER_E_VALUE,
        z_value=case.z_value,
        dom_z_value=case.z_value,
        max_sequences=JACKHMMER_MAX_SEQUENCES,
        filter_f1=JACKHMMER_FILTER_F1,
        filter_f2=JACKHMMER_FILTER_F2,
        filter_f3=JACKHMMER_FILTER_F3,
        max_threads=case.active_shards,
    )
    _mark_resource_phase(phase_state, "query", sample_started)
    search_started = perf_counter()
    if case.layout == "monolith":
        shard_started = perf_counter()
        result = tool._query_db_shard(  # noqa: SLF001
            target_sequence=query.sequence,
            db_shard_path=database_path,
            get_tblout=True,
        )
        shard_finished = perf_counter()
        if result.tblout is None:
            raise ValueError("Monolith Jackhmmer result did not contain tblout")
        search_wall_seconds = perf_counter() - search_started
        return (
            result.a3m,
            [("monolith", result.tblout)],
            {
                "search_wall_seconds": search_wall_seconds,
                "merge_wall_seconds": 0.0,
                "shards": [
                    {
                        "source": "monolith",
                        "started_seconds": shard_started - search_started,
                        "finished_seconds": shard_finished - search_started,
                        "wall_seconds": shard_finished - shard_started,
                    }
                ],
            },
        )

    shard_paths = tuple(profile_root / "shards" / name for name in _shard_names())
    global_temp_dir = tempfile.mkdtemp(prefix="af3-msa-search-")

    def query_shard(shard_path: Path) -> tuple[Any, dict[str, object]]:
        shard_started = perf_counter()
        result = tool._query_db_shard(  # noqa: SLF001
            target_sequence=query.sequence,
            db_shard_path=str(shard_path),
            get_tblout=True,
            global_temp_dir=global_temp_dir,
        )
        shard_finished = perf_counter()
        return result, {
            "source": shard_path.name,
            "started_seconds": shard_started - search_started,
            "finished_seconds": shard_finished - search_started,
            "wall_seconds": shard_finished - shard_started,
        }

    try:
        with ThreadPoolExecutor(max_workers=case.active_shards) as executor:
            outputs = tuple(executor.map(query_shard, shard_paths))
    finally:
        shutil.rmtree(global_temp_dir, ignore_errors=True)
    results = [result for result, unused_timing in outputs]
    raw_tblouts: list[tuple[str, str]] = []
    for result, timing in outputs:
        if result.tblout is None:
            raise ValueError(f"Shard {timing['source']} did not contain tblout")
        raw_tblouts.append((str(timing["source"]), result.tblout))
    _mark_resource_phase(phase_state, "merge", sample_started)
    merge_started = perf_counter()
    merged = jackhmmer._merge_jackhmmer_results(  # noqa: SLF001
        results,
        JACKHMMER_MAX_SEQUENCES,
    )
    merge_wall_seconds = perf_counter() - merge_started
    search_wall_seconds = perf_counter() - search_started
    return (
        merged.a3m,
        raw_tblouts,
        {
            "search_wall_seconds": search_wall_seconds,
            "merge_wall_seconds": merge_wall_seconds,
            "shards": [timing for unused_result, timing in outputs],
        },
    )


def _run_search_sample(
    query_id: str,
    case_id: str,
    sample_id: str,
    expected_search_identity: str,
    expected_sample_identity: str,
) -> dict[str, object]:
    """Run one immutable sample and publish its completion marker last."""
    sample_started = perf_counter()
    query = _search_query(query_id)
    case = _search_case(case_id)
    validated_sample_id = _validate_sample_id(sample_id)
    profile_root = Path(APP_INFO.sharded_db_dir) / APP_INFO.profile_relpath
    SHARDED_MSA_DB_VOLUME.reload()
    BENCHMARK_OUTPUT_VOLUME.reload()
    manifest = _validate_published_profile(profile_root, verify_digests=False)
    manifest_path = profile_root / "manifest.json"
    manifest_sha256 = _sha256_file(manifest_path)
    profile_scientific_identity = _profile_scientific_identity(manifest)
    search_identity = _search_identity(profile_scientific_identity, query, case)
    sample_identity = _search_sample_identity(
        profile_scientific_identity,
        query,
        case,
        validated_sample_id,
    )
    if search_identity != expected_search_identity:
        raise ValueError("Client and worker search identities differ")
    if sample_identity != expected_sample_identity:
        raise ValueError("Client and worker sample identities differ")

    sample_relpath = _search_sample_relpath(
        query,
        search_identity,
        validated_sample_id,
    )
    final_output_root = Path(APP_INFO.output_dir) / sample_relpath
    try:
        _validate_done_marker(final_output_root, expected_identity=sample_identity)
    except (FileNotFoundError, ValueError):
        pass
    else:
        metrics = _load_json_object(final_output_root / "metrics.json")
        return metrics | {"status": "reused"}

    output_root = (
        final_output_root.parent
        / ".staging"
        / f"{validated_sample_id}-{uuid.uuid4().hex}"
    )
    output_root.mkdir(parents=True)
    log_path = output_root / "run.log"
    trace_path = output_root / "trace.jsonl"
    container = _container_sample_metadata()
    phase_state: dict[str, Any] = {"current": None, "events": []}
    _mark_resource_phase(phase_state, "warmup", sample_started)
    _append_log(
        log_path,
        f"Starting {query.query_id} {case.case_id} sample {validated_sample_id}",
    )
    trace_stop = Event()
    trace_thread = Thread(
        target=_trace_resources,
        args=(
            trace_path,
            trace_stop,
            sample_started,
            phase_state,
            {
                "query_id": query.query_id,
                "case_id": case.case_id,
                "jackhmmer_n_cpu": case.jackhmmer_n_cpu,
                "active_shards": case.active_shards,
                "container_instance_id": container["container_instance_id"],
            },
        ),
        daemon=True,
    )
    trace_thread.start()
    try:
        merged_a3m, raw_tblouts, timings = _execute_jackhmmer_search(
            profile_root,
            query,
            case,
            phase_state,
            sample_started,
        )
        _mark_resource_phase(phase_state, "publish", sample_started)
        merged_a3m_path = output_root / "result.a3m"
        _write_bytes_exclusive(merged_a3m_path, merged_a3m.encode())
        raw_tblout_paths: list[Path] = []
        for source, tblout in raw_tblouts:
            if source == "monolith":
                tblout_path = output_root / "result.tblout"
            else:
                tblout_path = output_root / "shards" / f"{source}.tblout"
            _write_bytes_exclusive(tblout_path, tblout.encode())
            raw_tblout_paths.append(tblout_path)
        hit_rows = _normalized_hit_rows(merged_a3m, raw_tblouts)
        normalized_hits_path = output_root / "hits.parquet"
        _write_bytes_exclusive(
            normalized_hits_path,
            _normalized_hits_parquet(hit_rows),
        )
        _append_log(
            log_path,
            f"Completed search with {len(hit_rows)} merged hit rows",
        )
    except Exception as exc:
        trace_stop.set()
        trace_thread.join()
        _append_log(log_path, f"Failed with {type(exc).__name__}: {exc}")
        _write_json_atomic(
            output_root / "failure.json",
            {
                "failed_at": _utc_now(),
                "query": query.as_dict(),
                "case": case.as_dict(),
                "sample_id": validated_sample_id,
                "sample_identity": sample_identity,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        raise
    _mark_resource_phase(phase_state, "complete", sample_started)
    trace_stop.set()
    trace_thread.join()
    resource_summary = _summarize_resource_trace(trace_path)
    core_artifacts = [
        _artifact_record(merged_a3m_path, output_root),
        _artifact_record(normalized_hits_path, output_root),
        _artifact_record(trace_path, output_root),
        _artifact_record(log_path, output_root),
        *(_artifact_record(path, output_root) for path in raw_tblout_paths),
    ]
    BENCHMARK_OUTPUT_VOLUME.commit()
    sample_wall_seconds = perf_counter() - sample_started
    cost_estimate = _estimate_compute_cost(resource_summary, sample_wall_seconds)
    unique_hit_count = len({str(row["target_id"]) for row in hit_rows})
    cross_shard_duplicate_count = len({
        str(row["target_id"])
        for row in hit_rows
        if row["cross_shard_duplicate"] is True
    })
    metrics: dict[str, object] = {
        "status": "published",
        "campaign_id": CAMPAIGN_ID,
        "database_id": DATABASE_ID,
        "profile_id": PROFILE_ID,
        "profile_manifest_sha256": manifest_sha256,
        "profile_scientific_identity": profile_scientific_identity,
        "query": query.as_dict(),
        "case": case.as_dict(),
        "scientific_config": _scientific_search_config(case),
        "search_identity": search_identity,
        "sample_id": validated_sample_id,
        "sample_identity": sample_identity,
        "result_path": sample_relpath,
        "search_wall_seconds": timings["search_wall_seconds"],
        "merge_wall_seconds": timings["merge_wall_seconds"],
        "sample_wall_seconds": sample_wall_seconds,
        "sample_wall_endpoint": "durable-core-evidence-commit",
        "shard_timings": timings["shards"],
        "hit_rows": len(hit_rows),
        "unique_hits": unique_hit_count,
        "duplicate_hit_rows": len(hit_rows) - unique_hit_count,
        "cross_shard_duplicate_targets": cross_shard_duplicate_count,
        "resource_summary": resource_summary,
        "cost_estimate": cost_estimate,
        "container": container,
    }
    metrics_path = output_root / "metrics.json"
    _write_json_atomic(metrics_path, metrics)
    BENCHMARK_OUTPUT_VOLUME.commit()
    artifacts = [*core_artifacts, _artifact_record(metrics_path, output_root)]
    _write_json_atomic(
        output_root / "done.json",
        {
            "schema_version": DONE_SCHEMA_VERSION,
            "status": "complete",
            "identity": sample_identity,
            "completed_at": _utc_now(),
            "artifacts": artifacts,
        },
    )
    BENCHMARK_OUTPUT_VOLUME.commit()
    if final_output_root.exists():
        try:
            _validate_done_marker(
                final_output_root,
                expected_identity=sample_identity,
            )
        except (FileNotFoundError, ValueError):
            orphan_root = final_output_root.parent / ".orphaned"
            orphan_root.mkdir(parents=True, exist_ok=True)
            final_output_root.replace(
                orphan_root / f"{validated_sample_id}-{uuid.uuid4().hex}"
            )
        else:
            duplicate_root = final_output_root.parent / ".orphaned"
            duplicate_root.mkdir(parents=True, exist_ok=True)
            output_root.replace(
                duplicate_root / f"{validated_sample_id}-duplicate-{uuid.uuid4().hex}"
            )
            BENCHMARK_OUTPUT_VOLUME.commit()
            existing = _load_json_object(final_output_root / "metrics.json")
            return existing | {"status": "reused-concurrent"}
    output_root.replace(final_output_root)
    BENCHMARK_OUTPUT_VOLUME.commit()
    return metrics


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 131_072),
    timeout=CONF.timeout,
    max_containers=1,
    volumes={
        APP_INFO.sharded_db_dir: SHARDED_MSA_DB_VOLUME.with_mount_options(
            read_only=True
        ),
        APP_INFO.output_dir: BENCHMARK_OUTPUT_VOLUME,
    },
)
def benchmark_small_bfd_search(
    query_id: str,
    case_id: str,
    sample_id: str,
    expected_search_identity: str,
    expected_sample_identity: str,
) -> dict[str, object]:
    """Run one cache-aware small-BFD Jackhmmer benchmark sample."""
    return _run_search_sample(
        query_id,
        case_id,
        sample_id,
        expected_search_identity,
        expected_sample_identity,
    )


def _unique_hit_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Keep the first merged occurrence of each AlphaFold target name."""
    unique_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        target_id = str(row["target_id"])
        if target_id not in seen:
            seen.add(target_id)
            unique_rows.append(row)
    return unique_rows


def _top_hits_tie_equivalent(
    oracle_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> bool:
    """Compare ranked hits while ignoring order inside exact score ties."""
    if len(oracle_rows) != len(candidate_rows):
        return False

    def tie_blocks(
        rows: list[dict[str, object]],
    ) -> list[tuple[tuple[str, str], tuple[str, ...]]]:
        blocks: list[tuple[tuple[str, str], tuple[str, ...]]] = []
        current_key: tuple[str, str] | None = None
        current_ids: list[str] = []
        for row in rows:
            key = (str(row["e_value_text"]), str(row["bit_score_text"]))
            if current_key is not None and key != current_key:
                blocks.append((current_key, tuple(sorted(current_ids))))
                current_ids = []
            current_key = key
            current_ids.append(str(row["target_id"]))
        if current_key is not None:
            blocks.append((current_key, tuple(sorted(current_ids))))
        return blocks

    return tie_blocks(oracle_rows) == tie_blocks(candidate_rows)


def _compare_normalized_hits(
    oracle_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Apply the fixed smoke scientific-equivalence gate."""
    oracle_unique = _unique_hit_rows(oracle_rows)
    candidate_unique = _unique_hit_rows(candidate_rows)
    oracle_ids = [str(row["target_id"]) for row in oracle_unique]
    candidate_ids = [str(row["target_id"]) for row in candidate_unique]
    top_width = min(100, max(len(oracle_ids), len(candidate_ids)))
    top_oracle = oracle_ids[:top_width]
    top_candidate = candidate_ids[:top_width]
    top_hits_exact = top_oracle == top_candidate
    top_hits_tie_equivalent = _top_hits_tie_equivalent(
        oracle_unique[:top_width],
        candidate_unique[:top_width],
    )

    oracle_by_id = {str(row["target_id"]): row for row in oracle_unique}
    candidate_by_id = {str(row["target_id"]): row for row in candidate_unique}
    shared_ids = set(oracle_by_id) & set(candidate_by_id)
    score_mismatches = sorted(
        target_id
        for target_id in shared_ids
        if (
            oracle_by_id[target_id]["e_value_text"],
            oracle_by_id[target_id]["bit_score_text"],
        )
        != (
            candidate_by_id[target_id]["e_value_text"],
            candidate_by_id[target_id]["bit_score_text"],
        )
    )
    sequence_mismatches = sorted(
        target_id
        for target_id in shared_ids
        if oracle_by_id[target_id]["normalized_sequence_sha256"]
        != candidate_by_id[target_id]["normalized_sequence_sha256"]
    )
    oracle_set = set(oracle_ids)
    candidate_set = set(candidate_ids)
    union = oracle_set | candidate_set
    overlap = len(oracle_set & candidate_set) / len(union) if union else 1.0
    oracle_only = [
        target_id for target_id in oracle_ids if target_id not in candidate_set
    ]
    candidate_only = [
        target_id for target_id in candidate_ids if target_id not in oracle_set
    ]
    oracle_positions = {
        target_id: position for position, target_id in enumerate(oracle_ids, start=1)
    }
    candidate_positions = {
        target_id: position for position, target_id in enumerate(candidate_ids, start=1)
    }
    difference_positions = [
        *(oracle_positions[target_id] for target_id in oracle_only),
        *(candidate_positions[target_id] for target_id in candidate_only),
    ]
    differences_are_below_top_100 = all(
        position > 100 for position in difference_positions
    )
    duplicate_targets = sorted({
        str(row["target_id"])
        for row in candidate_rows
        if row.get("cross_shard_duplicate") is True
    })
    has_set_differences = bool(oracle_only or candidate_only)
    both_results_reached_hit_row_limit = (
        len(oracle_rows) == JACKHMMER_MAX_SEQUENCES - 1
        and len(candidate_rows) == JACKHMMER_MAX_SEQUENCES - 1
    )
    candidate_duplicate_hit_rows = len(candidate_rows) - len(candidate_unique)
    oracle_tail_start = len(oracle_ids) - candidate_duplicate_hit_rows + 1
    oracle_only_is_displaced_tail = (
        not candidate_only
        and len(oracle_only) <= candidate_duplicate_hit_rows <= SHARD_COUNT
        and all(
            oracle_positions[target_id] >= oracle_tail_start
            for target_id in oracle_only
        )
    )
    differences_characterized = not has_set_differences or (
        differences_are_below_top_100
        and bool(duplicate_targets)
        and both_results_reached_hit_row_limit
        and oracle_only_is_displaced_tail
    )
    passed = (
        top_hits_tie_equivalent
        and not score_mismatches
        and not sequence_mismatches
        and overlap >= 0.99
        and differences_characterized
    )
    return {
        "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
        "passed": passed,
        "top_comparison_width": top_width,
        "top_hits_exact": top_hits_exact,
        "top_hits_tie_equivalent": top_hits_tie_equivalent,
        "top_order_differs_only_within_ties": (
            top_hits_tie_equivalent and not top_hits_exact
        ),
        "oracle_top_ids": top_oracle,
        "candidate_top_ids": top_candidate,
        "oracle_unique_hits": len(oracle_ids),
        "candidate_unique_hits": len(candidate_ids),
        "shared_unique_hits": len(shared_ids),
        "full_unique_hit_jaccard": overlap,
        "required_full_unique_hit_jaccard": 0.99,
        "score_mismatch_count": len(score_mismatches),
        "score_mismatch_ids": score_mismatches,
        "sequence_mismatch_count": len(sequence_mismatches),
        "sequence_mismatch_ids": sequence_mismatches,
        "oracle_only_ids": oracle_only,
        "candidate_only_ids": candidate_only,
        "differences_are_below_top_100": differences_are_below_top_100,
        "both_results_reached_hit_row_limit": both_results_reached_hit_row_limit,
        "candidate_duplicate_hit_rows": candidate_duplicate_hit_rows,
        "oracle_tail_start": oracle_tail_start,
        "oracle_only_is_displaced_tail": oracle_only_is_displaced_tail,
        "candidate_cross_shard_duplicate_targets": duplicate_targets,
        "differences_characterized_as_duplicate_tail": differences_characterized,
    }


def _read_normalized_hits(sample_relpath: str) -> list[dict[str, object]]:
    """Read one sample's normalized hit table through the Volume client."""
    import polars as pl

    data = _read_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{sample_relpath}/hits.parquet",
    )
    return pl.read_parquet(io.BytesIO(data)).to_dicts()


def _search_results_parquet(results: list[dict[str, Any]]) -> bytes:
    """Serialize one flat timing and resource row per search sample."""
    import polars as pl

    rows: list[dict[str, object]] = []
    for result in results:
        case = result.get("case")
        query = result.get("query")
        resource = result.get("resource_summary")
        cost = result.get("cost_estimate")
        container = result.get("container")
        if not isinstance(case, dict):
            raise ValueError("Search result is missing case metadata")
        if not isinstance(query, dict):
            raise ValueError("Search result is missing query metadata")
        if not isinstance(resource, dict):
            raise ValueError("Search result is missing resource metadata")
        if not isinstance(cost, dict):
            raise ValueError("Search result is missing cost metadata")
        if not isinstance(container, dict):
            raise ValueError("Search result is missing nested metadata")
        rows.append({
            "campaign_id": CAMPAIGN_ID,
            "sample_kind": result["sample_kind"],
            "block_index": result.get("block_index"),
            "query_id": query["query_id"],
            "query_length": query["length"],
            "case_id": case["case_id"],
            "layout": case["layout"],
            "jackhmmer_n_cpu": case["jackhmmer_n_cpu"],
            "active_shards": case["active_shards"],
            "aggregate_cpu_slots": case["aggregate_cpu_slots"],
            "search_identity": result["search_identity"],
            "sample_id": result["sample_id"],
            "sample_identity": result["sample_identity"],
            "search_wall_seconds": result["search_wall_seconds"],
            "merge_wall_seconds": result["merge_wall_seconds"],
            "sample_wall_seconds": result["sample_wall_seconds"],
            "remote_call_wall_seconds": result["remote_call_wall_seconds"],
            "hit_rows": result["hit_rows"],
            "unique_hits": result["unique_hits"],
            "duplicate_hit_rows": result["duplicate_hit_rows"],
            "cross_shard_duplicate_targets": result["cross_shard_duplicate_targets"],
            "cpu_core_seconds": resource.get("cpu_core_seconds"),
            "peak_interval_cpu_cores": resource.get("peak_interval_cpu_cores"),
            "cpu_throttled_periods": resource.get("cpu_throttled_periods"),
            "cpu_throttled_seconds": resource.get("cpu_throttled_seconds"),
            "peak_memory_current_bytes": resource.get("peak_memory_current_bytes"),
            "estimated_compute_cost_usd": cost.get("estimated_compute_cost_usd"),
            "container_hostname": container.get("hostname"),
            "container_instance_id": container.get("container_instance_id"),
            "container_sample_ordinal": container.get("container_sample_ordinal"),
            "container_reused_for_sample": container.get("container_reused_for_sample"),
            "result_path": result["result_path"],
            "reused": result["reused"],
        })
    buffer = io.BytesIO()
    pl.DataFrame(rows).sort(["query_id", "case_id", "sample_id"]).write_parquet(buffer)
    return buffer.getvalue()


def _smoke_summary_markdown(
    results: list[dict[str, Any]],
    comparisons: dict[str, dict[str, object]],
) -> str:
    """Render a compact durable smoke-test report."""
    gate_passed = comparisons["S3_vs_B1"]["passed"] is True
    jaccard = comparisons["S3_vs_B1"]["full_unique_hit_jaccard"]
    if not isinstance(jaccard, int | float):
        raise ValueError("Smoke comparison has an invalid Jaccard value")
    lines = [
        f"# {CAMPAIGN_ID} search smoke",
        "",
        f"Scientific gate: **{'PASS' if gate_passed else 'FAIL'}**",
        "",
        "| Case | Search wall (s) | Sample wall (s) | Unique hits | Reused |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for result in sorted(results, key=lambda item: str(item["case"]["case_id"])):
        case = result["case"]
        lines.append(
            f"| {case['case_id']} | {float(result['search_wall_seconds']):.3f} | "
            f"{float(result['sample_wall_seconds']):.3f} | "
            f"{int(result['unique_hits'])} | {bool(result['reused'])} |"
        )
    lines.extend([
        "",
        "B1 is the explicit-Z oracle. B0 is descriptive only; S3 is the "
        "required sharded scientific gate.",
        "",
        f"Scientific comparison policy: {SCIENTIFIC_COMPARISON_POLICY}",
        f"S3/B1 top hits exact: {comparisons['S3_vs_B1']['top_hits_exact']}",
        "S3/B1 top hits equivalent modulo exact score ties: "
        f"{comparisons['S3_vs_B1']['top_hits_tie_equivalent']}",
        f"S3/B1 full unique-hit Jaccard: {float(jaccard):.6f}",
        "",
    ])
    return "\n".join(lines)


def _smoke_operation_identity(
    manifest_sha256: str,
    sample_identities: dict[str, str],
) -> str:
    """Hash the complete smoke operation and its immutable sample identities."""
    return _sha256_bytes(
        _json_bytes({
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
            "profile_manifest_sha256": manifest_sha256,
            "plan": _build_search_plan("smoke"),
            "sample_identities": sample_identities,
        })
    )


def _submit_search_smoke() -> dict[str, object]:
    """Run only missing smoke samples, then publish the scientific gate."""
    _ensure_campaign_plan_client()
    manifest_relpath = f"{APP_INFO.profile_relpath}/manifest.json"
    manifest_bytes = _read_volume_bytes(SHARDED_MSA_DB_VOLUME, manifest_relpath)
    manifest = orjson.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Profile manifest must be a JSON object")
    _validate_profile_manifest(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    profile_scientific_identity = _profile_scientific_identity(manifest)
    cases = [_search_case(case_id) for case_id in SMOKE_CASE_IDS]
    sample_ids = {case.case_id: f"smoke-{case.case_id.lower()}" for case in cases}
    search_identities = {
        case.case_id: _search_identity(
            profile_scientific_identity,
            SCREENING_QUERY,
            case,
        )
        for case in cases
    }
    sample_identities = {
        case.case_id: _search_sample_identity(
            profile_scientific_identity,
            SCREENING_QUERY,
            case,
            sample_ids[case.case_id],
        )
        for case in cases
    }
    sample_relpaths = {
        case.case_id: _search_sample_relpath(
            SCREENING_QUERY,
            search_identities[case.case_id],
            sample_ids[case.case_id],
        )
        for case in cases
    }
    operation_root = f"benchmarks/{CAMPAIGN_ID}/search/smoke"
    operation_identity = _smoke_operation_identity(
        manifest_sha256,
        sample_identities,
    )
    operation_marker_path = f"{operation_root}/done.json"
    samples_complete = all(
        _client_done_marker_valid(
            BENCHMARK_OUTPUT_VOLUME,
            f"{sample_relpaths[case.case_id]}/done.json",
            expected_identity=sample_identities[case.case_id],
        )
        for case in cases
    )
    if samples_complete and _client_done_marker_valid(
        BENCHMARK_OUTPUT_VOLUME,
        operation_marker_path,
        expected_identity=operation_identity,
    ):
        summary = _read_volume_json(
            BENCHMARK_OUTPUT_VOLUME,
            f"{operation_root}/summary.json",
        )
        gate_status = (
            "smoke gate passed; measured matrix pending"
            if summary.get("scientific_gate_passed") is True
            else "blocked by smoke scientific gate"
        )
        _publish_campaign_progress(
            stage="search smoke",
            status=gate_status,
            details=[
                "B0, B1, and S3 smoke evidence is available under "
                f"`{operation_root}/`.",
            ],
        )
        return summary | {
            "status": "reused",
            "remote_function_inputs_submitted": 0,
        }

    results: list[dict[str, Any]] = []
    submitted_inputs = 0
    for case in cases:
        marker_path = f"{sample_relpaths[case.case_id]}/done.json"
        reused = _client_done_marker_valid(
            BENCHMARK_OUTPUT_VOLUME,
            marker_path,
            expected_identity=sample_identities[case.case_id],
        )
        if reused:
            result = _read_volume_json(
                BENCHMARK_OUTPUT_VOLUME,
                f"{sample_relpaths[case.case_id]}/metrics.json",
            )
            remote_call_wall_seconds = 0.0
        else:
            submitted_inputs += 1
            remote_started = perf_counter()
            result = benchmark_small_bfd_search.remote(
                query_id=SCREENING_QUERY.query_id,
                case_id=case.case_id,
                sample_id=sample_ids[case.case_id],
                expected_search_identity=search_identities[case.case_id],
                expected_sample_identity=sample_identities[case.case_id],
            )
            remote_call_wall_seconds = perf_counter() - remote_started
        results.append(
            result
            | {
                "sample_kind": "smoke",
                "remote_call_wall_seconds": remote_call_wall_seconds,
                "reused": reused,
            }
        )

    hits_by_case = {
        case.case_id: _read_normalized_hits(sample_relpaths[case.case_id])
        for case in cases
    }
    comparisons = {
        "B0_vs_B1": _compare_normalized_hits(hits_by_case["B1"], hits_by_case["B0"])
        | {"oracle_case": "B1", "candidate_case": "B0", "is_gate": False},
        "S3_vs_B1": _compare_normalized_hits(hits_by_case["B1"], hits_by_case["S3"])
        | {"oracle_case": "B1", "candidate_case": "S3", "is_gate": True},
    }
    scientific_gate_passed = comparisons["S3_vs_B1"]["passed"] is True
    results_bytes = _search_results_parquet(results)
    comparisons_bytes = _json_bytes(comparisons)
    summary_markdown = _smoke_summary_markdown(results, comparisons).encode()
    summary = {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": CAMPAIGN_ID,
        "operation": "search",
        "mode": "smoke",
        "operation_identity": operation_identity,
        "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
        "profile_manifest_sha256": manifest_sha256,
        "scientific_gate_passed": scientific_gate_passed,
        "oracle_case": "B1",
        "gate_case": "S3",
        "sample_paths": sample_relpaths,
        "remote_function_inputs_submitted": submitted_inputs,
        "completed_at": _utc_now(),
        "results_path": f"{operation_root}/results.parquet",
        "comparisons_path": f"{operation_root}/comparisons.json",
    }
    summary_bytes = _json_bytes(summary)
    operation_artifacts = {
        "results.parquet": results_bytes,
        "comparisons.json": comparisons_bytes,
        "summary.md": summary_markdown,
        "summary.json": summary_bytes,
    }
    for relative_path, data in operation_artifacts.items():
        _upload_volume_bytes(
            BENCHMARK_OUTPUT_VOLUME,
            f"{operation_root}/{relative_path}",
            data,
        )
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        operation_marker_path,
        _json_bytes({
            "schema_version": DONE_SCHEMA_VERSION,
            "status": "complete",
            "identity": operation_identity,
            "completed_at": _utc_now(),
            "artifacts": [
                _volume_artifact_record(relative_path, data)
                for relative_path, data in operation_artifacts.items()
            ],
        }),
    )
    _publish_campaign_progress(
        stage="search smoke",
        status=(
            "smoke gate passed; measured matrix pending"
            if scientific_gate_passed
            else "blocked by smoke scientific gate"
        ),
        details=[
            f"Scientific gate: {'PASS' if scientific_gate_passed else 'FAIL'}.",
            f"Submitted {submitted_inputs} remote search inputs.",
        ],
    )
    return summary


def _validate_screening_block_orders() -> None:
    """Require three distinct permutations of every fixed search case."""
    expected = set(MATRIX_CASE_IDS)
    if len(SCREENING_BLOCK_ORDERS) != 3:
        raise ValueError("The screening matrix must contain three blocks")
    if len(set(SCREENING_BLOCK_ORDERS)) != len(SCREENING_BLOCK_ORDERS):
        raise ValueError("Screening block orders must be distinct")
    for order in SCREENING_BLOCK_ORDERS:
        if len(order) != len(expected) or set(order) != expected:
            raise ValueError("Each screening block must contain every case once")


def _focused_sweep_operation_identity(
    manifest_sha256: str,
    sample_identities: dict[str, str],
) -> str:
    """Hash the one-shot sweep plan, policy, profile, and sample identities."""
    if set(sample_identities) != set(FOCUSED_SWEEP_CASE_IDS):
        raise ValueError("Focused sweep sample identities do not match its cases")
    return _sha256_bytes(
        _json_bytes({
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
            "profile_manifest_sha256": manifest_sha256,
            "plan": _build_search_plan("sweep"),
            "sample_identities": sample_identities,
        })
    )


def _matrix_operation_identity(manifest_sha256: str) -> str:
    """Hash the fixed one-shot matrix plan and profile identity."""
    _validate_screening_block_orders()
    return _sha256_bytes(
        _json_bytes({
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
            "profile_manifest_sha256": manifest_sha256,
            "plan": _build_search_plan("matrix"),
        })
    )


def _matrix_sample_id(kind: str, case_id: str, block_index: int) -> str:
    """Return one deterministic screening or stress sample ID."""
    if kind not in {"screen", "stress"}:
        raise ValueError("Matrix sample kind must be 'screen' or 'stress'")
    _search_case(case_id)
    if block_index not in {1, 2, 3}:
        raise ValueError("Matrix block index must be 1, 2, or 3")
    return f"{kind}-{case_id.lower()}-block-{block_index:02d}"


def _matrix_sample_spec(
    profile_scientific_identity: str,
    query: SearchQuery,
    case: SearchCase,
    sample_id: str,
) -> dict[str, str]:
    """Build all immutable identifiers and paths for one matrix sample."""
    search_identity = _search_identity(profile_scientific_identity, query, case)
    sample_identity = _search_sample_identity(
        profile_scientific_identity,
        query,
        case,
        sample_id,
    )
    return {
        "search_identity": search_identity,
        "sample_identity": sample_identity,
        "sample_relpath": _search_sample_relpath(
            query,
            search_identity,
            sample_id,
        ),
    }


def _run_or_reuse_matrix_sample(
    profile_scientific_identity: str,
    query: SearchQuery,
    case: SearchCase,
    sample_id: str,
    *,
    sample_kind: str,
    block_index: int | None,
) -> tuple[dict[str, Any], bool]:
    """Check durable evidence before submitting exactly one remote sample."""
    spec = _matrix_sample_spec(
        profile_scientific_identity,
        query,
        case,
        sample_id,
    )
    marker_path = f"{spec['sample_relpath']}/done.json"
    reused = _client_done_marker_valid(
        BENCHMARK_OUTPUT_VOLUME,
        marker_path,
        expected_identity=spec["sample_identity"],
    )
    if reused:
        result = _read_volume_json(
            BENCHMARK_OUTPUT_VOLUME,
            f"{spec['sample_relpath']}/metrics.json",
        )
        remote_call_wall_seconds = 0.0
    else:
        remote_started = perf_counter()
        result = benchmark_small_bfd_search.remote(
            query_id=query.query_id,
            case_id=case.case_id,
            sample_id=sample_id,
            expected_search_identity=spec["search_identity"],
            expected_sample_identity=spec["sample_identity"],
        )
        remote_call_wall_seconds = perf_counter() - remote_started
    return (
        result
        | {
            "sample_kind": sample_kind,
            "block_index": block_index,
            "remote_call_wall_seconds": remote_call_wall_seconds,
            "reused": reused,
        },
        not reused,
    )


def _load_focused_sweep_reference(
    profile_scientific_identity: str,
    case_id: str,
) -> dict[str, Any]:
    """Load one smoke result after the prerequisite gate validated its marker."""
    if case_id not in FOCUSED_SWEEP_REUSED_CASE_IDS:
        raise ValueError("Focused sweep references must be B1 or S3")
    case = _search_case(case_id)
    sample_id = _focused_sweep_sample_id(case_id)
    spec = _matrix_sample_spec(
        profile_scientific_identity,
        SCREENING_QUERY,
        case,
        sample_id,
    )
    result = _read_volume_json(
        BENCHMARK_OUTPUT_VOLUME,
        f"{spec['sample_relpath']}/metrics.json",
    )
    return result | {
        "sample_kind": "sweep-reference",
        "block_index": None,
        "remote_call_wall_seconds": 0.0,
        "reused": True,
    }


def _require_passing_smoke_gate(
    manifest_sha256: str,
    profile_scientific_identity: str,
) -> None:
    """Block matrix submission unless the exact current smoke gate passed."""
    cases = [_search_case(case_id) for case_id in SMOKE_CASE_IDS]
    sample_identities = {
        case.case_id: _search_sample_identity(
            profile_scientific_identity,
            SCREENING_QUERY,
            case,
            f"smoke-{case.case_id.lower()}",
        )
        for case in cases
    }
    operation_identity = _smoke_operation_identity(
        manifest_sha256,
        sample_identities,
    )
    smoke_root = f"benchmarks/{CAMPAIGN_ID}/search/smoke"
    if not _client_done_marker_valid(
        BENCHMARK_OUTPUT_VOLUME,
        f"{smoke_root}/done.json",
        expected_identity=operation_identity,
    ):
        raise RuntimeError("The current profile does not have a complete smoke gate")
    for case in cases:
        search_identity = _search_identity(
            profile_scientific_identity,
            SCREENING_QUERY,
            case,
        )
        sample_relpath = _search_sample_relpath(
            SCREENING_QUERY,
            search_identity,
            f"smoke-{case.case_id.lower()}",
        )
        if not _client_done_marker_valid(
            BENCHMARK_OUTPUT_VOLUME,
            f"{sample_relpath}/done.json",
            expected_identity=sample_identities[case.case_id],
        ):
            raise RuntimeError(f"Smoke sample {case.case_id} is incomplete")
    summary = _read_volume_json(
        BENCHMARK_OUTPUT_VOLUME,
        f"{smoke_root}/summary.json",
    )
    if summary.get("scientific_gate_passed") is not True:
        raise RuntimeError("The current profile's smoke scientific gate did not pass")


def _metric_float(result: dict[str, Any], key: str) -> float:
    """Read one finite, nonnegative numeric sample metric."""
    value = result.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Search result has invalid numeric metric {key!r}")
    numeric = float(value)
    if numeric < 0 or numeric == float("inf") or numeric != numeric:
        raise ValueError(f"Search result has non-finite metric {key!r}")
    return numeric


def _sample_cost(result: dict[str, Any]) -> float:
    """Read the pinned compute-cost estimate from one sample."""
    estimate = result.get("cost_estimate")
    if not isinstance(estimate, dict):
        raise ValueError("Search result is missing its cost estimate")
    value = estimate.get("estimated_compute_cost_usd")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("Search result has an invalid compute-cost estimate")
    return float(value)


def _median_summary(values: list[float]) -> dict[str, float]:
    """Report median, range, MAD, and three-sample variation."""
    if not values:
        raise ValueError("Cannot summarize an empty metric sample")
    center = median(values)
    absolute_deviations = [abs(value - center) for value in values]
    variation = (max(values) - min(values)) / center if center else 0.0
    return {
        "median": center,
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
        "median_absolute_deviation": median(absolute_deviations),
        "relative_range": variation,
    }


def _case_performance_statistics(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    """Aggregate three measured samples for every represented case."""
    case_ids = sorted({str(result["case"]["case_id"]) for result in results})
    statistics: dict[str, dict[str, object]] = {}
    for case_id in case_ids:
        case_results = [
            result for result in results if result["case"]["case_id"] == case_id
        ]
        if len(case_results) != 3:
            raise ValueError(
                f"Expected three samples for {case_id}, got {len(case_results)}"
            )
        statistics[case_id] = {
            "search_wall_seconds": _median_summary([
                _metric_float(result, "search_wall_seconds") for result in case_results
            ]),
            "sample_wall_seconds": _median_summary([
                _metric_float(result, "sample_wall_seconds") for result in case_results
            ]),
            "remote_call_wall_seconds": _median_summary([
                _metric_float(result, "remote_call_wall_seconds")
                for result in case_results
            ]),
            "estimated_compute_cost_usd": _median_summary([
                _sample_cost(result) for result in case_results
            ]),
            "new_container_samples": sum(
                result["container"].get("container_reused_for_sample") is False
                for result in case_results
            ),
            "reused_container_samples": sum(
                result["container"].get("container_reused_for_sample") is True
                for result in case_results
            ),
        }
    return statistics


def _case_statistic_float(
    statistics: dict[str, dict[str, object]],
    case_id: str,
    metric: str,
    statistic: str,
) -> float:
    """Read one checked numeric value from nested case statistics."""
    summary = statistics[case_id].get(metric)
    if not isinstance(summary, dict):
        raise ValueError(f"Missing {metric} statistics for {case_id}")
    value = summary.get(statistic)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Invalid {metric}.{statistic} for {case_id}")
    return float(value)


def _ranking_float(row: dict[str, object], key: str) -> float:
    """Read one checked numeric candidate-ranking value."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"Invalid candidate ranking value: {key}")
    return float(value)


def _matrix_comparisons(
    results: list[dict[str, Any]],
    candidate_case_ids: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """Compare every candidate and B0 with the same-block B1 oracle."""
    indexed = {
        (int(result["block_index"]), str(result["case"]["case_id"])): result
        for result in results
    }
    hit_tables = {
        key: _read_normalized_hits(str(result["result_path"]))
        for key, result in indexed.items()
    }
    comparisons: dict[str, dict[str, object]] = {}
    for block_index in (1, 2, 3):
        oracle = hit_tables[(block_index, "B1")]
        for case_id in ("B0", *candidate_case_ids):
            comparison_id = f"block-{block_index:02d}-{case_id}-vs-B1"
            comparisons[comparison_id] = _compare_normalized_hits(
                oracle,
                hit_tables[(block_index, case_id)],
            ) | {
                "block_index": block_index,
                "oracle_case": "B1",
                "candidate_case": case_id,
                "is_gate": case_id != "B0",
            }
    return comparisons


def _focused_sweep_comparisons(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, object]]:
    """Compare each one-shot sharded topology with the reused B1 oracle."""
    indexed = {str(result["case"]["case_id"]): result for result in results}
    if set(indexed) != set(FOCUSED_SWEEP_CASE_IDS) or len(indexed) != len(results):
        raise ValueError("Focused sweep results do not contain each case exactly once")
    hit_tables = {
        case_id: _read_normalized_hits(str(result["result_path"]))
        for case_id, result in indexed.items()
    }
    oracle = hit_tables["B1"]
    return {
        f"{case_id}-vs-B1": _compare_normalized_hits(
            oracle,
            hit_tables[case_id],
        )
        | {
            "oracle_case": "B1",
            "candidate_case": case_id,
            "is_gate": True,
        }
        for case_id in FOCUSED_SWEEP_CASE_IDS
        if case_id != "B1"
    }


def _rank_focused_sweep_cases(
    results: list[dict[str, Any]],
    comparisons: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Rank one sample per topology without claiming measured stability."""
    indexed = {str(result["case"]["case_id"]): result for result in results}
    if set(indexed) != set(FOCUSED_SWEEP_CASE_IDS) or len(indexed) != len(results):
        raise ValueError("Focused sweep results do not contain each case exactly once")
    b1_search = _metric_float(indexed["B1"], "search_wall_seconds")
    b1_sample = _metric_float(indexed["B1"], "sample_wall_seconds")
    candidate_rows: list[dict[str, object]] = []
    for case_id in FOCUSED_SWEEP_CASE_IDS:
        if case_id == "B1":
            continue
        result = indexed[case_id]
        case = _search_case(case_id)
        resource = result.get("resource_summary")
        if not isinstance(resource, dict):
            raise ValueError(f"Focused sweep result {case_id} lacks resource metrics")
        comparison = comparisons.get(f"{case_id}-vs-B1")
        if not isinstance(comparison, dict):
            raise ValueError(f"Focused sweep result {case_id} lacks a comparison")
        search_wall = _metric_float(result, "search_wall_seconds")
        sample_wall = _metric_float(result, "sample_wall_seconds")
        search_improvement = 1.0 - search_wall / b1_search
        sample_improvement = 1.0 - sample_wall / b1_sample
        scientific_valid = comparison.get("passed") is True
        meaningful = (
            scientific_valid and search_improvement >= 0.20 and sample_improvement > 0
        )
        candidate_rows.append({
            "case_id": case_id,
            "jackhmmer_n_cpu": case.jackhmmer_n_cpu,
            "active_shards": case.active_shards,
            "aggregate_cpu_slots": case.jackhmmer_n_cpu * case.active_shards,
            "scientific_valid": scientific_valid,
            "search_wall_seconds": search_wall,
            "sample_wall_seconds": sample_wall,
            "search_improvement_vs_B1": search_improvement,
            "sample_improvement_vs_B1": sample_improvement,
            "cpu_core_seconds": _metric_float(resource, "cpu_core_seconds"),
            "peak_interval_cpu_cores": _metric_float(
                resource,
                "peak_interval_cpu_cores",
            ),
            "estimated_compute_cost_usd": _sample_cost(result),
            "meaningful_20_percent_success": meaningful,
        })

    invalid_cases = [
        str(row["case_id"])
        for row in candidate_rows
        if row["scientific_valid"] is False
    ]
    meaningful = [
        row for row in candidate_rows if row["meaningful_20_percent_success"] is True
    ]
    fastest_case_id: str | None = None
    lowest_cost_case_id: str | None = None
    selected_case_ids: list[str] = []
    close_case_ids: list[str] = []
    if meaningful:
        fastest = min(
            meaningful,
            key=lambda row: (
                _ranking_float(row, "search_wall_seconds"),
                _ranking_float(row, "estimated_compute_cost_usd"),
                str(row["case_id"]),
            ),
        )
        fastest_case_id = str(fastest["case_id"])
        fastest_wall = _ranking_float(fastest, "search_wall_seconds")
        near_fastest = [
            row
            for row in meaningful
            if _ranking_float(row, "search_wall_seconds") <= fastest_wall * 1.15
        ]
        lowest_cost = min(
            near_fastest,
            key=lambda row: (
                _ranking_float(row, "estimated_compute_cost_usd"),
                _ranking_float(row, "search_wall_seconds"),
                str(row["case_id"]),
            ),
        )
        lowest_cost_case_id = str(lowest_cost["case_id"])
        selected_case_ids = list(dict.fromkeys((fastest_case_id, lowest_cost_case_id)))
        close_case_ids = sorted(
            str(row["case_id"])
            for row in near_fastest
            if row["case_id"] != fastest_case_id
        )

    if invalid_cases:
        status = "complete_scientific_review_required"
    elif not meaningful:
        status = "complete_no_meaningful_candidate"
    else:
        status = "complete"
    return {
        "status": status,
        "single_sample_only": True,
        "stability_measured": False,
        "scientific_gate_passed": not invalid_cases,
        "invalid_scientific_case_ids": invalid_cases,
        "fastest_case_id": fastest_case_id,
        "lowest_cost_within_15_percent_case_id": lowest_cost_case_id,
        "selected_case_ids": selected_case_ids,
        "close_case_ids": close_case_ids,
        "requires_tiebreak_review": bool(close_case_ids),
        "candidate_rankings": candidate_rows,
        "rules": _build_search_plan("sweep")["selection_policy"],
    }


def _rank_screening_cases(
    results: list[dict[str, Any]],
    comparisons: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Apply scientific, stability, speed, overhead, and cost promotion rules."""
    statistics = _case_performance_statistics(results)
    b1_search = _case_statistic_float(statistics, "B1", "search_wall_seconds", "median")
    b1_sample = _case_statistic_float(statistics, "B1", "sample_wall_seconds", "median")
    candidate_rows: list[dict[str, object]] = []
    for case_id in MATRIX_CASE_IDS:
        case = _search_case(case_id)
        if not case.case_id.startswith("S"):
            continue
        scientific_valid = all(
            comparisons[f"block-{block_index:02d}-{case.case_id}-vs-B1"]["passed"]
            is True
            for block_index in (1, 2, 3)
        )
        search_median = _case_statistic_float(
            statistics, case.case_id, "search_wall_seconds", "median"
        )
        sample_median = _case_statistic_float(
            statistics, case.case_id, "sample_wall_seconds", "median"
        )
        cost_median = _case_statistic_float(
            statistics, case.case_id, "estimated_compute_cost_usd", "median"
        )
        search_improvement = 1.0 - search_median / b1_search
        sample_improvement = 1.0 - sample_median / b1_sample
        stable = (
            _case_statistic_float(
                statistics,
                case.case_id,
                "search_wall_seconds",
                "relative_range",
            )
            <= 0.10
        )
        operational_overhead_preserves_improvement = sample_improvement > 0
        meaningful = (
            scientific_valid
            and stable
            and search_improvement >= 0.20
            and operational_overhead_preserves_improvement
        )
        candidate_rows.append({
            "case_id": case.case_id,
            "scientific_valid": scientific_valid,
            "stable_within_10_percent": stable,
            "search_improvement_vs_B1": search_improvement,
            "sample_improvement_vs_B1": sample_improvement,
            "operational_overhead_preserves_improvement": (
                operational_overhead_preserves_improvement
            ),
            "meaningful_20_percent_success": meaningful,
            "median_search_wall_seconds": search_median,
            "median_sample_wall_seconds": sample_median,
            "median_estimated_compute_cost_usd": cost_median,
        })

    invalid_cases = [
        str(row["case_id"])
        for row in candidate_rows
        if row["scientific_valid"] is False
    ]
    unstable_cases = [
        str(row["case_id"])
        for row in candidate_rows
        if row["scientific_valid"] is True and row["stable_within_10_percent"] is False
    ]
    meaningful = [
        row for row in candidate_rows if row["meaningful_20_percent_success"] is True
    ]
    status = "promoted"
    selected: list[str] = []
    if invalid_cases:
        status = "blocked_scientific_review"
    elif unstable_cases:
        status = "blocked_high_variation"
    elif len(meaningful) < 2:
        status = "complete_insufficient_meaningful_layouts"
    else:
        fastest = min(
            meaningful,
            key=lambda row: (
                _ranking_float(row, "median_search_wall_seconds"),
                _ranking_float(row, "median_estimated_compute_cost_usd"),
                str(row["case_id"]),
            ),
        )
        cost_pool = [
            row
            for row in meaningful
            if row["case_id"] != fastest["case_id"]
            and _ranking_float(row, "median_search_wall_seconds")
            <= _ranking_float(fastest, "median_search_wall_seconds") * 1.15
        ]
        if not cost_pool:
            status = "complete_no_distinct_cost_candidate_within_15_percent"
        else:
            lowest_cost = min(
                cost_pool,
                key=lambda row: (
                    _ranking_float(row, "median_estimated_compute_cost_usd"),
                    _ranking_float(row, "median_search_wall_seconds"),
                    str(row["case_id"]),
                ),
            )
            selected = [str(fastest["case_id"]), str(lowest_cost["case_id"])]
    return {
        "status": status,
        "selected_case_ids": selected,
        "invalid_scientific_case_ids": invalid_cases,
        "unstable_case_ids": unstable_cases,
        "candidate_rankings": candidate_rows,
        "case_statistics": statistics,
        "rules": _build_search_plan("matrix")["performance_gate"],
    }


def _stress_block_orders(
    selected_case_ids: tuple[str, str],
) -> tuple[tuple[str, ...], ...]:
    """Return three distinct deterministic stress permutations."""
    first, second = selected_case_ids
    if first == second or not all(
        case_id.startswith("S") for case_id in (first, second)
    ):
        raise ValueError("Stress promotion requires two distinct sharded cases")
    _search_case(first)
    _search_case(second)
    return (
        ("B0", "B1", first, second),
        (second, "B0", first, "B1"),
        ("B1", first, "B0", second),
    )


def _matrix_sample_records(results: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Build marker-validation references for every measured sample."""
    return [
        {
            "query_id": result["query"]["query_id"],
            "case_id": result["case"]["case_id"],
            "sample_id": result["sample_id"],
            "sample_identity": result["sample_identity"],
            "result_path": result["result_path"],
        }
        for result in results
    ]


def _focused_sweep_summary_markdown(
    summary: dict[str, object],
    rankings: dict[str, object],
) -> str:
    """Render the one-shot topology sweep and its explicit limitations."""
    candidates = rankings.get("candidate_rankings")
    if not isinstance(candidates, list):
        raise ValueError("Focused sweep rankings are missing candidates")
    typed_candidates: list[dict[str, object]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("Focused sweep candidate must be an object")
        typed_candidates.append({str(key): value for key, value in candidate.items()})
    ordered = sorted(
        typed_candidates,
        key=lambda row: (
            not bool(row["scientific_valid"]),
            _ranking_float(row, "search_wall_seconds"),
            _ranking_float(row, "estimated_compute_cost_usd"),
            str(row["case_id"]),
        ),
    )
    selected = summary.get("selected_case_ids")
    if not isinstance(selected, list):
        raise ValueError("Focused sweep summary has invalid selected cases")
    lines = [
        f"# {CAMPAIGN_ID} focused topology sweep",
        "",
        f"Status: **{summary['status']}**",
        f"Scientific gate: **{'PASS' if summary['scientific_gate_passed'] else 'FAIL'}**",
        f"Submitted remote samples: {summary['remote_function_inputs_submitted']}",
        f"Reused smoke samples: {summary['reused_smoke_samples']}",
        f"Fastest case: {rankings['fastest_case_id']}",
        "Lowest-cost case within 15%: "
        f"{rankings['lowest_cost_within_15_percent_case_id']}",
        f"Selected cases: {', '.join(str(item) for item in selected) or 'none'}",
        "",
        "| Rank | Case | Shards | CPU/shard | Slots | Scientific | Search (s) | "
        "vs B1 | Sample (s) | CPU-core-s | Peak cores | Cost (USD) |",
        "| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | "
        "---: | ---: | ---: |",
    ]
    for rank, candidate in enumerate(ordered, start=1):
        lines.append(
            f"| {rank} | {candidate['case_id']} | {candidate['active_shards']} | "
            f"{candidate['jackhmmer_n_cpu']} | {candidate['aggregate_cpu_slots']} | "
            f"{'PASS' if candidate['scientific_valid'] else 'FAIL'} | "
            f"{_ranking_float(candidate, 'search_wall_seconds'):.3f} | "
            f"{_ranking_float(candidate, 'search_improvement_vs_B1'):.1%} | "
            f"{_ranking_float(candidate, 'sample_wall_seconds'):.3f} | "
            f"{_ranking_float(candidate, 'cpu_core_seconds'):.3f} | "
            f"{_ranking_float(candidate, 'peak_interval_cpu_cores'):.3f} | "
            f"{_ranking_float(candidate, 'estimated_compute_cost_usd'):.6f} |"
        )
    lines.extend([
        "",
        "B1 Search Wall Time: "
        f"{_ranking_float(summary, 'oracle_search_wall_seconds'):.3f} s",
        "B1 Sample Wall Time: "
        f"{_ranking_float(summary, 'oracle_sample_wall_seconds'):.3f} s",
        "",
        "Each new topology was measured once. Stability was not measured; cases "
        "within 15% of the fastest require review before a tie-break run.",
        "No stress-query sample was submitted automatically.",
        "",
    ])
    return "\n".join(lines)


def _matrix_summary_markdown(
    summary: dict[str, object],
    rankings: dict[str, object],
) -> str:
    """Render the matrix outcome, gates, variability, and candidate ranks."""
    selected = summary.get("selected_case_ids")
    if not isinstance(selected, list):
        raise ValueError("Matrix summary has invalid selected cases")
    selected_ids = {str(case_id) for case_id in selected}
    selected_text = ", ".join(sorted(selected_ids)) or "none"
    candidate_rows = rankings.get("candidate_rankings")
    statistics = rankings.get("case_statistics")
    if not isinstance(candidate_rows, list) or not isinstance(statistics, dict):
        raise ValueError("Matrix rankings are missing candidate statistics")
    typed_candidates: list[dict[str, object]] = []
    for candidate in candidate_rows:
        if not isinstance(candidate, dict):
            raise ValueError("Candidate ranking row must be an object")
        typed_candidates.append({str(key): value for key, value in candidate.items()})
    typed_statistics: dict[str, dict[str, object]] = {}
    for case_id, case_statistics in statistics.items():
        if not isinstance(case_statistics, dict):
            raise ValueError(f"Invalid matrix statistics for {case_id}")
        typed_statistics[str(case_id)] = {
            str(key): value for key, value in case_statistics.items()
        }

    ordered_candidates = sorted(
        typed_candidates,
        key=lambda row: (
            not bool(row["scientific_valid"]),
            _ranking_float(row, "median_search_wall_seconds"),
            _ranking_float(row, "median_estimated_compute_cost_usd"),
            str(row["case_id"]),
        ),
    )
    campaign_complete = not str(summary["status"]).startswith("blocked_")
    lines = [
        f"# {CAMPAIGN_ID} measured matrix",
        "",
        f"Status: **{summary['status']}**",
        f"Campaign complete: **{'YES' if campaign_complete else 'NO'}**",
        f"Screening samples: {summary['screening_samples']}",
        f"Stress samples: {summary['stress_samples']}",
        f"Submitted remote samples: {summary['remote_function_inputs_submitted']}",
        f"Promoted sharded cases: {selected_text}",
        "",
        "B1 is the scientific and performance oracle. B0 remains descriptive.",
        "No additional diagnostic samples are submitted automatically.",
        "",
        "## Screening candidate ranking",
        "",
        "| Rank | Case | Scientific | Stable | Search median (s) | "
        "Search min-max (s) | Search MAD (s) | vs B1 | Sample median (s) | "
        "Cost median (USD) | 20% gate | Selected |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
        "---: | --- | --- |",
    ]
    for rank, candidate in enumerate(ordered_candidates, start=1):
        case_id = str(candidate["case_id"])
        minimum = _case_statistic_float(
            typed_statistics, case_id, "search_wall_seconds", "minimum"
        )
        maximum = _case_statistic_float(
            typed_statistics, case_id, "search_wall_seconds", "maximum"
        )
        mad = _case_statistic_float(
            typed_statistics,
            case_id,
            "search_wall_seconds",
            "median_absolute_deviation",
        )
        lines.append(
            f"| {rank} | {case_id} | "
            f"{'PASS' if candidate['scientific_valid'] is True else 'FAIL'} | "
            f"{'PASS' if candidate['stable_within_10_percent'] is True else 'FAIL'} | "
            f"{_ranking_float(candidate, 'median_search_wall_seconds'):.3f} | "
            f"{minimum:.3f}-{maximum:.3f} | {mad:.3f} | "
            f"{_ranking_float(candidate, 'search_improvement_vs_B1'):.1%} | "
            f"{_ranking_float(candidate, 'median_sample_wall_seconds'):.3f} | "
            f"{_ranking_float(candidate, 'median_estimated_compute_cost_usd'):.6f} | "
            f"{'PASS' if candidate['meaningful_20_percent_success'] is True else 'FAIL'} | "
            f"{'yes' if case_id in selected_ids else 'no'} |"
        )

    b1_search = _case_statistic_float(
        typed_statistics, "B1", "search_wall_seconds", "median"
    )
    b1_sample = _case_statistic_float(
        typed_statistics, "B1", "sample_wall_seconds", "median"
    )
    b1_remote = _case_statistic_float(
        typed_statistics, "B1", "remote_call_wall_seconds", "median"
    )
    lines.extend([
        "",
        "## Oracle timing",
        "",
        f"B1 median Search Wall Time: {b1_search:.3f} s",
        f"B1 median Sample Wall Time: {b1_sample:.3f} s",
        f"B1 median Remote Call Wall Time: {b1_remote:.3f} s",
        "",
        "The ranking uses median Search Wall Time. Variation is the three-sample "
        "range divided by the median; MAD is reported instead of p95.",
        "",
    ])
    return "\n".join(lines)


def _campaign_results_parquet(
    matrix_results: list[dict[str, Any]] | None = None,
) -> bytes:
    """Combine available scan and search sample indexes at campaign root."""
    import polars as pl

    tables: list[pl.DataFrame] = []
    operation_paths = (
        (
            "storage-scan",
            f"benchmarks/{CAMPAIGN_ID}/storage-scans/results.parquet",
        ),
        ("smoke", f"benchmarks/{CAMPAIGN_ID}/search/smoke/results.parquet"),
        (
            "focused-sweep",
            f"benchmarks/{CAMPAIGN_ID}/search/focused-sweep/results.parquet",
        ),
    )
    for campaign_stage, relative_path in operation_paths:
        try:
            data = _read_volume_bytes(BENCHMARK_OUTPUT_VOLUME, relative_path)
        except FileNotFoundError:
            continue
        tables.append(
            pl.read_parquet(io.BytesIO(data)).with_columns(
                pl.lit(campaign_stage).alias("campaign_stage")
            )
        )

    if matrix_results is None:
        matrix_paths = (
            f"benchmarks/{CAMPAIGN_ID}/search/matrix/all-results.parquet",
            f"benchmarks/{CAMPAIGN_ID}/search/matrix/screening-results.parquet",
        )
        for relative_path in matrix_paths:
            try:
                data = _read_volume_bytes(BENCHMARK_OUTPUT_VOLUME, relative_path)
            except FileNotFoundError:
                continue
            tables.append(
                pl.read_parquet(io.BytesIO(data)).with_columns(
                    pl.lit("matrix").alias("campaign_stage")
                )
            )
            break
    else:
        tables.append(
            pl.read_parquet(
                io.BytesIO(_search_results_parquet(matrix_results))
            ).with_columns(pl.lit("matrix").alias("campaign_stage"))
        )

    if not tables:
        raise ValueError("No completed scan or search samples are available")
    combined = pl.concat(tables, how="diagonal_relaxed")
    sort_columns = [
        column
        for column in (
            "campaign_stage",
            "sample_kind",
            "query_id",
            "case_id",
            "block_index",
            "partition_index",
            "pass",
        )
        if column in combined.columns
    ]
    buffer = io.BytesIO()
    combined.sort(sort_columns).write_parquet(buffer)
    return buffer.getvalue()


def _publish_campaign_progress(
    *,
    stage: str,
    status: str,
    details: list[str],
) -> None:
    """Publish a clearly incomplete campaign snapshot after an operation."""
    campaign_root = f"benchmarks/{CAMPAIGN_ID}"
    try:
        matrix_summary = _read_volume_json(
            BENCHMARK_OUTPUT_VOLUME,
            f"{campaign_root}/search/matrix/summary.json",
        )
    except FileNotFoundError:
        pass
    else:
        if matrix_summary.get("campaign_id") == CAMPAIGN_ID:
            return
    results = _campaign_results_parquet()
    summary = "\n".join([
        f"# {CAMPAIGN_ID}",
        "",
        f"Current stage: **{stage}**",
        f"Status: **{status}**",
        "Campaign complete: **NO**",
        "",
        *details,
        "",
        "This is a progress snapshot, not a final sharding recommendation.",
        "",
    ]).encode()
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{campaign_root}/results.parquet",
        results,
    )
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{campaign_root}/summary.md",
        summary,
    )


def _publish_campaign_matrix_snapshot(
    summary: dict[str, object],
    rankings: dict[str, object],
    matrix_results: list[dict[str, Any]] | None = None,
) -> None:
    """Publish the campaign-wide sample index and measured matrix report."""
    campaign_root = f"benchmarks/{CAMPAIGN_ID}"
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{campaign_root}/results.parquet",
        _campaign_results_parquet(matrix_results),
    )
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{campaign_root}/summary.md",
        _matrix_summary_markdown(summary, rankings).encode(),
    )


def _publish_matrix_artifacts(
    operation_root: str,
    artifacts: dict[str, bytes],
) -> None:
    """Upload a complete set of small matrix index artifacts."""
    for relative_path, data in artifacts.items():
        _upload_volume_bytes(
            BENCHMARK_OUTPUT_VOLUME,
            f"{operation_root}/{relative_path}",
            data,
        )


def _finalize_matrix_operation(
    operation_root: str,
    operation_identity: str,
    artifacts: dict[str, bytes],
) -> None:
    """Publish all matrix indexes, then its completion marker last."""
    _publish_matrix_artifacts(operation_root, artifacts)
    _upload_volume_bytes(
        BENCHMARK_OUTPUT_VOLUME,
        f"{operation_root}/done.json",
        _json_bytes({
            "schema_version": DONE_SCHEMA_VERSION,
            "status": "complete",
            "identity": operation_identity,
            "completed_at": _utc_now(),
            "artifacts": [
                _volume_artifact_record(relative_path, data)
                for relative_path, data in artifacts.items()
            ],
        }),
    )


def _completed_matrix_is_valid(
    operation_root: str,
    operation_identity: str,
) -> dict[str, Any] | None:
    """Validate a complete matrix index and each referenced sample marker."""
    if not _client_done_marker_valid(
        BENCHMARK_OUTPUT_VOLUME,
        f"{operation_root}/done.json",
        expected_identity=operation_identity,
    ):
        return None
    summary = _read_volume_json(
        BENCHMARK_OUTPUT_VOLUME,
        f"{operation_root}/summary.json",
    )
    samples = summary.get("samples")
    if not isinstance(samples, list):
        return None
    for sample in samples:
        if not isinstance(sample, dict):
            return None
        result_path = sample.get("result_path")
        sample_identity = sample.get("sample_identity")
        if not isinstance(result_path, str) or not isinstance(sample_identity, str):
            return None
        if not _client_done_marker_valid(
            BENCHMARK_OUTPUT_VOLUME,
            f"{result_path}/done.json",
            expected_identity=sample_identity,
        ):
            return None
    return summary


def _submit_focused_sweep() -> dict[str, object]:
    """Run only missing one-shot topology samples and publish their ranking."""
    _ensure_campaign_plan_client()
    manifest_relpath = f"{APP_INFO.profile_relpath}/manifest.json"
    manifest_bytes = _read_volume_bytes(SHARDED_MSA_DB_VOLUME, manifest_relpath)
    manifest = orjson.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Profile manifest must be a JSON object")
    _validate_profile_manifest(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    profile_scientific_identity = _profile_scientific_identity(manifest)
    _require_passing_smoke_gate(
        manifest_sha256,
        profile_scientific_identity,
    )

    sample_specs = {
        case_id: _matrix_sample_spec(
            profile_scientific_identity,
            SCREENING_QUERY,
            _search_case(case_id),
            _focused_sweep_sample_id(case_id),
        )
        for case_id in FOCUSED_SWEEP_CASE_IDS
    }
    operation_identity = _focused_sweep_operation_identity(
        manifest_sha256,
        {case_id: spec["sample_identity"] for case_id, spec in sample_specs.items()},
    )
    operation_root = f"benchmarks/{CAMPAIGN_ID}/search/focused-sweep"
    completed = _completed_matrix_is_valid(operation_root, operation_identity)
    if completed is not None:
        return completed | {
            "status": "reused",
            "remote_function_inputs_submitted": 0,
        }

    results = [
        _load_focused_sweep_reference(profile_scientific_identity, case_id)
        for case_id in FOCUSED_SWEEP_REUSED_CASE_IDS
    ]
    submitted_inputs = 0
    for case_id in FOCUSED_SWEEP_NEW_CASE_IDS:
        result, submitted = _run_or_reuse_matrix_sample(
            profile_scientific_identity,
            SCREENING_QUERY,
            _search_case(case_id),
            _focused_sweep_sample_id(case_id),
            sample_kind="focused-sweep",
            block_index=None,
        )
        results.append(result)
        submitted_inputs += submitted

    comparisons = _focused_sweep_comparisons(results)
    rankings = _rank_focused_sweep_cases(results, comparisons)
    selected_case_ids = rankings.get("selected_case_ids")
    if not isinstance(selected_case_ids, list):
        raise ValueError("Focused sweep rankings have invalid selected cases")
    scientific_gate_passed = rankings.get("scientific_gate_passed") is True
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": rankings["status"],
        "campaign_id": CAMPAIGN_ID,
        "operation": "search",
        "mode": "sweep",
        "operation_identity": operation_identity,
        "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
        "profile_manifest_sha256": manifest_sha256,
        "scientific_gate_passed": scientific_gate_passed,
        "oracle_case": "B1",
        "oracle_search_wall_seconds": _metric_float(
            results[0],
            "search_wall_seconds",
        ),
        "oracle_sample_wall_seconds": _metric_float(
            results[0],
            "sample_wall_seconds",
        ),
        "selected_case_ids": selected_case_ids,
        "requires_tiebreak_review": rankings["requires_tiebreak_review"],
        "reused_smoke_samples": len(FOCUSED_SWEEP_REUSED_CASE_IDS),
        "reused_new_samples": len(FOCUSED_SWEEP_NEW_CASE_IDS) - submitted_inputs,
        "new_samples": len(FOCUSED_SWEEP_NEW_CASE_IDS),
        "stress_samples": 0,
        "remote_function_inputs_submitted": submitted_inputs,
        "samples": _matrix_sample_records(results),
        "completed_at": _utc_now(),
        "results_path": f"{operation_root}/results.parquet",
        "comparisons_path": f"{operation_root}/comparisons.json",
        "rankings_path": f"{operation_root}/rankings.json",
    }
    artifacts = {
        "results.parquet": _search_results_parquet(results),
        "comparisons.json": _json_bytes(comparisons),
        "rankings.json": _json_bytes(rankings),
        "summary.json": _json_bytes(summary),
        "summary.md": _focused_sweep_summary_markdown(summary, rankings).encode(),
    }
    _finalize_matrix_operation(
        operation_root,
        operation_identity,
        artifacts,
    )
    return summary


def _submit_search_matrix() -> dict[str, object]:
    """Run the fixed screening matrix and conditionally run its stress matrix."""
    _ensure_campaign_plan_client()
    _validate_screening_block_orders()
    manifest_relpath = f"{APP_INFO.profile_relpath}/manifest.json"
    manifest_bytes = _read_volume_bytes(SHARDED_MSA_DB_VOLUME, manifest_relpath)
    manifest = orjson.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("Profile manifest must be a JSON object")
    _validate_profile_manifest(manifest)
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    profile_scientific_identity = _profile_scientific_identity(manifest)
    _require_passing_smoke_gate(
        manifest_sha256,
        profile_scientific_identity,
    )
    operation_root = f"benchmarks/{CAMPAIGN_ID}/search/matrix"
    operation_identity = _matrix_operation_identity(manifest_sha256)
    completed = _completed_matrix_is_valid(operation_root, operation_identity)
    if completed is not None:
        completed_rankings = _read_volume_json(
            BENCHMARK_OUTPUT_VOLUME,
            f"{operation_root}/rankings.json",
        )
        _publish_campaign_matrix_snapshot(completed, completed_rankings)
        return completed | {
            "status": "reused",
            "remote_function_inputs_submitted": 0,
        }

    screening_results: list[dict[str, Any]] = []
    submitted_inputs = 0
    for block_index, order in enumerate(SCREENING_BLOCK_ORDERS, start=1):
        for case_id in order:
            result, submitted = _run_or_reuse_matrix_sample(
                profile_scientific_identity,
                SCREENING_QUERY,
                _search_case(case_id),
                _matrix_sample_id("screen", case_id, block_index),
                sample_kind="screening",
                block_index=block_index,
            )
            screening_results.append(result)
            submitted_inputs += submitted

    sharded_case_ids = tuple(
        case_id for case_id in MATRIX_CASE_IDS if case_id.startswith("S")
    )
    screening_comparisons = _matrix_comparisons(
        screening_results,
        sharded_case_ids,
    )
    rankings = _rank_screening_cases(screening_results, screening_comparisons)
    screening_summary = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "operation_identity": operation_identity,
        "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
        "status": rankings["status"],
        "sample_count": len(screening_results),
        "remote_function_inputs_submitted": submitted_inputs,
        "samples": _matrix_sample_records(screening_results),
        "completed_at": _utc_now(),
    }
    screening_artifacts = {
        "screening-results.parquet": _search_results_parquet(screening_results),
        "screening-comparisons.json": _json_bytes(screening_comparisons),
        "rankings.json": _json_bytes(rankings),
        "screening-summary.json": _json_bytes(screening_summary),
    }
    _publish_matrix_artifacts(operation_root, screening_artifacts)

    ranking_status = str(rankings["status"])
    if ranking_status.startswith("blocked_"):
        summary: dict[str, object] = {
            "schema_version": 1,
            "status": ranking_status,
            "campaign_id": CAMPAIGN_ID,
            "operation": "search",
            "mode": "matrix",
            "operation_identity": operation_identity,
            "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
            "profile_manifest_sha256": manifest_sha256,
            "screening_samples": len(screening_results),
            "stress_samples": 0,
            "selected_case_ids": [],
            "samples": _matrix_sample_records(screening_results),
            "remote_function_inputs_submitted": submitted_inputs,
            "completed_at": _utc_now(),
            "requires_human_review": True,
        }
        blocked_artifacts = screening_artifacts | {
            "summary.json": _json_bytes(summary),
            "summary.md": _matrix_summary_markdown(summary, rankings).encode(),
        }
        _publish_matrix_artifacts(operation_root, blocked_artifacts)
        _publish_campaign_matrix_snapshot(summary, rankings, screening_results)
        return summary

    selected = rankings.get("selected_case_ids")
    if not isinstance(selected, list):
        raise ValueError("Screening rankings have invalid selected cases")
    if len(selected) != 2:
        summary = {
            "schema_version": 1,
            "status": ranking_status,
            "campaign_id": CAMPAIGN_ID,
            "operation": "search",
            "mode": "matrix",
            "operation_identity": operation_identity,
            "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
            "profile_manifest_sha256": manifest_sha256,
            "screening_samples": len(screening_results),
            "stress_samples": 0,
            "selected_case_ids": selected,
            "samples": _matrix_sample_records(screening_results),
            "remote_function_inputs_submitted": submitted_inputs,
            "completed_at": _utc_now(),
            "requires_human_review": False,
        }
        final_artifacts = screening_artifacts | {
            "summary.json": _json_bytes(summary),
            "summary.md": _matrix_summary_markdown(summary, rankings).encode(),
        }
        _finalize_matrix_operation(
            operation_root,
            operation_identity,
            final_artifacts,
        )
        _publish_campaign_matrix_snapshot(summary, rankings, screening_results)
        return summary

    selected_pair = (str(selected[0]), str(selected[1]))
    stress_results: list[dict[str, Any]] = []
    for block_index, order in enumerate(
        _stress_block_orders(selected_pair),
        start=1,
    ):
        for case_id in order:
            result, submitted = _run_or_reuse_matrix_sample(
                profile_scientific_identity,
                STRESS_QUERY,
                _search_case(case_id),
                _matrix_sample_id("stress", case_id, block_index),
                sample_kind="stress",
                block_index=block_index,
            )
            stress_results.append(result)
            submitted_inputs += submitted

    stress_comparisons = _matrix_comparisons(stress_results, selected_pair)
    stress_gate_passed = all(
        comparison["passed"] is True
        for comparison in stress_comparisons.values()
        if comparison["is_gate"] is True
    )
    stress_statistics = _case_performance_statistics(stress_results)
    final_status = "complete" if stress_gate_passed else "complete_stress_gate_failed"
    all_results = [*screening_results, *stress_results]
    summary = {
        "schema_version": 1,
        "status": final_status,
        "campaign_id": CAMPAIGN_ID,
        "operation": "search",
        "mode": "matrix",
        "operation_identity": operation_identity,
        "scientific_comparison_policy": SCIENTIFIC_COMPARISON_POLICY,
        "profile_manifest_sha256": manifest_sha256,
        "screening_samples": len(screening_results),
        "stress_samples": len(stress_results),
        "selected_case_ids": list(selected_pair),
        "stress_scientific_gate_passed": stress_gate_passed,
        "stress_statistics": stress_statistics,
        "samples": _matrix_sample_records(all_results),
        "remote_function_inputs_submitted": submitted_inputs,
        "completed_at": _utc_now(),
        "requires_human_review": not stress_gate_passed,
    }
    final_artifacts = screening_artifacts | {
        "stress-results.parquet": _search_results_parquet(stress_results),
        "stress-comparisons.json": _json_bytes(stress_comparisons),
        "stress-statistics.json": _json_bytes(stress_statistics),
        "all-results.parquet": _search_results_parquet(all_results),
        "summary.json": _json_bytes(summary),
        "summary.md": _matrix_summary_markdown(summary, rankings).encode(),
    }
    _finalize_matrix_operation(
        operation_root,
        operation_identity,
        final_artifacts,
    )
    _publish_campaign_matrix_snapshot(summary, rankings, all_results)
    return summary


def _default_profile_query(spec: DatabaseProfileSpec) -> str:
    """Return the fixed representative query for one polymer class."""
    if spec.polymer == "protein":
        return PEMBROLIZUMAB_VH_SEQUENCE
    if spec.polymer == "rna":
        return RNA_ORACLE_SEQUENCE
    raise RuntimeError(f"Unsupported polymer type: {spec.polymer}")


def _validate_profile_query(
    spec: DatabaseProfileSpec,
    sequence: str,
) -> str:
    """Validate a query before it reaches an upstream command wrapper."""
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string")
    if not 1 <= len(sequence) <= 10_000:
        raise ValueError("sequence length must be between 1 and 10,000")
    pattern = r"[A-Z]+" if spec.polymer == "protein" else r"[ACGU]+"
    if re.fullmatch(pattern, sequence) is None:
        raise ValueError(f"sequence contains invalid {spec.polymer} characters")
    return sequence


def _production_scientific_search_parameters(
    spec: DatabaseProfileSpec,
) -> dict[str, object]:
    """Return result-affecting parameters from the pinned data pipeline."""
    common: dict[str, object] = {
        "database_id": spec.database_id,
        "polymer": spec.polymer,
        "max_sequences": spec.max_sequences,
        "z_value": spec.search_space_value,
    }
    if spec.polymer == "protein":
        return common | {
            "tool": "jackhmmer",
            "n_iter": 1,
            "e_value": JACKHMMER_E_VALUE,
            "dom_z_value": spec.search_space_value,
            "filter_f1": JACKHMMER_FILTER_F1,
            "filter_f2": JACKHMMER_FILTER_F2,
            "filter_f3": JACKHMMER_FILTER_F3,
        }
    return common | {
        "tool": "nhmmer",
        "e_value": 1e-3,
        "filter_f3": 1e-5,
        "alphabet": "rna",
        "short_sequence_filter_f3": 0.02,
    }


def _production_search_parameters(
    spec: DatabaseProfileSpec,
    layout: str = "sharded",
) -> dict[str, object]:
    """Return scientific and operational parameters for evidence."""
    monolith = layout == "monolith"
    return _production_scientific_search_parameters(spec) | {
        "n_cpu": ORACLE_MONOLITH_N_CPU if monolith else PRODUCTION_SEARCH_N_CPU,
        "max_parallel_shards": (
            1 if monolith else PRODUCTION_SEARCH_MAX_PARALLEL_SHARDS
        ),
    }


def _production_search_plan(
    database_id: str,
    sequence: str,
) -> dict[str, object]:
    """Build one side-effect-free fixed profile-search plan."""
    spec = _database_profile_spec(database_id)
    query = _validate_profile_query(
        spec,
        sequence or _default_profile_query(spec),
    )
    return {
        "operation": "search-profile",
        "remote_calls": 1,
        "database_id": spec.database_id,
        "profile_id": spec.profile_id,
        "sequence_length": len(query),
        "sequence_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "parameters": _production_search_parameters(spec),
        "resources": {
            "cpu": [0.125, 32.125],
            "memory_mib": [1024, 131_072],
            "timeout_seconds": CONF.timeout,
        },
        "input_volume": {
            "name": SHARDED_DB_VOLUME_NAME,
            "mount": "read-only",
        },
        "output_volume": OUTPUT_VOLUME_NAME,
        "existing_result_policy": "validate-and-reuse",
    }


def _profile_search_identity(
    spec: DatabaseProfileSpec,
    sequence: str,
    manifest_sha256: str,
    layout: str,
) -> str:
    """Hash the scientific identity of one sequence-by-profile search."""
    return _sha256_bytes(
        _json_bytes({
            "schema_version": 1,
            "profile_id": spec.profile_id,
            "profile_manifest_sha256": manifest_sha256,
            "layout": layout,
            "sequence": sequence,
            "parameters": _production_scientific_search_parameters(spec),
            "alphafold_commit": CONF.repo_commit_hash,
            "hmmer_version": HMMER_VERSION,
            "jackhmmer_patch_sha256": JACKHMMER_PATCH_SHA256,
        })
    )


def _profile_search_relpath(
    spec: DatabaseProfileSpec,
    sequence: str,
    search_identity: str,
    layout: str,
) -> str:
    """Return one deterministic experimental raw-result directory."""
    sequence_hash = hashlib.sha256(sequence.encode()).hexdigest()
    polymer_dir = "Protein" if spec.polymer == "protein" else "RNA"
    return (
        "production-candidates/searches/"
        f"{polymer_dir}/{sequence_hash[:2]}/{sequence_hash}/raw-msa/"
        f"{spec.database_id}/{layout}/{search_identity}"
    )


def _load_reusable_profile_search(
    result_root: Path,
    search_identity: str,
) -> dict[str, object] | None:
    """Validate and return one completed experimental search result."""
    done_path = result_root / "done.json"
    if not done_path.is_file():
        return None
    done = _load_json_object(done_path)
    if (
        done.get("schema_version") != 1
        or done.get("status") != "complete"
        or done.get("search_identity") != search_identity
    ):
        return None
    result_record = done.get("result")
    hits_record = done.get("hits")
    if not isinstance(result_record, dict) or not isinstance(hits_record, dict):
        return None
    result_path = result_root / "result.a3m"
    hits_path = result_root / "hits.parquet"
    if not result_path.is_file() or not hits_path.is_file():
        return None
    result_bytes = result_path.read_bytes()
    hits_bytes = hits_path.read_bytes()
    if (
        result_record.get("size_bytes") != len(result_bytes)
        or result_record.get("sha256") != hashlib.sha256(result_bytes).hexdigest()
        or hits_record.get("size_bytes") != len(hits_bytes)
        or hits_record.get("sha256") != hashlib.sha256(hits_bytes).hexdigest()
    ):
        return None
    metrics_path = result_root / "metrics.json"
    if not metrics_path.is_file():
        return None
    metrics = _load_json_object(metrics_path)
    return metrics | {"status": "reused"}


def _validate_profile_search_layout(layout: str) -> str:
    """Validate the two fixed scientific-oracle database layouts."""
    if layout not in {"monolith", "sharded"}:
        raise ValueError("layout must be 'monolith' or 'sharded'")
    return layout


def _normalized_rna_hit_rows(
    merged_a3m: str,
    raw_tblouts: list[tuple[str, str]],
) -> list[dict[str, object]]:
    """Normalize Nhmmer A3M/tblout evidence using its coordinate hit ID."""
    latest_by_target: dict[str, dict[str, object]] = {}
    occurrences: dict[str, list[dict[str, object]]] = {}
    for source, tblout in raw_tblouts:
        for line_number, line in enumerate(tblout.splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split(maxsplit=15)
            if len(fields) < 14:
                raise ValueError(
                    f"Invalid Nhmmer tblout line {line_number} from {source}: {line!r}"
                )
            target_id = f"{fields[0]}/{fields[6]}-{fields[7]}"
            entry: dict[str, object] = {
                "target_id": target_id,
                "e_value_text": fields[12],
                "bit_score_text": fields[13],
                "e_value": float(fields[12]),
                "bit_score": float(fields[13]),
                "source": source,
                "line": line,
            }
            occurrences.setdefault(target_id, []).append(entry)
            latest_by_target[target_id] = entry

    rows: list[dict[str, object]] = []
    for ordinal, (sequence, description) in enumerate(
        _parse_a3m_records(merged_a3m)[1:],
        start=1,
    ):
        target_id = description.partition(" ")[0]
        hit_entry = latest_by_target.get(target_id)
        if hit_entry is None:
            raise ValueError(f"Merged RNA A3M target has no tblout row: {target_id}")
        target_occurrences = occurrences[target_id]
        occurrence_sources = [str(item["source"]) for item in target_occurrences]
        normalized_sequence = _normalize_a3m_sequence(sequence)
        rows.append({
            "ordinal": ordinal,
            "target_id": target_id,
            "description": description,
            "aligned_sequence": sequence,
            "normalized_sequence": normalized_sequence,
            "normalized_sequence_sha256": _sha256_bytes(normalized_sequence.encode()),
            "e_value": hit_entry["e_value"],
            "e_value_text": hit_entry["e_value_text"],
            "bit_score": hit_entry["bit_score"],
            "bit_score_text": hit_entry["bit_score_text"],
            "tblout_source": hit_entry["source"],
            "tblout_line": hit_entry["line"],
            "raw_occurrence_count": len(target_occurrences),
            "raw_occurrence_sources": ",".join(occurrence_sources),
            "cross_shard_duplicate": len(set(occurrence_sources)) > 1,
        })
    return rows


def _execute_profile_database_search(
    spec: DatabaseProfileSpec,
    sequence: str,
    layout: str,
    profile_root: Path,
) -> tuple[str, list[tuple[str, str]]]:
    """Run a monolithic or sharded pinned search while retaining tblout."""
    from importlib import import_module

    selected_layout = _validate_profile_search_layout(layout)
    if selected_layout == "monolith":
        database_path = str(Path(APP_INFO.source_db_dir) / spec.source_filename)
        search_paths = (Path(database_path),)
    else:
        database_path = (
            profile_root / "shards" / spec.source_filename
        ).as_posix() + f"@{spec.shard_count}"
        search_paths = tuple(
            profile_root / "shards" / name for name in _production_shard_names(spec)
        )
    n_cpu = (
        ORACLE_MONOLITH_N_CPU
        if selected_layout == "monolith"
        else PRODUCTION_SEARCH_N_CPU
    )

    if spec.polymer == "protein":
        module = import_module("alphafold3.data.tools.jackhmmer")
        tool = module.Jackhmmer(
            binary_path=JACKHMMER_BINARY_PATH,
            database_path=database_path,
            n_cpu=n_cpu,
            n_iter=1,
            e_value=JACKHMMER_E_VALUE,
            z_value=spec.search_space_value,
            dom_z_value=spec.search_space_value,
            max_sequences=spec.max_sequences,
            filter_f1=JACKHMMER_FILTER_F1,
            filter_f2=JACKHMMER_FILTER_F2,
            filter_f3=JACKHMMER_FILTER_F3,
            max_threads=PRODUCTION_SEARCH_MAX_PARALLEL_SHARDS,
        )
    else:
        module = import_module("alphafold3.data.tools.nhmmer")
        tool = module.Nhmmer(
            binary_path=NHMMER_BINARY_PATH,
            hmmalign_binary_path=HMMALIGN_BINARY_PATH,
            hmmbuild_binary_path=HMMBUILD_BINARY_PATH,
            database_path=database_path,
            n_cpu=n_cpu,
            e_value=1e-3,
            z_value=spec.search_space_value,
            max_sequences=spec.max_sequences,
            filter_f3=1e-5,
            alphabet="rna",
            max_threads=PRODUCTION_SEARCH_MAX_PARALLEL_SHARDS,
        )

    global_temp_dir = tempfile.mkdtemp(
        prefix=f"af3-{spec.database_id}-{selected_layout}-",
        dir=PRODUCTION_SCRATCH_ROOT,
    )

    def query_one(search_path: Path) -> Any:
        return tool._query_db_shard(  # noqa: SLF001
            target_sequence=sequence,
            db_shard_path=str(search_path),
            get_tblout=True,
            global_temp_dir=global_temp_dir,
        )

    try:
        if selected_layout == "monolith":
            results = (query_one(search_paths[0]),)
        else:
            with ThreadPoolExecutor(
                max_workers=PRODUCTION_SEARCH_MAX_PARALLEL_SHARDS
            ) as executor:
                results = tuple(executor.map(query_one, search_paths))
    finally:
        shutil.rmtree(global_temp_dir, ignore_errors=True)

    raw_tblouts: list[tuple[str, str]] = []
    for search_path, result in zip(search_paths, results, strict=True):
        if result.tblout is None:
            raise ValueError(f"{search_path.name} search did not return tblout")
        raw_tblouts.append((search_path.name, result.tblout))
    if selected_layout == "monolith":
        merged = results[0]
    elif spec.polymer == "protein":
        merged = module._merge_jackhmmer_results(  # noqa: SLF001
            results,
            spec.max_sequences,
        )
    else:
        merged = module._merge_nhmmer_results(  # noqa: SLF001
            results,
            spec.max_sequences,
        )
    return merged.a3m, raw_tblouts


def _run_profile_search(
    database_id: str,
    sequence: str,
    layout: str,
) -> dict[str, object]:
    """Search one fixed published profile using the pinned upstream wrapper."""
    spec = _database_profile_spec(database_id)
    query = _validate_profile_query(spec, sequence)
    selected_layout = _validate_profile_search_layout(layout)
    sharded_root = Path(APP_INFO.sharded_db_dir)
    output_root = Path(APP_INFO.output_dir)
    profile_root = _production_profile_root(sharded_root, spec)
    if selected_layout == "monolith":
        SOURCE_MSA_DB_VOLUME.reload()
    SHARDED_MSA_DB_VOLUME.reload()
    BENCHMARK_OUTPUT_VOLUME.reload()
    manifest_path = profile_root / "manifest.json"
    _require_regular_file(manifest_path)
    manifest = _load_json_object(manifest_path)
    _validate_production_profile_manifest(manifest, spec)
    manifest_sha256 = _sha256_file(manifest_path)
    search_identity = _profile_search_identity(
        spec,
        query,
        manifest_sha256,
        selected_layout,
    )
    result_root = output_root / _profile_search_relpath(
        spec,
        query,
        search_identity,
        selected_layout,
    )
    reusable = _load_reusable_profile_search(result_root, search_identity)
    if reusable is not None:
        return reusable

    log_path = result_root / "run.log"
    result_root.mkdir(parents=True, exist_ok=True)
    _append_log(
        log_path,
        f"Searching {spec.profile_id} {selected_layout} for a "
        f"{len(query)}-residue query",
    )
    started = perf_counter()
    try:
        a3m, raw_tblouts = _execute_profile_database_search(
            spec,
            query,
            selected_layout,
            profile_root,
        )
        if not isinstance(a3m, str) or not a3m.startswith(">query\n"):
            raise ValueError("Pinned MSA wrapper returned an invalid A3M")
        hit_rows = (
            _normalized_hit_rows(a3m, raw_tblouts)
            if spec.polymer == "protein"
            else _normalized_rna_hit_rows(a3m, raw_tblouts)
        )
        result_bytes = a3m.encode()
        hits_bytes = _normalized_hits_parquet(hit_rows)
        elapsed_seconds = perf_counter() - started
        result_record = {
            "path": "result.a3m",
            "size_bytes": len(result_bytes),
            "sha256": hashlib.sha256(result_bytes).hexdigest(),
        }
        hits_record = {
            "path": "hits.parquet",
            "size_bytes": len(hits_bytes),
            "sha256": hashlib.sha256(hits_bytes).hexdigest(),
        }
        metrics: dict[str, object] = {
            "schema_version": 1,
            "status": "published",
            "database_id": spec.database_id,
            "profile_id": spec.profile_id,
            "profile_manifest_sha256": manifest_sha256,
            "layout": selected_layout,
            "search_identity": search_identity,
            "sequence_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "sequence_length": len(query),
            "elapsed_seconds": elapsed_seconds,
            "parameters": _production_search_parameters(
                spec,
                selected_layout,
            ),
            "hit_rows": len(hit_rows),
            "result": result_record,
            "hits": hits_record,
            "result_path": str(result_root / "result.a3m"),
            "hits_path": str(result_root / "hits.parquet"),
        }
        _append_log(
            log_path,
            f"Completed {selected_layout} search with {len(hit_rows)} hit rows "
            f"in {elapsed_seconds:.3f} seconds",
        )
        _write_bytes_atomic(result_root / "result.a3m", result_bytes)
        _write_bytes_atomic(result_root / "hits.parquet", hits_bytes)
        _write_json_atomic(result_root / "metrics.json", metrics)
        BENCHMARK_OUTPUT_VOLUME.commit()
        _write_json_atomic(
            result_root / "done.json",
            {
                "schema_version": 1,
                "status": "complete",
                "search_identity": search_identity,
                "completed_at": _utc_now(),
                "result": result_record,
                "hits": hits_record,
            },
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        return metrics
    except Exception as exc:
        _append_log(
            log_path,
            f"Failed with {type(exc).__name__}: {exc}",
        )
        _write_json_atomic(
            result_root / "failure.json",
            {
                "failed_at": _utc_now(),
                "database_id": spec.database_id,
                "profile_id": spec.profile_id,
                "layout": selected_layout,
                "search_identity": search_identity,
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        BENCHMARK_OUTPUT_VOLUME.commit()
        raise


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 131_072),
    timeout=CONF.timeout,
    max_containers=4,
    volumes={
        APP_INFO.sharded_db_dir: SHARDED_MSA_DB_VOLUME.with_mount_options(
            read_only=True
        ),
        APP_INFO.output_dir: BENCHMARK_OUTPUT_VOLUME,
    },
)
def search_database_profile(
    database_id: str,
    sequence: str,
) -> dict[str, object]:
    """Search one published experimental profile without revalidating shards."""
    return _run_profile_search(database_id, sequence, "sharded")


@app.function(
    cpu=(0.125, 32.125),
    memory=(1024, 131_072),
    timeout=CONF.timeout,
    max_containers=4,
    volumes={
        APP_INFO.source_db_dir: SOURCE_MSA_DB_VOLUME.with_mount_options(read_only=True),
        APP_INFO.sharded_db_dir: SHARDED_MSA_DB_VOLUME.with_mount_options(
            read_only=True
        ),
        APP_INFO.output_dir: BENCHMARK_OUTPUT_VOLUME,
    },
)
def search_unsharded_database_oracle(
    database_id: str,
    sequence: str,
) -> dict[str, object]:
    """Run the matching monolithic search for current-pipeline evidence."""
    return _run_profile_search(database_id, sequence, "monolith")


@dataclass(frozen=True)
class MsaOracleCase:
    """One fixed current-pipeline versus sharded scientific gate."""

    case_id: str
    polymer: str
    sequence: str
    database_ids: tuple[str, ...]


MSA_ORACLE_CASES = (
    MsaOracleCase(
        case_id="pembrolizumab-vh",
        polymer="protein",
        sequence=PEMBROLIZUMAB_VH_SEQUENCE,
        database_ids=("uniref90", "small_bfd", "mgnify", "uniprot"),
    ),
    MsaOracleCase(
        case_id="upstream-rna-25nt",
        polymer="rna",
        sequence=RNA_ORACLE_SEQUENCE,
        database_ids=("rfam", "rnacentral", "ntrna"),
    ),
)


def _msa_oracle_case(case_id: str) -> MsaOracleCase:
    """Resolve one fixed scientific oracle case."""
    for case in MSA_ORACLE_CASES:
        if case.case_id == case_id:
            return case
    choices = ", ".join(case.case_id for case in MSA_ORACLE_CASES)
    raise ValueError(f"Unknown oracle_case {case_id!r}; expected one of {choices}")


def _msa_oracle_plan(case_id: str) -> dict[str, object]:
    """Build a side-effect-free monolith-versus-sharded validation plan."""
    case = _msa_oracle_case(case_id)
    return {
        "operation": "validate-oracle",
        "case_id": case.case_id,
        "polymer": case.polymer,
        "sequence_length": len(case.sequence),
        "sequence_sha256": hashlib.sha256(case.sequence.encode()).hexdigest(),
        "database_ids": list(case.database_ids),
        "remote_calls": len(case.database_ids) * 2 + 1,
        "batches": [
            {
                "layout": "monolith",
                "workers": len(case.database_ids),
                "maximum_concurrent_workers": 4,
            },
            {
                "layout": "sharded",
                "workers": len(case.database_ids),
                "maximum_concurrent_workers": 4,
            },
            {
                "operation": "assemble-and-compare",
                "workers": 1,
            },
        ],
        "search_resources_per_worker": {
            "cpu": [0.125, 32.125],
            "memory_mib": [1024, 131_072],
            "monolith_hmmer_cpus": ORACLE_MONOLITH_N_CPU,
            "sharded_hmmer_cpus": PRODUCTION_SEARCH_N_CPU,
            "sharded_active_shards": PRODUCTION_SEARCH_MAX_PARALLEL_SHARDS,
        },
        "scientific_gate": {
            "per_database": [
                "hit identities",
                "E-values",
                "bit scores",
                "aligned sequences",
            ],
            "final_fields": (
                ["unpairedMsa", "pairedMsa"]
                if case.polymer == "protein"
                else ["unpairedMsa"]
            ),
            "equal_score_permutations": "equivalent",
            "rna_requires_non_query_monolithic_hit": case.polymer == "rna",
        },
        "inference": False,
        "template_search": False,
        "output_volume": OUTPUT_VOLUME_NAME,
    }


def _read_profile_search_evidence(
    result: dict[str, object],
    *,
    expected_spec: DatabaseProfileSpec,
    expected_layout: str,
) -> tuple[str, list[dict[str, object]]]:
    """Read one worker's result and normalized hits from the output Volume."""
    if result.get("database_id") != expected_spec.database_id:
        raise ValueError("Oracle search result database ID differs")
    if result.get("profile_id") != expected_spec.profile_id:
        raise ValueError("Oracle search result Profile ID differs")
    if result.get("layout") != expected_layout:
        raise ValueError("Oracle search result layout differs")
    search_identity = result.get("search_identity")
    result_path_value = result.get("result_path")
    hits_path_value = result.get("hits_path")
    if (
        not isinstance(search_identity, str)
        or not isinstance(result_path_value, str)
        or not isinstance(hits_path_value, str)
    ):
        raise ValueError("Oracle search result paths are invalid")

    output_root = Path(APP_INFO.output_dir).resolve()
    result_path = Path(result_path_value).resolve()
    hits_path = Path(hits_path_value).resolve()
    if not result_path.is_relative_to(output_root) or not hits_path.is_relative_to(
        output_root
    ):
        raise ValueError("Oracle search result escapes the output Volume")
    _require_regular_file(result_path)
    _require_regular_file(hits_path)
    a3m = result_path.read_text(encoding="utf-8")

    import polars as pl

    hit_rows = pl.read_parquet(hits_path).to_dicts()
    return a3m, hit_rows


def _compare_profile_hit_rows(
    spec: DatabaseProfileSpec,
    oracle_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Compare one database while allowing only characterized tail effects."""
    oracle_unique = _unique_hit_rows(oracle_rows)
    candidate_unique = _unique_hit_rows(candidate_rows)
    oracle_ids = [str(row["target_id"]) for row in oracle_unique]
    candidate_ids = [str(row["target_id"]) for row in candidate_unique]
    top_width = min(100, max(len(oracle_ids), len(candidate_ids)))
    top_hits_exact = oracle_ids[:top_width] == candidate_ids[:top_width]
    top_hits_tie_equivalent = _top_hits_tie_equivalent(
        oracle_unique[:top_width],
        candidate_unique[:top_width],
    )
    oracle_by_id = {str(row["target_id"]): row for row in oracle_unique}
    candidate_by_id = {str(row["target_id"]): row for row in candidate_unique}
    shared_ids = set(oracle_by_id) & set(candidate_by_id)
    score_mismatches = sorted(
        target_id
        for target_id in shared_ids
        if (
            oracle_by_id[target_id]["e_value_text"],
            oracle_by_id[target_id]["bit_score_text"],
        )
        != (
            candidate_by_id[target_id]["e_value_text"],
            candidate_by_id[target_id]["bit_score_text"],
        )
    )
    sequence_mismatches = sorted(
        target_id
        for target_id in shared_ids
        if oracle_by_id[target_id]["normalized_sequence_sha256"]
        != candidate_by_id[target_id]["normalized_sequence_sha256"]
    )
    oracle_set = set(oracle_ids)
    candidate_set = set(candidate_ids)
    union = oracle_set | candidate_set
    overlap = len(oracle_set & candidate_set) / len(union) if union else 1.0
    oracle_only = [
        target_id for target_id in oracle_ids if target_id not in candidate_set
    ]
    candidate_only = [
        target_id for target_id in candidate_ids if target_id not in oracle_set
    ]
    candidate_duplicate_rows = len(candidate_rows) - len(candidate_unique)
    both_reached_limit = (
        len(oracle_rows) == spec.max_sequences - 1
        and len(candidate_rows) == spec.max_sequences - 1
    )
    oracle_positions = {
        target_id: position for position, target_id in enumerate(oracle_ids, start=1)
    }
    oracle_tail_start = len(oracle_ids) - candidate_duplicate_rows + 1
    duplicate_tail_only = (
        bool(oracle_only)
        and not candidate_only
        and both_reached_limit
        and 0 < candidate_duplicate_rows <= spec.shard_count
        and all(
            oracle_positions[target_id] >= oracle_tail_start
            for target_id in oracle_only
        )
        and any(row.get("cross_shard_duplicate") is True for row in candidate_rows)
    )
    identifiers_equivalent = (
        not oracle_only and not candidate_only
    ) or duplicate_tail_only
    full_order_tie_equivalent = (
        not oracle_only
        and not candidate_only
        and _top_hits_tie_equivalent(oracle_unique, candidate_unique)
    )
    order_equivalent = (
        full_order_tie_equivalent
        if not oracle_only and not candidate_only
        else top_hits_tie_equivalent and duplicate_tail_only
    )
    passed = (
        order_equivalent
        and not score_mismatches
        and not sequence_mismatches
        and overlap >= 0.99
        and identifiers_equivalent
    )
    return {
        "database_id": spec.database_id,
        "passed": passed,
        "top_comparison_width": top_width,
        "top_hits_exact": top_hits_exact,
        "top_hits_tie_equivalent": top_hits_tie_equivalent,
        "top_order_differs_only_within_ties": (
            top_hits_tie_equivalent and not top_hits_exact
        ),
        "full_order_tie_equivalent": full_order_tie_equivalent,
        "oracle_hit_rows": len(oracle_rows),
        "candidate_hit_rows": len(candidate_rows),
        "oracle_unique_hits": len(oracle_ids),
        "candidate_unique_hits": len(candidate_ids),
        "full_unique_hit_jaccard": overlap,
        "score_mismatch_count": len(score_mismatches),
        "score_mismatch_ids": score_mismatches,
        "sequence_mismatch_count": len(sequence_mismatches),
        "sequence_mismatch_ids": sequence_mismatches,
        "oracle_only_ids": oracle_only,
        "candidate_only_ids": candidate_only,
        "candidate_duplicate_hit_rows": candidate_duplicate_rows,
        "both_results_reached_hit_limit": both_reached_limit,
        "differences_characterized_as_duplicate_tail": duplicate_tail_only,
    }


def _compare_final_a3m(oracle: str, candidate: str) -> dict[str, object]:
    """Compare final upstream-assembled A3Ms modulo record permutation."""
    from collections import Counter

    oracle_records = _parse_a3m_records(oracle)
    candidate_records = _parse_a3m_records(candidate)
    query_equal = oracle_records[0] == candidate_records[0]
    aligned_record_multiset_equal = Counter(oracle_records[1:]) == Counter(
        candidate_records[1:]
    )
    return {
        "passed": query_equal and aligned_record_multiset_equal,
        "byte_exact": oracle == candidate,
        "query_equal": query_equal,
        "aligned_record_multiset_equal": aligned_record_multiset_equal,
        "oracle_depth": len(oracle_records),
        "candidate_depth": len(candidate_records),
        "order_differs_only": (
            oracle != candidate and query_equal and aligned_record_multiset_equal
        ),
    }


def _assert_pinned_msa_assembly_contract() -> dict[str, str]:
    """Bind the local assembly adapter to the pinned upstream function bodies."""
    import inspect
    from importlib import import_module

    pipeline = import_module("alphafold3.data.pipeline")
    msa_module = import_module("alphafold3.data.msa")
    protein_source = inspect.getsource(
        pipeline._get_protein_msa_and_templates  # noqa: SLF001
    )
    rna_source = inspect.getsource(pipeline._get_rna_msa)  # noqa: SLF001
    compact_protein = re.sub(r"\s+", "", protein_source)
    compact_rna = re.sub(r"\s+", "", rna_source)
    required_protein = (
        "msas=[uniref90_msa,small_bfd_msa,mgnify_msa],deduplicate=True",
        "msas=[uniprot_msa],deduplicate=False",
    )
    required_rna = "msas=[rfam_msa,rnacentral_msa,nt_rna_msa],deduplicate=True"
    if not all(pattern in compact_protein for pattern in required_protein):
        raise RuntimeError("Pinned protein MSA assembly contract changed")
    if required_rna not in compact_rna:
        raise RuntimeError("Pinned RNA MSA assembly contract changed")
    get_msa_source = inspect.getsource(msa_module.get_msa)
    deduplicate_parameter = inspect.signature(msa_module.get_msa).parameters.get(
        "deduplicate"
    )
    if (
        deduplicate_parameter is None
        or deduplicate_parameter.default is not False
        or "deduplicate=deduplicate" not in re.sub(r"\s+", "", get_msa_source)
    ):
        raise RuntimeError("Pinned per-database MSA deduplication contract changed")
    return {
        "protein_function_sha256": hashlib.sha256(protein_source.encode()).hexdigest(),
        "rna_function_sha256": hashlib.sha256(rna_source.encode()).hexdigest(),
        "get_msa_function_sha256": hashlib.sha256(get_msa_source.encode()).hexdigest(),
    }


def _assemble_current_pipeline_msas(
    case: MsaOracleCase,
    database_a3ms: dict[str, str],
) -> dict[str, str]:
    """Apply the exact pinned upstream database order and deduplication."""
    from importlib import import_module

    msa = import_module("alphafold3.data.msa")
    mmcif_names = import_module("alphafold3.constants.mmcif_names")
    if case.polymer == "protein":
        unpaired = msa.Msa.from_multiple_a3ms(
            a3ms=[
                database_a3ms["uniref90"],
                database_a3ms["small_bfd"],
                database_a3ms["mgnify"],
            ],
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            deduplicate=True,
        ).to_a3m()
        paired = msa.Msa.from_multiple_a3ms(
            a3ms=[database_a3ms["uniprot"]],
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            deduplicate=False,
        ).to_a3m()
        return {"unpairedMsa": unpaired, "pairedMsa": paired}
    unpaired = msa.Msa.from_multiple_a3ms(
        a3ms=[
            database_a3ms["rfam"],
            database_a3ms["rnacentral"],
            database_a3ms["ntrna"],
        ],
        chain_poly_type=mmcif_names.RNA_CHAIN,
        deduplicate=True,
    ).to_a3m()
    return {"unpairedMsa": unpaired}


def _oracle_fold_input(
    case: MsaOracleCase,
    msa_fields: dict[str, str],
    *,
    name: str,
) -> dict[str, object]:
    """Create the MSA-bearing JSON produced by the current data stage."""
    chain: dict[str, object] = {
        "id": "A",
        "sequence": case.sequence,
        **msa_fields,
    }
    if case.polymer == "protein":
        chain["templates"] = []
    return {
        "name": name,
        "modelSeeds": [1],
        "sequences": [{case.polymer: chain}],
        "dialect": "alphafold3",
        "version": 1,
    }


def _run_msa_oracle_comparison(
    case_id: str,
    monolith_results: list[dict[str, object]],
    sharded_results: list[dict[str, object]],
) -> dict[str, object]:
    """Assemble current-pipeline fields and publish one scientific verdict."""
    case = _msa_oracle_case(case_id)
    if len(monolith_results) != len(case.database_ids) or len(sharded_results) != len(
        case.database_ids
    ):
        raise ValueError("Oracle comparison received an incomplete search set")
    BENCHMARK_OUTPUT_VOLUME.reload()
    contract = _assert_pinned_msa_assembly_contract()
    monolith_a3ms: dict[str, str] = {}
    sharded_a3ms: dict[str, str] = {}
    database_comparisons: dict[str, dict[str, object]] = {}
    search_identities: dict[str, dict[str, str]] = {}
    monolith_non_query_hits = 0
    for database_id, monolith_result, sharded_result in zip(
        case.database_ids,
        monolith_results,
        sharded_results,
        strict=True,
    ):
        spec = _database_profile_spec(database_id)
        monolith_a3m, monolith_hits = _read_profile_search_evidence(
            monolith_result,
            expected_spec=spec,
            expected_layout="monolith",
        )
        sharded_a3m, sharded_hits = _read_profile_search_evidence(
            sharded_result,
            expected_spec=spec,
            expected_layout="sharded",
        )
        monolith_a3ms[database_id] = monolith_a3m
        sharded_a3ms[database_id] = sharded_a3m
        monolith_non_query_hits += len(monolith_hits)
        database_comparisons[database_id] = _compare_profile_hit_rows(
            spec,
            monolith_hits,
            sharded_hits,
        )
        monolith_identity = monolith_result.get("search_identity")
        sharded_identity = sharded_result.get("search_identity")
        if not isinstance(monolith_identity, str) or not isinstance(
            sharded_identity,
            str,
        ):
            raise ValueError("Oracle search identity is invalid")
        search_identities[database_id] = {
            "monolith": monolith_identity,
            "sharded": sharded_identity,
        }

    monolith_fields = _assemble_current_pipeline_msas(case, monolith_a3ms)
    sharded_fields = _assemble_current_pipeline_msas(case, sharded_a3ms)
    final_comparisons = {
        field: _compare_final_a3m(monolith_fields[field], sharded_fields[field])
        for field in monolith_fields
    }
    rna_hit_gate = case.polymer != "rna" or monolith_non_query_hits > 0
    passed = (
        rna_hit_gate
        and all(
            comparison["passed"] is True for comparison in database_comparisons.values()
        )
        and all(
            comparison["passed"] is True for comparison in final_comparisons.values()
        )
    )
    oracle_identity = _sha256_bytes(
        _json_bytes({
            "schema_version": 1,
            "case_id": case.case_id,
            "sequence": case.sequence,
            "search_identities": search_identities,
            "assembly_contract": contract,
        })
    )
    sequence_hash = hashlib.sha256(case.sequence.encode()).hexdigest()
    output_root = (
        Path(APP_INFO.output_dir)
        / "production-candidates"
        / "oracles"
        / case.case_id
        / sequence_hash[:2]
        / sequence_hash
        / oracle_identity
    )
    oracle_input = _oracle_fold_input(
        case,
        monolith_fields,
        name=f"{case.case_id}-unsharded",
    )
    candidate_input = _oracle_fold_input(
        case,
        sharded_fields,
        name=f"{case.case_id}-sharded",
    )
    summary: dict[str, object] = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "case_id": case.case_id,
        "polymer": case.polymer,
        "sequence_sha256": sequence_hash,
        "oracle_identity": oracle_identity,
        "search_identities": search_identities,
        "assembly_contract": contract,
        "database_comparisons": database_comparisons,
        "final_msa_comparisons": final_comparisons,
        "monolith_non_query_hits": monolith_non_query_hits,
        "rna_non_query_hit_gate_passed": rna_hit_gate,
        "completed_at": _utc_now(),
        "output_path": str(output_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_root / "unsharded-current-pipeline.json", oracle_input)
    _write_json_atomic(output_root / "sharded-candidate.json", candidate_input)
    _write_json_atomic(output_root / "summary.json", summary)
    BENCHMARK_OUTPUT_VOLUME.commit()
    _write_json_atomic(
        output_root / "done.json",
        {
            "schema_version": 1,
            "status": "complete",
            "oracle_identity": oracle_identity,
            "scientific_gate_passed": passed,
            "completed_at": _utc_now(),
        },
    )
    BENCHMARK_OUTPUT_VOLUME.commit()
    return summary


@app.function(
    cpu=2,
    memory=4096,
    timeout=1_800,
    max_containers=1,
    volumes={
        APP_INFO.output_dir: BENCHMARK_OUTPUT_VOLUME,
    },
)
def compare_msa_profile_oracle(
    case_id: str,
    monolith_results: list[dict[str, object]],
    sharded_results: list[dict[str, object]],
) -> dict[str, object]:
    """Assemble and compare one fixed protein or RNA oracle."""
    return _run_msa_oracle_comparison(
        case_id,
        monolith_results,
        sharded_results,
    )


@app.local_entrypoint()
def submit_alphafold3_msa_task(
    operation: str = "prepare",
    submit: bool = False,
    seqkit_threads: int = DEFAULT_SEQKIT_THREADS,
    search_mode: str = "smoke",
    database_id: str = "small_bfd",
    source_policy: str = "keep",
    sequence: str = "",
    oracle_case: str = "pembrolizumab-vh",
) -> None:
    """Plan or submit one isolated AlphaFold 3 MSA benchmark operation.

    Args:
        operation: Operation to plan or run: ``prepare``, ``build-profile``,
            ``benchmark-validator``, ``search-profile``, ``validate-oracle``,
            ``scan``, or ``search``.
        submit: Submit the displayed remote work. Defaults to false, which only
            prints the plan and incurs no Modal compute work.
        seqkit_threads: SeqKit/native worker count for preparation, default 8.
        search_mode: Fixed search workload: ``smoke``, ``sweep``, or ``matrix``.
        database_id: Fixed database ID for production-candidate operations.
        source_policy: Post-validation source action: keep, compress, or delete.
        sequence: Optional profile-search query; defaults by polymer type.
        oracle_case: Fixed scientific comparison case.
    """
    if operation == "prepare":
        plan = _build_prepare_plan(seqkit_threads)
    elif operation == "build-profile":
        plan = _production_profile_plan(
            database_id,
            seqkit_threads,
            source_policy,
        )
    elif operation == "benchmark-validator":
        plan = _record_multiset_benchmark_plan(seqkit_threads)
    elif operation == "search-profile":
        plan = _production_search_plan(database_id, sequence)
    elif operation == "validate-oracle":
        plan = _msa_oracle_plan(oracle_case)
    elif operation == "scan":
        plan = _build_scan_plan()
    elif operation == "search":
        plan = _build_search_plan(search_mode)
    else:
        raise ValueError(
            "operation must be 'prepare', 'build-profile', "
            "'benchmark-validator', 'search-profile', 'validate-oracle', "
            "'scan', or 'search'"
        )
    print(_json_bytes(plan).decode(), end="")
    if not submit:
        print("🧬 Plan only; no Modal function was submitted.")
        return
    if operation == "prepare":
        print("🧬 Submitting one small-BFD profile preparation function...")
        result = prepare_small_bfd_profile.remote(seqkit_threads=seqkit_threads)
    elif operation == "build-profile":
        print(f"🧬 Submitting the fixed {database_id} profile builder...")
        result = build_sharded_database.remote(
            database_id=database_id,
            seqkit_threads=seqkit_threads,
            source_policy=source_policy,
        )
    elif operation == "benchmark-validator":
        print("🧬 Submitting the read-only UniProt validator benchmark...")
        result = benchmark_record_multiset_validator.remote(
            seqkit_threads=seqkit_threads,
        )
    elif operation == "search-profile":
        spec = _database_profile_spec(database_id)
        query = _validate_profile_query(
            spec,
            sequence or _default_profile_query(spec),
        )
        print(f"🧬 Submitting one fixed {database_id} profile search...")
        result = search_database_profile.remote(
            database_id=database_id,
            sequence=query,
        )
    elif operation == "validate-oracle":
        case = _msa_oracle_case(oracle_case)
        inputs = [(database_id, case.sequence) for database_id in case.database_ids]
        print(
            "🧬 Submitting the fixed monolithic current-pipeline "
            f"{case.case_id} searches..."
        )
        monolith_results = list(search_unsharded_database_oracle.starmap(inputs))
        print(f"🧬 Submitting the fixed sharded {case.case_id} searches...")
        sharded_results = list(search_database_profile.starmap(inputs))
        print("🧬 Submitting the upstream assembly and scientific comparison...")
        result = compare_msa_profile_oracle.remote(
            case_id=case.case_id,
            monolith_results=monolith_results,
            sharded_results=sharded_results,
        )
    elif operation == "scan":
        print("🧬 Submitting the sequential Volume scan matrix...")
        result = _submit_scan_matrix()
    else:
        if search_mode == "smoke":
            print("🧬 Submitting the sequential small-BFD search smoke...")
            result = _submit_search_smoke()
        elif search_mode == "sweep":
            print("🧬 Submitting the focused one-shot small-BFD sweep...")
            result = _submit_focused_sweep()
        else:
            print("🧬 Submitting the one-shot measured small-BFD matrix...")
            result = _submit_search_matrix()
    print(_json_bytes(result).decode(), end="")
