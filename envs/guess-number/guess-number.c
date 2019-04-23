#include "libenv.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#include <math.h>

#define NUM_ELEMS(arr) (sizeof(arr) / sizeof(arr[0]))

void fatal(const char *fmt, ...) {
    printf("fatal: ");
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    exit(EXIT_FAILURE);
}

#define fassert(cond)                                                          \
    do {                                                                       \
        if (!(cond)) {                                                         \
            printf("fassert failed %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

struct instance {
    bool *bits;
    bool random_bits;
    uint32_t num_bits;
    uint32_t step_count;
    bool failed;
    bool ready;
};

struct environment {
    void **pending_acts;
    struct instance *instances;
    uint32_t num_envs;
    struct libenv_step *step;
};

void libenv_load() {
    srand(time(0));
}

const struct libenv_space observation_spaces[] = {
    {
        .name = "observation",
        .shape = {1},
        .ndim = 1,
        .type = LIBENV_SPACE_TYPE_BOX,
        .dtype = LIBENV_DTYPE_FLOAT32,
        .low.float32 = -INFINITY,
        .high.float32 = INFINITY,
    },
};

const struct libenv_space action_spaces[] = {
    {
        .name = "action",
        .shape = {1},
        .ndim = 1,
        .type = LIBENV_SPACE_TYPE_BOX,
        .dtype = LIBENV_DTYPE_UINT8,
        .low.uint8 = 0,
        .high.uint8 = 1,
    },
};

libenv_venv *libenv_make(int num_envs, const struct libenv_options options) {
    struct environment *e = calloc(1, sizeof(struct environment));
    e->num_envs = num_envs;
    e->instances = calloc(num_envs, sizeof(struct instance));

    int num_bits = 0;
    bool random_bits = true;
    uint64_t *n_values = calloc(num_envs, sizeof(uint64_t));

    for (int i = 0; i < options.count; i++) {
        struct libenv_option opt = options.items[i];
        if (strcmp(opt.name, "num_bits") == 0) {
            if (!random_bits) {
                fatal("must specify one of num_bits and n\n");
            }
            fassert(opt.dtype == LIBENV_DTYPE_INT32);
            num_bits = *(int32_t *)(opt.data);
        } else if (strcmp(opt.name, "n") == 0) {
            fassert(opt.dtype == LIBENV_DTYPE_INT32);
            fassert(opt.count == num_envs);
            int32_t *values = opt.data;
            for (int env_idx = 0; env_idx < num_envs; env_idx++) {
                n_values[env_idx] = values[env_idx];
            }
            if (num_bits != 0) {
                fatal("must specify one of num_bits and n\n");
            }
            random_bits = false;
            num_bits = 64;
        } else {
            fatal("unrecognized option %s\n", opt.name);
        }
    }

    if (num_bits == 0) {
        fatal("neither num_bits nor n were specified\n");
    }

    for (int env_idx = 0; env_idx < num_envs; env_idx++) {
        struct instance *inst = &e->instances[env_idx];
        inst->num_bits = num_bits;
        inst->random_bits = random_bits;
        uint64_t n = n_values[env_idx];
        if (n != 0) {
            inst->bits = calloc(64, sizeof(bool));
            size_t i = 0;
            while (n > 0) {
                inst->bits[i] = n & 0x1;
                n >>= 1;
                i++;
            }
        }
    }
    free(n_values);
    return e;
}

int libenv_get_spaces(libenv_venv *env, enum libenv_spaces_name name, struct libenv_space *out_spaces) {
    int count = 0;
    const struct libenv_space *spaces = NULL;

    if (name == LIBENV_SPACES_OBSERVATION) {
        count = NUM_ELEMS(observation_spaces);
        spaces = observation_spaces;
    } else if (name == LIBENV_SPACES_ACTION) {
        count = NUM_ELEMS(action_spaces);
        spaces = action_spaces;
    }

    if (out_spaces != NULL && spaces != NULL) {
        for (int i = 0; i < count; i++) {
            out_spaces[i] = spaces[i];
        }
    }
    return count;
}

void libenv_reset(libenv_venv *env, struct libenv_step *step) {
    struct environment *e = env;

    for (int env_idx = 0; env_idx < e->num_envs; env_idx++) {
        struct instance *inst = &e->instances[env_idx];
        inst->ready = true;
        inst->step_count = 0;
        inst->failed = false;
        if (inst->random_bits) {
            free(inst->bits);
            inst->bits = calloc(inst->num_bits, sizeof(bool));
            for (int i = 0; i < inst->num_bits; i++) {
                inst->bits[i] = rand() % 2;
            }
        }

        float *obs = step->obs[env_idx];
        *obs = (float)inst->step_count;
        step->rews[env_idx] = 0.0;
        step->dones[env_idx] = false;
    }
}

void libenv_step_async(libenv_venv *env, const void **acts, struct libenv_step *step) {
    struct environment *e = env;
    e->pending_acts = (void **)(acts);
    e->step = step;
}

void libenv_step_wait(libenv_venv *env) {
    struct environment *e = env;

    for (int env_idx = 0; env_idx < e->num_envs; env_idx++) {
        struct instance *inst = &e->instances[env_idx];

        if (!inst->ready) {
            fatal("environment not reset before initial use\n");
        }

        if (inst->step_count > inst->num_bits) {
            printf("step %d bits %d\n", inst->step_count, inst->num_bits);
            fatal("stepped past end of environment\n");
        }

        uint8_t guess = *(uint8_t *)(e->pending_acts[env_idx]);

        float rew = 0.0;
        bool done = false;

        if (guess != inst->bits[inst->step_count] || inst->step_count == inst->num_bits - 1 || inst->failed) {
            inst->failed = true;
            rew = 0.0;
            done = true;
        } else {
            rew = 1.0;
            done = false;
        }
        inst->step_count++;

        e->step->rews[env_idx] = rew;
        e->step->dones[env_idx] = done;

        if (done) {
            // automatically reset because this is a vecenv
            inst->step_count = 0;
        }

        float *obs = e->step->obs[env_idx];
        *obs = (float)inst->step_count;
    }
    e->step = NULL;
}

void libenv_close(libenv_venv *env) {
    struct environment *e = env;
    for (int num_env = 0; num_env < e->num_envs; num_env++) {
        struct instance *inst = &e->instances[num_env];
        free(inst->bits);
    }
    free(e->instances);
    free(e);
}

void libenv_unload() {
}