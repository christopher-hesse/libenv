#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include "libenv.h"

#define NUM_ELEMS(arr) (sizeof(arr) / sizeof(arr[0]))

// calc_counts takes an array of spaces and the length of that array and
// populates the out_counts array with the number of elements in each space
void calc_counts(const struct libenv_space *spaces, int space_count, int *out_counts) {
    for (int space_idx = 0; space_idx < space_count; space_idx++) {
        int count = 1;
        struct libenv_space space = spaces[space_idx];
        for (int dim = 0; dim < space.ndim; dim++) {
            count *= space.shape[dim];
        }
        out_counts[space_idx] = count;
    }
}

void fatal(const char *fmt, ...) {
    printf("fatal: ");
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    exit(EXIT_FAILURE);
}

const struct libenv_space OBSERVATION_SPACES[] = {
    {
        .name = "uint8_obs",
        .shape = {1, 2, 3},
        .ndim = 3,
        .type = LIBENV_SPACE_TYPE_BOX,
        .dtype = LIBENV_DTYPE_UINT8,
        .low.uint8 = 0,
        .high.uint8 = 128,
    },
    {
        .name = "int32_obs",
        .shape = {4, 5, 6},
        .ndim = 3,
        .type = LIBENV_SPACE_TYPE_BOX,
        .dtype = LIBENV_DTYPE_INT32,
        .low.float32 = -100,
        .high.float32 = 100,
    },
    {
        .name = "float32_obs",
        .shape = {7, 8, 9},
        .ndim = 3,
        .type = LIBENV_SPACE_TYPE_BOX,
        .dtype = LIBENV_DTYPE_FLOAT32,
        .low.float32 = -1000.0,
        .high.float32 = 1000.0,
    },
};

const struct libenv_space ACTION_SPACES[] = {{
    .name = "action",
    .shape = {1},
    .ndim = 1,
    .type = LIBENV_SPACE_TYPE_DISCRETE,
    .dtype = LIBENV_DTYPE_UINT8,
    .low.uint8 = 0,
    .high.uint8 = 32,
}};

const struct libenv_space INFO_SPACES[] = {{
    .name = "info",
    .shape = {1},
    .ndim = 1,
    .type = LIBENV_SPACE_TYPE_DISCRETE,
    .dtype = LIBENV_DTYPE_INT32,
    .low.int32 = 0,
    .high.int32 = 10000,
}};

const struct libenv_space RENDER_SPACES[] = {{
    .name = "rgb_array",
    .shape = {8, 8, 3},
    .ndim = 3,
    .type = LIBENV_SPACE_TYPE_BOX,
    .dtype = LIBENV_DTYPE_INT32,
    .low.int32 = 0,
    .high.int32 = 10000,
}};

struct environment {
    int num_envs;
    int step_count;
    bool right_action;
    struct libenv_step *pending_step;
};

void libenv_load() {
    srand(time(0));
}

libenv_venv *libenv_make(int num_envs, const struct libenv_options options) {
    for (int i = 0; i < options.count; i++) {
        struct libenv_option opt = options.items[i];
        fatal("unrecognized option %s\n", opt.name);
    }

    struct environment *e = calloc(1, sizeof(struct environment));
    e->num_envs = num_envs;
    e->step_count = 0;
    e->right_action = true;
    return e;
}

int libenv_get_spaces(libenv_venv *env, enum libenv_spaces_name name, struct libenv_space *out_spaces) {
    int count = 0;
    const struct libenv_space *spaces = NULL;

    if (name == LIBENV_SPACES_OBSERVATION) {
        count = NUM_ELEMS(OBSERVATION_SPACES);
        spaces = OBSERVATION_SPACES;
    } else if (name == LIBENV_SPACES_ACTION) {
        count = NUM_ELEMS(ACTION_SPACES);
        spaces = ACTION_SPACES;
    } else if (name == LIBENV_SPACES_INFO) {
        count = NUM_ELEMS(INFO_SPACES);
        spaces = INFO_SPACES;
    } else if (name == LIBENV_SPACES_RENDER) {
        count = NUM_ELEMS(RENDER_SPACES);
        spaces = RENDER_SPACES;
    }

    if (out_spaces != NULL && spaces != NULL) {
        for (int i = 0; i < count; i++) {
            out_spaces[i] = spaces[i];
        }
    }
    return count;
}

void observe(struct environment *e, struct libenv_step *step) {
    int counts[NUM_ELEMS(OBSERVATION_SPACES)];
    calc_counts(OBSERVATION_SPACES, NUM_ELEMS(OBSERVATION_SPACES), counts);

    for (int env_idx = 0; env_idx < e->num_envs; env_idx++) {
        uint8_t *uint8_obs = step->obs[0 * e->num_envs + env_idx];
        for (int i = 0; i < counts[0]; i++) {
            uint8_obs[i] = i;
        }
        int32_t *int32_obs = step->obs[1 * e->num_envs + env_idx];
        for (int i = 0; i < counts[1]; i++) {
            int32_obs[i] = i;
        }
        float *float_obs = step->obs[2 * e->num_envs + env_idx];
        for (int i = 0; i < counts[2]; i++) {
            float_obs[i] = i;
        }
        step->rews[env_idx] = e->step_count * env_idx * e->right_action;
        bool done = false;
        if (e->step_count >= 100) {
            done = true;
        }
        step->dones[env_idx] = done;
        if (env_idx == 0) {
            // check alignment, use 64 bytes for 512 width instructions
            // if the user cares, they should also make sure the size of each space
            // is a multiple of 64 as well, otherwise individual observations
            // will not end on a multiple of alignment and subsequent observations
            // will not be aligned
            assert((uintptr_t)(uint8_obs) % 64 == 0);
            assert((uintptr_t)(int32_obs) % 64 == 0);
            assert((uintptr_t)(float_obs) % 64 == 0);
            assert((uintptr_t)(step->rews) % 64 == 0);
            assert((uintptr_t)(step->dones) % 64 == 0);
        }

        int32_t *info = step->infos[0 * e->num_envs + env_idx];
        info[0] = e->step_count * env_idx;
    }
}

void libenv_reset(libenv_venv *env, struct libenv_step *step) {
    struct environment *e = env;
    observe(e, step);
}

void libenv_step_async(libenv_venv *env, const void **acts, struct libenv_step *step) {
    struct environment *e = env;
    e->right_action = true;
    for (int env_idx = 0; env_idx < e->num_envs; env_idx++) {
        const uint8_t *act = acts[env_idx];
        if (*act != env_idx) {
            e->right_action = false;
        }
    }
    e->pending_step = step;
    e->step_count++;
}

void libenv_step_wait(libenv_venv *env) {
    struct environment *e = env;
    observe(e, e->pending_step);
    e->pending_step = NULL;
}

bool libenv_render(libenv_venv *env, const char *mode, void **frames) {
    struct environment *e = env;
    if (strcmp(mode, "rgb_array") == 0) {
        int counts[NUM_ELEMS(RENDER_SPACES)];
        calc_counts(RENDER_SPACES, NUM_ELEMS(RENDER_SPACES), counts);
        for (int env_idx = 0; env_idx < e->num_envs; env_idx++) {
            uint32_t *rgb = frames[env_idx];
            for (int i = 0; i < counts[0]; i++) {
                rgb[i] = e->step_count * env_idx;
            }
        }
    }
    return true;
}

void libenv_close(libenv_venv *env) {
    struct environment *e = env;
    free(e);
}

void libenv_unload() {
}

LIBENV_API int special_function(int x) {
    return x;
}
