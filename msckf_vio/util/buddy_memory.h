#ifndef BUDDY_MEMORY_H
#define BUDDY_MEMORY_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>

#define NODE_UNUSED 0
#define NODE_USED 1
#define NODE_SPLIT 2
#define NODE_FULL 3

typedef struct {
    int32_t level;
    uint8_t tree[1];
} buddy_t;

static int32_t buddy_size(buddy_t *self, int32_t offset) {
    assert(offset < (1 << self->level));
    int32_t left = 0;
    int32_t length = 1 << self->level;
    int32_t index = 0;

    for (;;) {
        switch (self->tree[index]) {
        case NODE_USED:
            assert(offset == left);
            return length;
        case NODE_UNUSED:
            assert(0);
            return length;
        default:
            length /= 2;
            if (offset < left + length) {
                index = index * 2 + 1;
            }
            else {
                left += length;
                index = index * 2 + 2;
            }
            break;
        }
    }
}

static inline int32_t is_pow_of_2(uint32_t x) {
    return !(x & (x - 1));
}

static inline uint32_t next_pow_of_2(uint32_t x) {
    if (is_pow_of_2(x))
        return x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x + 1;
}

static inline int32_t _index_offset(
    int32_t index,
    int32_t level,
    int32_t max_level) {
    return ((index + 1) - (1 << level)) << (max_level - level);
}

static void _mark_parent(buddy_t * self, int32_t index) {
    for (;;) {
        int32_t buddy = index - 1 + (index & 1) * 2;
        if (buddy > 0 &&
            (self->tree[buddy] == NODE_USED
            || self->tree[buddy] == NODE_FULL)) {
            index = (index + 1) / 2 - 1;
            self->tree[index] = NODE_FULL;
        }
        else {
            return;
        }
    }
}

static void _combine(buddy_t *self, int32_t index) {
    for (;;) {
        int32_t buddy = index - 1 + (index & 1) * 2;
        if (buddy < 0 || self->tree[buddy] != NODE_UNUSED) {
            self->tree[index] = NODE_UNUSED;
            while (
                ((index = (index + 1) / 2 - 1) >= 0)
                && self->tree[index] == NODE_FULL){
                self->tree[index] = NODE_SPLIT;
            }
            return;
        }
        index = (index + 1) / 2 - 1;
    }
}

static void _dump(buddy_t *self, int32_t index, int32_t level) {
    switch (self->tree[index]) {
    case NODE_UNUSED:
        printf("(%d:%d)", _index_offset(index, level, self->level), 1 << (self->level - level));
        break;
    case NODE_USED:
        printf("[%d:%d]", _index_offset(index, level, self->level), 1 << (self->level - level));
        break;
    case NODE_FULL:
        printf("{");
        _dump(self, index * 2 + 1, level + 1);
        _dump(self, index * 2 + 2, level + 1);
        printf("}");
        break;
    default:
        printf("(");
        _dump(self, index * 2 + 1, level + 1);
        _dump(self, index * 2 + 2, level + 1);
        printf(")");
        break;
    }
}

static void buddy_dump(buddy_t *self) {
    _dump(self, 0, 0);
    printf("\n");
}

static buddy_t *buddy_new(int32_t level) {
    int32_t size = 1 << level;
    int32_t total_size =
        sizeof(buddy_t)+sizeof(uint8_t)* (size * 2 - 2);
    buddy_t *self = (buddy_t *)malloc(total_size);
    self->level = level;
    memset(self->tree, NODE_UNUSED, size * 2 - 1);
    return self;
}

static void buddy_delete(buddy_t* self) {
    free(self);
}

static int32_t buddy_alloc(buddy_t *self, int32_t s) {
    int32_t size;
    if (s == 0) {
        size = 1;
    }
    else {
        size = (int32_t)next_pow_of_2(s);
    }
    int32_t length = 1 << self->level;

    if (size > length)
        return -1;

    int32_t index = 0;
    int32_t level = 0;

    while (index >= 0) {
        if (size == length) {
            if (self->tree[index] == NODE_UNUSED) {
                self->tree[index] = NODE_USED;
                _mark_parent(self, index);
                return _index_offset(index, level, self->level);
            }
        }
        else {
            // size < length
            switch (self->tree[index]) {
            case NODE_USED:
            case NODE_FULL:
                break;
            case NODE_UNUSED:
                // split first
                self->tree[index] = NODE_SPLIT;
                self->tree[index * 2 + 1] = NODE_UNUSED;
                self->tree[index * 2 + 2] = NODE_UNUSED;
            default:
                index = index * 2 + 1;
                length /= 2;
                level++;
                continue;
            }
        }
        if (index & 1) {
            ++index;
            continue;
        }
        for (;;) {
            level--;
            length *= 2;
            index = (index + 1) / 2 - 1;
            if (index < 0)
                return -1;
            if (index & 1) {
                ++index;
                break;
            }
        }
    }

    return -1;
}

static void buddy_free(buddy_t *self, int32_t offset) {
    assert(offset < (1 << self->level));
    int32_t left = 0;
    int32_t length = 1 << self->level;
    int32_t index = 0;

    for (;;) {
        switch (self->tree[index]) {
        case NODE_USED:
            assert(offset == left);
            _combine(self, index);
            return;
        case NODE_UNUSED:
            assert(0);
            return;
        default:
            length /= 2;
            if (offset < left + length) {
                index = index * 2 + 1;
            }
            else {
                left += length;
                index = index * 2 + 2;
            }
            break;
        }
    }
}

class buddy_memory
{
public:
    buddy_memory(int32_t level, int32_t unit) :
        elem_size(unit) {
        pbuddy = buddy_new(level);
        pmemory = malloc((1 << level) * unit);
    }

    buddy_memory() = delete;

    ~buddy_memory() {
        buddy_delete(pbuddy);
        free(pmemory);
    }

    buddy_memory& operator=(buddy_memory &other) = default;

    void *mymalloc(uint32_t size) {
        int32_t buddy_size = (int32_t)((float)size / elem_size + 0.5f);
        int32_t buddy_addr = buddy_alloc(pbuddy, buddy_size);
        if (buddy_addr < 0) {
            return NULL;
        }
        return (uint8_t *)pmemory + buddy_addr * elem_size;
    }

    void myfree(void *ptr) {
        int32_t buddy_addr =
            ((uint32_t)ptr - (uint32_t)pmemory) / elem_size;
        buddy_free(pbuddy, buddy_addr);
    }

private:
    void *pmemory;
    buddy_t *pbuddy;
    int32_t elem_size;
};

#endif