#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Xvfb and its xkbcomp child share this process-local cache redirection.
   Keep system executables intact; never change ownership or access checks. */
static const char *cache_path(const char *path, char *buffer)
{
    const char prefix[] = "/var/lib/xkb/";
    if (strncmp(path, prefix, sizeof(prefix) - 1))
        return path;
    const char *root = getenv("SOKURL_XKB_CACHE");
    const char *name = path + sizeof(prefix) - 1;
    if (!root || root[0] != '/' || strchr(name, '/') || !strcmp(name, "..")) {
        errno = EINVAL;
        return NULL;
    }
    if (snprintf(buffer, PATH_MAX, "%s/%s", root, name) >= PATH_MAX) {
        errno = ENAMETOOLONG;
        return NULL;
    }
    return buffer;
}

FILE *fopen(const char *path, const char *mode)
{
    FILE *(*original)(const char *, const char *) = dlsym(RTLD_NEXT, "fopen");
    char buffer[PATH_MAX];
    const char *target = cache_path(path, buffer);
    return target ? original(target, mode) : NULL;
}

int open(const char *path, int flags, ...)
{
    int (*original)(const char *, int, ...) = dlsym(RTLD_NEXT, "open");
    mode_t mode = 0;
    if ((flags & O_CREAT) || (flags & O_TMPFILE) == O_TMPFILE) {
        va_list arguments;
        va_start(arguments, flags);
        mode = va_arg(arguments, mode_t);
        va_end(arguments);
    }
    char buffer[PATH_MAX];
    const char *target = cache_path(path, buffer);
    return target ? original(target, flags, mode) : -1;
}

int unlink(const char *path)
{
    int (*original)(const char *) = dlsym(RTLD_NEXT, "unlink");
    char buffer[PATH_MAX];
    const char *target = cache_path(path, buffer);
    return target ? original(target) : -1;
}
