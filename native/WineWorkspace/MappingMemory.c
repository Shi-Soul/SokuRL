#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Wine 10.0 unlinks these files immediately. Keep that anonymous lifetime
   without allocating backing files on the host system disk. */
int open(const char *path, int flags, ...)
{
    mode_t mode = 0;
    if (flags & O_CREAT) {
        va_list args;
        va_start(args, flags);
        mode = va_arg(args, int);
        va_end(args);
    }
    if (strncmp(path, "tmpmap-", 7) == 0 && strchr(path, '/') == NULL &&
        (flags & (O_RDWR | O_CREAT | O_EXCL)) == (O_RDWR | O_CREAT | O_EXCL))
        return syscall(SYS_memfd_create, "sokurl-wine-mapping", MFD_CLOEXEC);
    int (*real_open)(const char *, int, ...) = dlsym(RTLD_NEXT, "open");
    return real_open(path, flags, mode);
}
