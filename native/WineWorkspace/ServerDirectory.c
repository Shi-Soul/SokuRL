#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Wine 10.0 server/create_server_dir and Unix ntdll format these two paths
   through asprintf. Redirect only those formats for an explicitly scoped
   process. The original Wine binaries and other filesystem calls stay intact. */
int asprintf(char **output, const char *format, ...)
{
    int (*original)(char **, const char *, ...) = dlsym(RTLD_NEXT, "asprintf");
    int (*original_va)(char **, const char *, va_list) = dlsym(RTLD_NEXT, "vasprintf");
    const int server = strcmp(format, "/tmp/.wine-%u") == 0;
    const int client = strcmp(format, "/tmp/.wine-%u/server-%llx-%llx") == 0;
    va_list arguments;
    va_start(arguments, format);
    int result;
    if (server || client) {
        const char *root = getenv("SOKURL_WINE_SERVER_ROOT");
        if (!root || root[0] != '/' || strchr(root, '%')) {
            va_end(arguments);
            errno = EINVAL;
            return -1;
        }
        const unsigned uid = va_arg(arguments, unsigned);
        if (server) {
            result = original(output, "%s/.wine-%u", root, uid);
        } else {
            const unsigned long long device = va_arg(arguments, unsigned long long);
            const unsigned long long inode = va_arg(arguments, unsigned long long);
            result = original(output, "%s/.wine-%u/server-%llx-%llx", root, uid, device, inode);
        }
    } else {
        result = original_va(output, format, arguments);
    }
    va_end(arguments);
    return result;
}
