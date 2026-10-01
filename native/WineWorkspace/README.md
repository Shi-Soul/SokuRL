# Wine 工作目录约束

`ServerDirectory.c` 供 Linux 上的隔离验证使用，不注入游戏。它只替换 Wine 10.0
生成服务端运行目录的两个 `asprintf` 格式，将目录放到
`SOKURL_WINE_SERVER_ROOT` 指定的绝对路径内。缺少该配置时直接失败。

Wine 客户端与 wineserver 必须同时通过 `LD_PRELOAD` 加载这个库，才能连接同一个目录。
使用前须确认当前 Wine 二进制仍通过这两个格式调用 `asprintf`。
该库不修改原 Wine 文件，不改变其他文件访问，也不迁移已运行的服务。

编译：

```sh
gcc -shared -fPIC -O2 -Wall -Wextra ServerDirectory.c -ldl -o ServerDirectory.so
```

除此之外，验证启动器必须把 `WINEPREFIX`、`TMPDIR`、`XDG_CACHE_HOME`、
Python 缓存及输出目录设在获准工作目录内。该库只负责 Wine 服务端目录，
不能单独保证所有依赖都遵守整个文件系统范围。

`XkbDirectory.c` 将 Xvfb 和它启动的 xkbcomp 对 `/var/lib/xkb/` 内缓存文件的
打开和删除操作改到 `SOKURL_XKB_CACHE` 指定的绝对目录。它只适用于已确认通过
`fopen`、`open` 和 `unlink` 访问缓存的二进制；其他路径不变，原可执行文件不变。
把该库加入 Xvfb 的 `LD_PRELOAD`，子进程会继承相同设置。

```sh
gcc -shared -fPIC -O2 -Wall -Wextra XkbDirectory.c -ldl -o XkbDirectory.so
```

Xvfb 的授权文件、帧缓冲、缓存、日志和临时目录也须放在获准目录中。
使用带授权的 TCP 连接，关闭 Unix 监听和锁文件，避免生成系统临时目录文件。
当前 Xvfb 要求以 root 身份使用 `-nolock`；测试游戏仍以非特权用户运行。
