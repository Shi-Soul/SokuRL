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
