# Network Service Module (OSCEAN)

## 简介 (Introduction)

本模块 (`network_service`) 是 OSCEAN 后端应用的网络通信入口点。它负责监听指定的网络端口，接收来自客户端（如 CesiumJS 前端或其他应用程序）的 HTTP 或 WebSocket 请求，解析这些请求，并将它们路由到适当的处理逻辑（通常是任务调度层）。处理完成后，它还负责将结果格式化为 HTTP/WebSocket 响应并发送回客户端。

本模块的设计重点是高性能、高并发和低延迟，主要采用基于 Boost.Asio 和 Boost.Beast 的异步 I/O 模型。

## 核心功能 (Core Functionality)

*   **监听端口**: 在配置的 IP 地址和端口上启动 TCP 监听。
*   **接受连接**: 异步接受传入的客户端 TCP 连接。
*   **协议处理**: 解析 HTTP/1.1 协议，未来可扩展支持 WebSocket。
*   **请求解析**: 解析 HTTP 请求行、头部和消息体 (支持如 JSON 格式的正文)。
*   **请求路由**: 根据请求方法 (GET, POST 等) 和 URL 目标，将解析后的请求分发到注册的处理程序。
*   **任务分发**: 将处理请求的任务委托给上层服务（通常是 `TaskDispatcher` 模块）。
*   **响应构建**: 接收来自上层服务处理结果。
*   **响应格式化**: 将结果格式化为 HTTP 响应（设置状态码、头部、序列化正文，如 JSON 或二进制数据）。
*   **响应发送**: 异步将 HTTP 响应发送回客户端。
*   **连接管理**: 管理客户端连接的生命周期，包括 Keep-Alive 和超时处理。
*   **大数据流支持**: 设计上支持通过 HTTP Chunked Encoding 或流式写入来高效传输大规模数据，避免一次性加载到内存。

## 技术栈 (Technology Stack)

*   **核心异步 I/O**: Boost.Asio
*   **HTTP/WebSocket 协议**: Boost.Beast
*   **JSON 处理**: nlohmann/json
*   **日志**: spdlog (通过依赖注入)
*   **线程管理**: C++ 标准库 (`<thread>`, `<mutex>`, `<future>`), `common_utils/ThreadPool`
*   **构建系统**: CMake
*   **语言**: C++17

## 架构与核心组件 (Architecture & Core Components)

本模块采用基于异步事件驱动的架构，主要组件如下：

```
      +-------------------------------------------+       +----------------------------------+
|         网络服务层 (NetworkService)       |------>| 工作线程池 (common_utils/ThreadPool) |
|                                           |       +----------------------------------+
|  +-------------------------------------+  |
|  | 1. 监听器 (Listener)                |  | Handles CPU-bound tasks:
|  |    (Binds, accepts conns)           |  | - Request Parsing Logic
|  +------------------+------------------+  | - Routing Logic
|                     | (New Connection)    | - Response Formatting
|                     v                     |
|  +-------------------------------------+  |
|  | 2. 连接管理器 (ConnectionManager)   |  | (Implicit or explicit tracking)
|  |    (Optional: Tracks connections)   |  |
|  +------------------+------------------+  |
|                     | (Manage Connection) |
|                     v                     |
|  +-------------------------------------+  |       +----------------------------------+
|  | 3. 连接处理单元 (HttpConnection)    |------>| 请求路由器 (RequestRouter)         |
|  |    (Handles one connection's IO)    |  |       +-------------+--------------------+
|  |                                     |  |                     | (Route Decision)
|  |    +-----------------------------+  |  |                     v
|  |    | 3a. 异步读写器 (Beast Stream)|  |  |       +----------------------------------+
|  |    +-------------+---------------+  |  |       | 任务调度器 (TaskDispatcher)      |
|  |                  | (Recv Data/Send Resp)| |       | (Calls Core/Output Services)   |
|  |                  v                  |  |       +-------------+--------------------+
|  |    +-----------------------------+  |  |                     | (ResponseData)
|  |    | 3b. 请求解析器 (HTTP Parser)|<--------------------+ (Via Callback/Future)
|  |    |     (Beast Parser/Internal) |  |                     
|  |    +-------------+---------------+  |                     
|  |                  | (RequestDTO)       |
|  |                  v                  | (To RequestRouter)
|  |    +-----------------------------+  |
|  |    | 3c. 响应构建/写入器        |  |                     
|  |    |     (Beast Serializer/Async Write)|
|  |    +-----------------------------+  |
|  +-------------------------------------+  |
+-------------------------------------------+
```

*   **`NetworkServer`**: (主类) 负责整个网络服务的生命周期管理。初始化 `asio::io_context`、`ThreadPool`、`RequestRouter` 和 `Listener`。启动和停止 I/O 线程和工作线程。
*   **`Listener`**: (监听器) 绑定到指定的 IP 地址和端口，使用 `async_accept` 监听新的 TCP 连接。每当有新连接时，创建一个 `HttpConnection` 实例来处理它。
*   **`HttpConnection`**: (连接处理单元) 每个实例处理一个客户端连接。使用 Boost.Beast 的异步操作 (`async_read`, `async_write`) 来接收 HTTP 请求和发送响应。负责管理连接状态和超时。将请求解析和路由任务分派到工作线程池。
*   **`RequestRouter`**: (请求路由器) 接收来自 `HttpConnection` 的已解析请求（封装为 `RequestDTO`）。根据请求方法和目标 URL，查找预先注册的路由规则，并将请求分发给相应的处理程序（通常是调用 `TaskDispatcher` 的方法）。
*   **`RequestDTO`**: (数据传输对象) 内部结构体，用于封装从 HTTP 请求中解析出的关键信息（方法、目标、头部、正文等），传递给路由器和后续处理层。
*   **`ResponseData`**: (数据传输对象) 内部结构体，用于封装构建 HTTP 响应所需的信息（状态码、头部、正文数据 - 支持字符串、二进制、流），由处理层返回给 `HttpConnection` 进行发送。

## 工作流程 (Workflow)

1.  `NetworkServer` 初始化并启动 `Listener` 和 I/O 线程。
2.  `Listener` 异步接受新连接，创建 `HttpConnection` 并启动其 `async_read`。
3.  `HttpConnection` 在 I/O 线程上接收到数据，将数据和解析任务提交给 `ThreadPool`。
4.  工作线程执行 HTTP 解析（使用 Boost.Beast parser），生成 `RequestDTO`。
5.  工作线程调用 `RequestRouter`，传入 `RequestDTO` 和一个用于发送响应的回调函数。
6.  `RequestRouter` 匹配路由，调用注册的处理程序（例如，`TaskDispatcher` 的某个方法），并将响应回调传递下去。
7.  `TaskDispatcher` （在自己的线程或 `ThreadPool` 中）异步执行核心业务逻辑。
8.  业务逻辑完成后，`TaskDispatcher` 通过之前传递的回调函数，将包含结果或错误的 `ResponseData` 返回。
9.  响应回调（通常在工作线程中被调用）触发 `HttpConnection` 的响应发送逻辑。
10. `HttpConnection` 将响应数据（`ResponseData`）调度回其所属的 I/O 线程。
11. I/O 线程使用 Boost.Beast 的 `async_write` 将响应发送给客户端。
12. 发送完成后，根据 Keep-Alive 决定继续读取下一个请求或关闭连接。

## 依赖关系 (Dependencies)

*   **外部库**: Boost (Asio, Beast, System), spdlog, nlohmann/json。
*   **内部模块**: 
    *   `common_utils`: 需要 `ThreadPool` 和可能的 `Config` 类。
    *   `TaskDispatcher`: `RequestRouter` 将请求处理委托给 `TaskDispatcher`。
    *   `output_generation/tile_service` (间接): `TaskDispatcher` 会调用瓦片服务等，网络服务层本身不直接依赖具体业务模块，只依赖 `TaskDispatcher` 接口。

## 配置 (Configuration)

本模块通常通过主应用程序传递的 `Config` 对象获取其配置，主要包括：

*   监听 IP 地址
*   监听端口号
*   I/O 线程数量
*   (可能共享) 工作线程池数量
*   连接超时设置 (读/写/空闲)

## 集成与使用 (Integration & Usage)

`NetworkServer` 类是本模块的主要入口点。通常在应用程序的主函数 (`main`) 或启动逻辑中：

1.  创建 `TaskDispatcher`、`ThreadPool`、`Config` 等核心对象。
2.  实例化 `NetworkService::NetworkServer`，将上述依赖项注入其构造函数。
3.  调用 `networkServer->run()` 启动服务。
4.  应用程序需要处理优雅关闭，调用 `networkServer->stop()`。

`RequestRouter` 需要在初始化时配置路由规则，将特定的 HTTP 路径和方法映射到 `TaskDispatcher` 提供的处理函数上。

## 构建 (Building)

本模块作为 OSCEAN 项目的一部分，通过根目录的 `CMakeLists.txt` 中的 `add_subdirectory(network_service)` 命令进行构建。需要确保 CMake 配置能够找到所需的 Boost (Asio, Beast) 库和头文件，以及 spdlog 和 nlohmann/json。

本模块通常会被编译成一个静态库 (`network_service_lib`)，供主应用程序链接。 