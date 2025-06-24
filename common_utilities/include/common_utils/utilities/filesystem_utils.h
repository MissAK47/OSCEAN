#pragma once

#include <string>
#include <vector>
#include <optional>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <functional>
#include <iostream>

namespace oscean {
namespace common_utils {
namespace fs = std::filesystem;

/**
 * @brief 文件系统操作工具类，提供高级文件和目录操作API
 */
class FilesystemUtils {
public:
    /**
     * @brief 文件类型枚举，用于列出目录内容时筛选类型
     */
    enum class FileType {
        FILE,        ///< 仅文件
        DIRECTORY,   ///< 仅目录
        SYMLINK,     ///< 仅符号链接
        OTHER,       ///< 其他类型
        ALL          ///< 所有类型
    };

    /**
     * @brief 文件信息结构体
     */
    struct FileInfo {
        std::string path;                      ///< 文件完整路径
        std::string name;                      ///< 文件名称
        std::string extension;                 ///< 文件扩展名（不含点）
        bool isDirectory = false;              ///< 是否为目录
        uintmax_t size = 0;                    ///< 文件大小（字节）
        fs::file_time_type lastModifiedTime;   ///< 最后修改时间
        fs::file_time_type creationTime;       ///< 创建时间
        fs::file_time_type lastAccessTime;     ///< 最后访问时间
        fs::perms permissions;                 ///< 文件权限
    };

    /**
     * @brief 确保目录存在，如果不存在则创建
     * @param directory 目录路径
     * @return 成功返回true，失败返回false
     */
    static bool ensureDirectoryExists(const fs::path& directory);

    /**
     * @brief 将字符串内容写入文件
     * @param filePath 文件路径
     * @param content 要写入的内容
     * @param append 是否追加模式（默认false，覆盖模式）
     * @return 成功返回true，失败返回false
     */
    static bool writeStringToFile(const fs::path& filePath, const std::string& content, bool append = false);

    /**
     * @brief 写入二进制数据到文件
     * @param path 文件路径
     * @param data 二进制数据
     * @param append 是否追加模式
     * @return 是否成功写入
     */
    static bool writeBinaryToFile(const fs::path& path, 
                                const std::vector<char>& data,
                                bool append = false);

    /**
     * @brief 读取文件内容到字符串
     * @param filePath 文件路径
     * @return 成功返回文件内容，失败返回std::nullopt
     */
    static std::optional<std::string> readFileToString(const fs::path& filePath);

    /**
     * @brief 读取文件内容到二进制缓冲区
     * @param filePath 文件路径
     * @return 成功返回二进制数据，失败返回std::nullopt
     */
    static std::optional<std::vector<unsigned char>> readFileToBinary(const fs::path& filePath);

    /**
     * @brief 检查路径是否存在
     * @param path 要检查的路径
     * @return 存在返回true，不存在返回false
     */
    static bool exists(const fs::path& path);

    /**
     * @brief 检查路径是否为目录
     * @param path 要检查的路径
     * @return 是目录返回true，不是返回false
     */
    static bool isDirectory(const fs::path& path);

    /**
     * @brief 检查路径是否为文件
     * @param path 要检查的路径
     * @return 是文件返回true，不是返回false
     */
    static bool isFile(const fs::path& path);

    /**
     * @brief 创建目录
     * @param path 要创建的目录路径
     * @param recursive 是否递归创建（默认true）
     * @return 成功返回true，失败返回false
     */
    static bool createDirectory(const fs::path& path, bool recursive = true);

    /**
     * @brief 删除文件或目录
     * @param path 要删除的路径
     * @param recursive 删除目录时是否递归删除内容（默认false）
     * @return 成功返回true，失败返回false
     */
    static bool remove(const fs::path& path, bool recursive = false);

    /**
     * @brief 复制文件
     * @param source 源文件路径
     * @param destination 目标文件路径
     * @param overwrite 如果目标已存在是否覆盖（默认true）
     * @return 成功返回true，失败返回false
     */
    static bool copyFile(const fs::path& source, const fs::path& destination, bool overwrite = true);

    /**
     * @brief 复制目录
     * @param source 源目录路径
     * @param destination 目标目录路径
     * @param recursive 是否递归复制
     * @return 是否成功复制
     */
    static bool copyDirectory(const fs::path& source, 
                            const fs::path& destination, 
                            bool recursive = true);

    /**
     * @brief 移动/重命名文件
     * @param source 源文件路径
     * @param destination 目标文件路径
     * @return 成功返回true，失败返回false
     */
    static bool moveFile(const fs::path& source, const fs::path& destination);

    /**
     * @brief 重命名文件或目录
     * @param oldPath 旧路径
     * @param newPath 新路径
     * @return 是否成功重命名
     */
    static bool rename(const fs::path& oldPath, const fs::path& newPath);

    /**
     * @brief 获取文件扩展名（不含点）
     * @param path 文件路径
     * @return 文件扩展名字符串
     */
    static std::string getFileExtension(const fs::path& path);

    /**
     * @brief 获取文件名
     * @param path 文件路径
     * @param withExtension 是否包含扩展名（默认true）
     * @return 文件名字符串
     */
    static std::string getFileName(const fs::path& path, bool withExtension = true);

    /**
     * @brief 获取文件名 (不含路径和扩展名)
     * @param path 文件路径
     * @return 文件名
     */
    static std::string getFileNameWithoutExtension(const fs::path& path);

    /**
     * @brief 获取父目录
     * @param path 文件或目录路径
     * @return 父目录路径
     */
    static std::string getParentDirectory(const fs::path& path);

    /**
     * @brief 获取文件所在目录路径
     * @param path 文件路径
     * @return 目录路径
     */
    static std::string getDirectoryPath(const fs::path& path);

    /**
     * @brief 合并路径
     * @param path1 第一部分路径
     * @param path2 第二部分路径
     * @return 合并后的路径
     */
    static std::string combinePath(const fs::path& path1, const fs::path& path2);

    /**
     * @brief 列出目录内容
     * @param directory 要列出内容的目录路径
     * @param recursive 是否递归列出子目录内容（默认false）
     * @param type 要列出的文件类型（默认ALL）
     * @return 文件/目录路径列表
     */
    static std::vector<std::string> listDirectory(const fs::path& directory, 
                                                 bool recursive = false,
                                                 FileType type = FileType::ALL);

    /**
     * @brief 列出目录内容
     * @param path 目录路径
     * @param recursive 是否递归列出子目录内容
     * @param type 文件类型过滤
     * @return 文件信息列表
     */
    static std::vector<FileInfo> listDirectoryInfo(const fs::path& path, 
                                             bool recursive = false,
                                             FileType type = FileType::ALL);

    /**
     * @brief 遍历目录
     * @param path 目录路径
     * @param callback 回调函数，接受FileInfo参数，返回bool值(true继续遍历，false停止遍历)
     * @param recursive 是否递归遍历子目录
     * @param type 文件类型过滤
     */
    static void traverseDirectory(const fs::path& path, 
                                std::function<bool(const FileInfo&)> callback,
                                bool recursive = true,
                                FileType type = FileType::ALL);

    /**
     * @brief 搜索目录中的文件
     * @param path 目录路径
     * @param pattern 文件名模式 (支持通配符 * 和 ?)
     * @param recursive 是否递归搜索子目录
     * @return 匹配的文件信息列表
     */
    static std::vector<FileInfo> findFiles(const fs::path& path, 
                                         const std::string& pattern,
                                         bool recursive = true);

    /**
     * @brief 获取文件大小
     * @param path 文件路径
     * @return 成功返回文件大小（字节），失败返回std::nullopt
     */
    static std::optional<uintmax_t> getFileSize(const fs::path& path);

    /**
     * @brief 获取文件最后修改时间
     * @param path 文件路径
     * @return 最后修改时间，如果文件不存在返回空
     */
    static std::optional<fs::file_time_type> getLastModifiedTime(const fs::path& path);
    
    /**
     * @brief 设置文件最后修改时间
     * @param path 文件路径
     * @param time 要设置的时间
     * @return 是否成功设置
     */
    static bool setLastModifiedTime(const fs::path& path, const fs::file_time_type& time);

    /**
     * @brief 规范化路径（解析相对路径、符号链接等）
     * @param path 要规范化的路径
     * @return 规范化后的路径
     */
    static std::string normalizePath(const fs::path& path);

    /**
     * @brief 获取两个路径的相对路径
     * @param path 路径
     * @param basePath 基础路径
     * @return 相对路径
     */
    static std::string getRelativePath(const fs::path& path, const fs::path& basePath);
    
    /**
     * @brief 获取绝对路径
     * @param path 路径
     * @return 绝对路径
     */
    static std::string getAbsolutePath(const fs::path& path);

    /**
     * @brief 获取当前工作目录
     * @return 当前工作目录
     */
    static std::string getCurrentWorkingDirectory();
    
    /**
     * @brief 设置当前工作目录
     * @param path 路径
     * @return 是否成功设置
     */
    static bool setCurrentWorkingDirectory(const fs::path& path);
    
    /**
     * @brief 获取临时目录路径
     * @return 临时目录路径
     */
    static std::string getTempDirectory();
    
    /**
     * @brief 创建临时文件
     * @param prefix 文件名前缀
     * @param suffix 文件名后缀
     * @return 临时文件路径
     */
    static std::string createTempFile(const std::string& prefix = "tmp_", 
                                    const std::string& suffix = ".tmp");
    
    /**
     * @brief 创建临时目录
     * @param prefix 目录名前缀
     * @return 临时目录路径
     */
    static std::string createTempDirectory(const std::string& prefix = "tmp_dir_");

    /**
     * @brief 设置文件权限
     * @param path 文件路径
     * @param permissions 权限
     * @return 是否成功设置
     */
    static bool setPermissions(const fs::path& path, fs::perms permissions);
    
    /**
     * @brief 获取文件权限
     * @param path 文件路径
     * @return 文件权限
     */
    static fs::perms getPermissions(const fs::path& path);

    /**
     * @brief 获取文件详细信息
     * @param path 文件或目录路径
     * @return 文件信息结构体
     */
    static FileInfo getFileInfo(const fs::path& path);

    /**
     * @brief 比较两个文件内容是否相同
     * @param path1 第一个文件路径
     * @param path2 第二个文件路径
     * @return 是否相同
     */
    static bool compareFiles(const fs::path& path1, const fs::path& path2);
    
    /**
     * @brief 计算文件哈希值 (MD5)
     * @param path 文件路径
     * @return 文件哈希值，如果文件不存在或无法读取返回空
     */
    static std::optional<std::string> calculateFileHash(const fs::path& path);

    /**
     * @brief Lists all files in a directory, optionally recursively and filtering by extension.
     *
     * @param directoryPath The path to the directory.
     * @param recursive Whether to search recursively into subdirectories.
     * @param extensions An optional list of file extensions to include (e.g., {".txt", ".csv"}).
     *                   If empty, all files are included.
     * @return std::vector<std::filesystem::path> A list of paths to the files found.
     */
    std::vector<std::filesystem::path> listFiles(const fs::path& directoryPath,
                                                   bool recursive = false,
                                                   const std::vector<std::string>& extensions = {});

private:
    // Private constructor to prevent instantiation
    FilesystemUtils() = delete;
    ~FilesystemUtils() = delete;
    FilesystemUtils(const FilesystemUtils&) = delete;
    FilesystemUtils& operator=(const FilesystemUtils&) = delete;

    // Helper function for logging errors - 内联实现以避免链接错误
    static void logFilesystemError(const std::string& operation, const fs::path& path, const std::error_code& ec) {
        if (ec) {
            // 使用简单的std::cerr输出错误信息，避免复杂的日志依赖
            std::cerr << "[FilesystemUtils] " << operation << " failed for path '" 
                      << path.string() << "': " << ec.message() << " (error code: " << ec.value() << ")" << std::endl;
        }
    }
};

} // namespace common_utils
} // namespace oscean

