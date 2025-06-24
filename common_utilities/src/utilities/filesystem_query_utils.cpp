#include "common_utils/utilities/filesystem_utils.h"
#include "common_utils/utilities/logging_utils.h"
#include <filesystem>
#include <fstream>
#include <iostream> // For std::cerr in some original logging/error cases if not fully replaced by LOG_ERROR
#include <algorithm> // For std::sort or other algorithms if used
#include <system_error>
#include <sstream>
#include <regex>
#include <cstring> // For memcmp, strchr
#include <iomanip> // For std::setfill, std::setw in calculateFileHash


#ifdef _WIN32
#include <windows.h> // Required for GetTempFileNameA, CreateFile, etc. in some original functions, though not directly in query utils here. For future PLATFORM_SPECIFIC sections.
#else
#include <unistd.h> // For POSIX specific functionalities (e.g. stat, access)
#include <sys/stat.h> // For stat
#include <openssl/md5.h> // For calculateFileHash
#endif

namespace oscean {
namespace common_utils {

namespace fs = std::filesystem;

// Private helper function for logging filesystem errors, specific to this compilation unit now.
// If this needs to be shared across more new .cpp files, it should go into an internal helper header.
static void logFilesystemError(const std::string& operation, const fs::path& path, const std::error_code& ec) {
    std::string path_str = path.string(); // Convert path to string once.
    LOG_ERROR("Filesystem error during '{}' on path '{}': {} ({})", 
              operation, path_str, ec.message(), ec.value());
}

// Private helper function for wildcard pattern matching, specific to this compilation unit.
// Originally in an anonymous namespace.
static bool matchesPattern(const std::string& text, const std::string& pattern) {
    std::string regexPattern = "";
    for (char c : pattern) {
        if (c == '*') {
            regexPattern += ".*";
        } else if (c == '?') {
            regexPattern += ".";
        } else {
            // 需要转义的正则表达式特殊字符
            if (strchr("[](){}+.|^$\\", c)) {
                regexPattern += "\\";
            }
            regexPattern += c;
        }
    }
    
    try {
        std::regex regex(regexPattern, std::regex::ECMAScript | std::regex::icase);
        return std::regex_match(text, regex);
    } catch (const std::regex_error& e) {
        LOG_ERROR("Regex error in matchesPattern for pattern '{}': {} (code: {})", pattern, e.what(), static_cast<int>(e.code()));
        return false; 
    }
}


bool FilesystemUtils::exists(const fs::path& path) {
    try {
        std::error_code ec;
        bool file_exists = fs::exists(path, ec);
        if (ec) {
             logFilesystemError("exists check", path, ec);
             return false; 
        }
        return file_exists;
    } catch (const fs::filesystem_error& e) { // Catch specific filesystem_error
         logFilesystemError("exists check", path, e.code()); // Log using the error code from exception
         return false;
    }
}

bool FilesystemUtils::isDirectory(const fs::path& path) {
    try {
        std::error_code ec;
        bool is_dir = fs::is_directory(path, ec);
        if (ec) {
             logFilesystemError("is_directory check", path, ec);
             return false; 
        }
        return is_dir;
    } catch (const fs::filesystem_error& e) {
        logFilesystemError("is_directory check", path, e.code());
        return false;
    }
}

bool FilesystemUtils::isFile(const fs::path& path) {
     try {
        std::error_code ec;
        bool is_file = fs::is_regular_file(path, ec);
        if (ec) {
             logFilesystemError("is_regular_file check", path, ec);
             return false; 
        }
        return is_file;
    } catch (const fs::filesystem_error& e) {
        logFilesystemError("is_regular_file check", path, e.code());
        return false;
    }
}

std::optional<uintmax_t> FilesystemUtils::getFileSize(const fs::path& path) {
    try {
        std::error_code ec;
        if (!fs::is_regular_file(path, ec) || ec) {
            if (ec) {
                 logFilesystemError("check is_regular_file in getFileSize", path, ec);
            }
            return std::nullopt;
        }

        uintmax_t size = fs::file_size(path, ec);
        if (ec) {
             logFilesystemError("get file size", path, ec);
            return std::nullopt;
        }
        return size;
    } catch (const fs::filesystem_error& e) {
         logFilesystemError("getting file size", path, e.code());
         return std::nullopt;
     }
}

std::optional<fs::file_time_type> FilesystemUtils::getLastModifiedTime(const fs::path& path) {
    try {
        std::error_code ec;
        auto time = fs::last_write_time(path, ec);
        if (ec) {
            logFilesystemError("get last write time", path, ec);
            return std::nullopt;
        }
        return time;
    } catch (const fs::filesystem_error& e) {
        logFilesystemError("get last write time", path, e.code());
        return std::nullopt;
    }
}

fs::perms FilesystemUtils::getPermissions(const fs::path& path) {
    try {
        std::error_code ec;
        fs::file_status status = fs::status(path, ec);
        if (ec) {
            logFilesystemError("getting permissions (status check)", path, ec);
            return fs::perms::unknown;
        }
        return status.permissions();
     } catch (const fs::filesystem_error& e) {
         logFilesystemError("getting permissions", path, e.code());
         return fs::perms::unknown;
    }
}

FilesystemUtils::FileInfo FilesystemUtils::getFileInfo(const fs::path& path) {
    FileInfo info;
    info.path = path.string();
    
    std::error_code ec_exists;
    if (!fs::exists(path, ec_exists) || ec_exists) {
        if (ec_exists) logFilesystemError("getFileInfo (exists check)", path, ec_exists);
        else LOG_WARN("Attempted to get FileInfo for non-existent path: {}", info.path);
        return info; 
    }

    try {
        std::error_code ec;
        fs::file_status status = fs::status(path, ec);
        if (ec) {
            logFilesystemError("getting status in getFileInfo", path, ec);
            return info;
        }

        info.name = path.filename().string();
        info.extension = path.extension().string();
        if (!info.extension.empty() && info.extension[0] == '.') { 
            info.extension = info.extension.substr(1); 
        }
        info.isDirectory = fs::is_directory(status);
        
        if (fs::is_regular_file(status)) {
            info.size = fs::file_size(path, ec);
            if (ec) logFilesystemError("getting file_size in getFileInfo", path, ec);
        }
        
        info.lastModifiedTime = fs::last_write_time(path, ec);
        if (ec) logFilesystemError("getting last_write_time in getFileInfo", path, ec);
        
        info.permissions = status.permissions();

        // Platform-specific time info
        #ifndef _WIN32
            struct stat st;
            if (stat(path.c_str(), &st) == 0) {
                auto to_time_point = [](time_t t) {
                    return fs::file_time_type(std::chrono::system_clock::from_time_t(t).time_since_epoch());
                };
                #ifdef __APPLE__
                    info.creationTime = to_time_point(st.st_birthtimespec.tv_sec);
                #elif defined(__linux__)
                    // Note: st.st_ctime is last status change time on Linux.
                    // True creation time (btime) might require statx() on newer kernels/glibc
                    // For simplicity and wider compatibility, this might be omitted or documented.
                    // info.creationTime = to_time_point(st.st_ctime); 
                #endif
                info.lastAccessTime = to_time_point(st.st_atime);
            } else {
                // LOG_WARN or similar if stat fails for an existing file
            }
        #else
            // Windows specific: CreateFile, GetFileTime for creation/access times
            // This part was commented out in the original, keeping it similar for now
            // or would require implementing the FILETIME to file_time_type conversion.
        #endif

    } catch (const fs::filesystem_error& e) {
        logFilesystemError("getFileInfo", path, e.code());
    }
    return info;
}

std::vector<std::string> FilesystemUtils::listDirectory(const fs::path& directory, bool recursive, FileType type) {
    std::vector<std::string> result;
    std::error_code ec_check;
    if (!fs::exists(directory, ec_check) || ec_check || !fs::is_directory(directory, ec_check) || ec_check) {
        if(ec_check) logFilesystemError("listDirectory (pre-check)", directory, ec_check);
        else LOG_ERROR("Cannot list directory, path is not a valid directory: {}", directory.string());
        return result;
    }

    try {
        auto checkType = [&](const fs::directory_entry& entry) {
            std::error_code ec_status;
            fs::file_status status = entry.status(ec_status);
            if (ec_status) {
                 logFilesystemError("status check in listDirectory/checkType", entry.path(), ec_status);
                 return false; // Skip if status cannot be determined
            }
            switch (type) {
                case FileType::FILE:      return fs::is_regular_file(status);
                case FileType::DIRECTORY: return fs::is_directory(status);
                case FileType::SYMLINK:   return fs::is_symlink(status);
                case FileType::OTHER:     return fs::is_other(status);
                case FileType::ALL:
                default:                  return true;
            }
        };

        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(directory, fs::directory_options::skip_permission_denied)) {
                if (checkType(entry)) {
                    result.push_back(entry.path().string());
                }
            }
        } else {
            for (const auto& entry : fs::directory_iterator(directory, fs::directory_options::skip_permission_denied)) {
                 if (checkType(entry)) {
                    result.push_back(entry.path().string());
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        logFilesystemError("listDirectory traversal", directory, e.code());
    }
    return result;
}

std::vector<FilesystemUtils::FileInfo> FilesystemUtils::listDirectoryInfo(const fs::path& path, bool recursive, FileType type) {
    std::vector<FileInfo> results;
    std::error_code ec_check;
    if (!fs::exists(path, ec_check) || ec_check || !fs::is_directory(path, ec_check) || ec_check) {
        if(ec_check) logFilesystemError("listDirectoryInfo (pre-check)", path, ec_check);
        else LOG_ERROR("Cannot list directory info, path is not a valid directory: {}", path.string());
        return results;
    }

    try {
        auto processEntry = [&](const fs::directory_entry& entry) -> std::optional<FileInfo> {
            std::error_code ec_status;
            fs::file_status status = entry.status(ec_status);
            if (ec_status) {
                logFilesystemError("getting status in listDirectoryInfo/processEntry", entry.path(), ec_status);
                return std::nullopt;
            }

            bool typeMatch = false;
            switch (type) {
                case FileType::FILE:      typeMatch = fs::is_regular_file(status); break;
                case FileType::DIRECTORY: typeMatch = fs::is_directory(status); break;
                case FileType::SYMLINK:   typeMatch = fs::is_symlink(status); break;
                case FileType::OTHER:     typeMatch = fs::is_other(status); break;
                case FileType::ALL:       typeMatch = true; break;
            }

            if (typeMatch) {
                // Use the existing getFileInfo to populate details, ensuring consistency
                // This avoids duplicating logic for populating FileInfo struct.
                // Note: getFileInfo itself handles non-existent paths, but here entry.path() should exist.
                return getFileInfo(entry.path());
            }
            return std::nullopt;
        };

        fs::directory_options options = fs::directory_options::skip_permission_denied;

        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(path, options)) {
                if(auto infoOpt = processEntry(entry)) {
                    results.push_back(*infoOpt);
                }
            }
        } else {
            for (const auto& entry : fs::directory_iterator(path, options)) {
                if(auto infoOpt = processEntry(entry)) {
                    results.push_back(*infoOpt);
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        logFilesystemError("listDirectoryInfo traversal", path, e.code());
    }
    return results;
}

void FilesystemUtils::traverseDirectory(const fs::path& path,
                                     std::function<bool(const FileInfo&)> callback,
                                     bool recursive,
                                     FileType type) {
    std::error_code ec_check;
    if (!fs::exists(path, ec_check) || ec_check || !fs::is_directory(path, ec_check) || ec_check) {
        if(ec_check) logFilesystemError("traverseDirectory (pre-check)", path, ec_check);
        else LOG_ERROR("Cannot traverse directory, path is not a valid directory: {}", path.string());
        return;
    }
    if (!callback) return;

    try {
        auto processEntry = [&](const fs::directory_entry& entry) -> bool {
            std::error_code ec_status;
            fs::file_status status = entry.status(ec_status);
            if (ec_status) {
                logFilesystemError("getting status in traverseDirectory/processEntry", entry.path(), ec_status);
                return true; // Continue traversal
            }

            bool typeMatch = false;
            switch (type) {
                case FileType::FILE:      typeMatch = fs::is_regular_file(status); break;
                case FileType::DIRECTORY: typeMatch = fs::is_directory(status); break;
                case FileType::SYMLINK:   typeMatch = fs::is_symlink(status); break;
                case FileType::OTHER:     typeMatch = fs::is_other(status); break;
                case FileType::ALL:       typeMatch = true; break;
            }

            if (typeMatch) {
                // Use getFileInfo to populate for consistency
                FileInfo info = getFileInfo(entry.path()); 
                if (!callback(info)) {
                    return false; // Callback requested to stop
                }
            }
            return true; // Continue traversal
        };
        
        fs::directory_options options = fs::directory_options::skip_permission_denied;

        if (recursive) {
            for (const auto& entry : fs::recursive_directory_iterator(path, options)) {
                if (!processEntry(entry)) break;
            }
        } else {
            for (const auto& entry : fs::directory_iterator(path, options)) {
                if (!processEntry(entry)) break;
            }
        }
    } catch (const fs::filesystem_error& e) {
        logFilesystemError("traverseDirectory traversal", path, e.code());
    }
}

std::vector<FilesystemUtils::FileInfo> FilesystemUtils::findFiles(const fs::path& path,
                                                               const std::string& pattern,
                                                               bool recursive) {
    std::vector<FileInfo> results;
    std::error_code ec_check;
    if (!fs::exists(path, ec_check) || ec_check || !fs::is_directory(path, ec_check) || ec_check) {
         if(ec_check) logFilesystemError("findFiles (pre-check dir)", path, ec_check);
         else LOG_ERROR("Cannot find files, path is not a valid directory: {}", path.string());
        return results;
    }
    
    std::string regex_str_from_pattern;
    regex_str_from_pattern.reserve(pattern.length() * 2);
     for (char c : pattern) {
         switch (c) {
             case '*':
                 regex_str_from_pattern += ".*";
                 break;
             case '?':
                 regex_str_from_pattern += ".";
                 break;
             case '.': case '^': case '$': case '|': case '(': case ')':
             case '[': case ']': case '{': case '}': case '+':
                 regex_str_from_pattern += "\\";
                 regex_str_from_pattern += c;
                 break;
             case '\\':
                 regex_str_from_pattern += "\\\\";
                 break;
             default:
                 regex_str_from_pattern += c;
                 break;
         }
     }

    std::optional<std::regex> file_regex_opt;
    try {
        file_regex_opt.emplace(regex_str_from_pattern, std::regex_constants::ECMAScript | std::regex_constants::icase);
    } catch (const std::regex_error& e) {
        LOG_ERROR("Regex error constructing regex for findFiles from pattern '{}' (generated: '{}'): {} (code: {})",
                  pattern, regex_str_from_pattern, e.what(), static_cast<int>(e.code()));
        return results;
    }
    
    traverseDirectory(path,
        [&](const FileInfo& info) {
            if (!info.isDirectory) { // Only check files
                // Use the pre-compiled regex
                if (std::regex_match(info.name, *file_regex_opt)) {
                    results.push_back(info);
                }
                // Or use the static helper:
                // if (matchesPattern(info.name, pattern)) {
                //     results.push_back(info);
                // }
            }
            return true; // Continue searching
        },
        recursive,
        FileType::ALL // Traverse all, then filter by isDirectory and pattern
    );
    return results;
}

bool FilesystemUtils::compareFiles(const fs::path& path1, const fs::path& path2) {
    std::error_code ec1, ec2;
    bool is_file1 = fs::is_regular_file(path1, ec1);
    bool is_file2 = fs::is_regular_file(path2, ec2);

    if (ec1 || !is_file1) {
        if (ec1) logFilesystemError("compareFiles (is_regular_file check path1)", path1, ec1);
        else LOG_WARN("Path1 is not a valid file for comparison: {}", path1.string());
        return false;
    }
    if (ec2 || !is_file2) {
        if (ec2) logFilesystemError("compareFiles (is_regular_file check path2)", path2, ec2);
        else LOG_WARN("Path2 is not a valid file for comparison: {}", path2.string());
        return false;
    }

    uintmax_t size1 = fs::file_size(path1, ec1);
    uintmax_t size2 = fs::file_size(path2, ec2);

    if (ec1 || ec2 || size1 != size2) {
        if(ec1) logFilesystemError("compareFiles (file_size path1)", path1, ec1);
        if(ec2) logFilesystemError("compareFiles (file_size path2)", path2, ec2);
        // If sizes differ, files are different (no need for log message if just different sizes)
        return false; 
    }

    std::ifstream file1(path1, std::ios::binary);
    std::ifstream file2(path2, std::ios::binary);

    if (!file1.is_open()) {
        LOG_ERROR("Failed to open file1 for comparison: {}", path1.string());
        return false;
    }
    if (!file2.is_open()) {
        LOG_ERROR("Failed to open file2 for comparison: {}", path2.string());
        return false;
    }

    constexpr size_t BUFFER_SIZE = 4096;
    std::vector<char> buffer1(BUFFER_SIZE);
    std::vector<char> buffer2(BUFFER_SIZE);

    try {
        while (file1.good() && file2.good()) { // Check stream state before read
            file1.read(buffer1.data(), BUFFER_SIZE);
            file2.read(buffer2.data(), BUFFER_SIZE);

            std::streamsize bytesRead1 = file1.gcount();
            std::streamsize bytesRead2 = file2.gcount();

            if (bytesRead1 != bytesRead2 || 
                (bytesRead1 > 0 && std::memcmp(buffer1.data(), buffer2.data(), static_cast<size_t>(bytesRead1)) != 0)) {
                return false; 
            }
            if (bytesRead1 == 0) break; // Both EOF or error
        }
        // Check if both ended at the same time (file1.eof() && file2.eof())
        // If sizes were equal and all blocks matched, this should be true.
        return file1.eof() && file2.eof() && !file1.bad() && !file2.bad();
    } catch (const std::ios_base::failure& e) { // Catch specific stream exceptions
        LOG_ERROR("IOS Exception during file comparison between '{}' and '{}': {}", path1.string(), path2.string(), e.what());
        return false;
    }
}

std::optional<std::string> FilesystemUtils::calculateFileHash(const fs::path& path) {
    std::error_code ec_check;
    if (!fs::is_regular_file(path, ec_check) || ec_check) {
        if(ec_check) logFilesystemError("calculateFileHash (is_regular_file check)", path, ec_check);
        else LOG_ERROR("Cannot calculate hash, path is not a valid file: {}", path.string());
        return std::nullopt;
    }

    #ifdef _WIN32
        LOG_WARN("File hashing (MD5) not implemented for Windows in FilesystemUtils::calculateFileHash using OpenSSL. Consider CNG API.");
        return std::nullopt; 
    #else // POSIX-like, use OpenSSL
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            LOG_ERROR("Failed to open file for hashing: {}", path.string());
            return std::nullopt;
        }

        MD5_CTX md5Context;
        if (MD5_Init(&md5Context) == 0) { // Check return of MD5_Init
            LOG_ERROR("MD5_Init failed for file hashing: {}", path.string());
            return std::nullopt;
        }

        constexpr size_t BUFFER_SIZE = 4096;
        std::vector<char> buffer(BUFFER_SIZE);

        try {
            while (file.good()) { // Check stream state before read
                file.read(buffer.data(), BUFFER_SIZE);
                std::streamsize bytesRead = file.gcount();
                if (bytesRead > 0) {
                    if (MD5_Update(&md5Context, buffer.data(), static_cast<size_t>(bytesRead)) == 0) {
                         LOG_ERROR("MD5_Update failed for file hashing: {}", path.string());
                         return std::nullopt; // Abort on OpenSSL error
                    }
                }
                if (bytesRead < BUFFER_SIZE && file.eof()) break; // End of file
                 if (file.fail() && !file.eof()) { // Check for read errors not EOF
                    LOG_ERROR("File read error during hashing for '{}'", path.string());
                    return std::nullopt;
                }
            }
            
            if (file.bad()) { // Check for unrecoverable stream errors
                LOG_ERROR("Unrecoverable stream error during hashing for '{}'", path.string());
                return std::nullopt;
            }

            unsigned char result[MD5_DIGEST_LENGTH];
            if (MD5_Final(result, &md5Context) == 0) {
                LOG_ERROR("MD5_Final failed for file hashing: {}", path.string());
                return std::nullopt;
            }

            std::stringstream ss;
            ss << std::hex << std::setfill('0');
            for (unsigned char i : result) { // Use range-based for loop
                ss << std::setw(2) << static_cast<unsigned>(i);
            }
            return ss.str();

        } catch (const std::ios_base::failure& e) { // Catch specific stream exceptions
             LOG_ERROR("IOS Exception during file hashing for '{}': {}", path.string(), e.what());
             return std::nullopt;
        }
    #endif
}

} // namespace common_utils
} // namespace oscean 