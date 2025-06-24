#pragma once

#include <any>
#include <map>
#include <string>
#include <stdexcept>
#include <mutex>
#include <shared_mutex>

namespace oscean {
namespace workflow_engine {

/**
 * @class WorkflowContext
 * @brief A thread-safe, generic key-value store for sharing data across workflow stages.
 *
 * This class allows different parts of a workflow to share data of various types
 * using a string key. It is designed to be thread-safe for concurrent read/write access.
 */
class WorkflowContext {
public:
    WorkflowContext() = default;

    /**
     * @brief Sets a value for a given key. Overwrites if the key already exists.
     * @tparam T The type of the value.
     * @param key The key to associate the value with.
     * @param value The value to store.
     */
    template<typename T>
    void set(const std::string& key, T value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        data_[key] = std::move(value);
    }

    /**
     * @brief Gets the value for a given key.
     * @tparam T The expected type of the value.
     * @param key The key of the value to retrieve.
     * @return A reference to the stored value.
     * @throw std::out_of_range if the key does not exist.
     * @throw std::bad_any_cast if the stored value is not of type T.
     */
    template<typename T>
    T& get(const std::string& key) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = data_.find(key);
        if (it == data_.end()) {
            throw std::out_of_range("Key not found in WorkflowContext: " + key);
        }
        return std::any_cast<T&>(it->second);
    }

    /**
     * @brief Gets the value for a given key (const version).
     * @tparam T The expected type of the value.
     * @param key The key of the value to retrieve.
     * @return A const reference to the stored value.
     */
    template<typename T>
    const T& get(const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        auto it = data_.find(key);
        if (it == data_.end()) {
            throw std::out_of_range("Key not found in WorkflowContext: " + key);
        }
        return std::any_cast<const T&>(it->second);
    }

    /**
     * @brief Checks if a key exists in the context.
     * @param key The key to check.
     * @return True if the key exists, false otherwise.
     */
    bool has(const std::string& key) const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return data_.count(key) > 0;
    }

private:
    mutable std::shared_mutex mutex_;
    std::map<std::string, std::any> data_;
};

} // namespace workflow_engine
} // namespace oscean 