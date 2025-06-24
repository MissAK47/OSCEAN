// ... existing code ...
    // 调试和管理
    std::vector<DatabaseConnectionInfo> getAllDatabaseInfo();
    void manageConnections(); // 定期检查和关闭空闲连接

private:
    // ... private methods ...
    DatabaseType getDatabaseTypeFor(DataType dataType) const;

    // ... member variables ...
};

} // namespace oscean::core_services::metadata::impl 