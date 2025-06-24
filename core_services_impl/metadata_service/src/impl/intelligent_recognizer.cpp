#include "impl/intelligent_recognizer.h"
#include <regex>
#include <algorithm>
#include <set>
#include <fstream>
#include <iostream>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <fmt/format.h>

#ifndef PROJECT_SOURCE_DIR
#define PROJECT_SOURCE_DIR "."
#endif

namespace oscean::core_services::metadata::impl {

IntelligentRecognizer::IntelligentRecognizer(
    std::shared_ptr<oscean::common_utils::infrastructure::logging::ILogger> logger,
    std::shared_ptr<core_services::data_access::IUnifiedDataAccessService> dataAccessService,
    std::shared_ptr<core_services::ICrsService> crsService,
    bool loadClassificationRules)
    : m_logger(std::move(logger)),
      m_dataAccessService(std::move(dataAccessService)),
      m_crsService(std::move(crsService)),
      m_rulesLoaded(false) {
    
    if (!m_logger) {
        throw std::invalid_argument("ILogger is null");
    }
    
    m_formatRules = std::make_unique<YAML::Node>();
    m_variableRules = std::make_unique<YAML::Node>();
    
    if (loadClassificationRules) {
        loadConfigurationFiles();
    } else {
        m_logger->info("IntelligentRecognizer initialized, skipping classification rule loading.");
    }
}

void IntelligentRecognizer::loadVariableClassificationRules(const std::string& path) {
    try {
        std::ifstream f(path);
        if (!f.good()) {
            m_logger->warn(fmt::format("Variable classification config not found at: {}", path));
            loadDefaultClassificationRules();
            return;
        }

        m_logger->info(fmt::format("Loading variable classification rules from: {}", path));
        *m_variableRules = YAML::LoadFile(path);
        m_logger->info("Successfully loaded variable classification rules.");

    } catch (const YAML::Exception& e) {
        m_logger->error(fmt::format("Failed to load or parse variable classification config: {}. Error: {}", path, e.what()));
        loadDefaultClassificationRules();
    } catch (const std::exception& e) {
        m_logger->error(fmt::format("Generic error loading variable classification config: {}. Error: {}", path, e.what()));
        loadDefaultClassificationRules();
    }
}

void IntelligentRecognizer::loadFileFormatRules(const std::string& path) {
    try {
        std::ifstream f(path);
        if (!f.good()) {
            m_logger->warn(fmt::format("File format config not found at: {}", path));
            return;
        }
        
        m_logger->info(fmt::format("Loading file format rules from: {}", path));
        *m_formatRules = YAML::LoadFile(path);
        m_logger->info("Successfully loaded file format rules.");
        
    } catch (const YAML::Exception& e) {
        m_logger->error(fmt::format("Failed to load or parse file format config: {}. Error: {}", path, e.what()));
    } catch (const std::exception& e) {
        m_logger->error(fmt::format("Generic error loading file format config: {}. Error: {}", path, e.what()));
    }
}

void IntelligentRecognizer::loadDefaultClassificationRules() {
    m_logger->warn("Loading default classification rules (which are none). System relies on YAML files.");
    m_logger->warn("Please check config/variable_classification.yaml");
    *m_variableRules = YAML::Node(YAML::NodeType::Map);
}

void IntelligentRecognizer::loadConfigurationFiles() {
    m_logger->info("Loading YAML configuration files...");
    
    try {
        std::string var_rules_path = std::string(PROJECT_SOURCE_DIR) + "/config/variable_classification.yaml";
        std::string format_rules_path = std::string(PROJECT_SOURCE_DIR) + "/core_services_impl/metadata_service/config/file_format_mapping.yaml";
        
        loadVariableClassificationRules(var_rules_path);
        loadFileFormatRules(format_rules_path);

        m_rulesLoaded = true;
        m_logger->info("YAML configuration files loaded successfully.");
    } catch (const std::exception& e) {
        m_logger->error(fmt::format("Failed to load YAML configuration files: {}", e.what()));
        m_rulesLoaded = false;
        throw;
    }
}

ClassificationResult IntelligentRecognizer::classifyFile(const core_services::FileMetadata& metadata) const {
    ClassificationResult result;
    result.reason = "Starting classification based on YAML rules.\n";
    
    m_logger->info(fmt::format("Classifying file: {}", metadata.filePath));

    if (!m_rulesLoaded) {
        result.reason += "Classification failed: Rules were not loaded.";
        m_logger->error(fmt::format("Cannot classify file '{}' because classification rules are not loaded.", metadata.filePath));
        return result;
    }

    std::string extension = metadata.format;
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    if ((*m_formatRules)["format_database_mapping"] && (*m_formatRules)["format_database_mapping"][extension]) {
        std::string db_type_str = (*m_formatRules)["format_database_mapping"][extension].as<std::string>();
        result.reason += "Classified based on file extension '" + extension + "'.\n";
    }

    if (!metadata.variables.empty()) {
        auto confidenceScores = determineDetailedDataTypes(metadata.variables);
        result.primaryCategory = determinePrimaryCategory(confidenceScores);
        result.confidenceScores = confidenceScores;
        result.reason += "Classified based on variable content analysis.\n";
        m_logger->debug(fmt::format("Classification scores determined for file '{}'", metadata.filePath));
    } else {
         m_logger->warn(fmt::format("No variables found in metadata for file '{}', skipping content-based classification.", metadata.filePath));
         result.reason += "Skipped content analysis: No variables present.\n";
    }

    m_logger->info(fmt::format("File classification finished for: {}", metadata.filePath));
    return result;
}

std::map<DataType, double> IntelligentRecognizer::determineDetailedDataTypes(
    const std::vector<core_services::VariableMeta>& variables) const {
    
    std::map<DataType, double> confidenceScores;
    if (!m_variableRules || !(*m_variableRules)["rules"]) {
        m_logger->warn("Variable classification rules are not loaded or are empty.");
        return confidenceScores;
    }

    const auto& rules = (*m_variableRules)["rules"];

    for (const auto& var : variables) {
        for (const auto& ruleNode : rules) {
            // ... (需要实现完整的匹配逻辑) ...
        }
    }
    
    // ... 归一化分数等 ...

    return confidenceScores;
}

DataType IntelligentRecognizer::determinePrimaryCategory(const std::map<DataType, double>& confidenceScores) const {
    if (confidenceScores.empty()) {
        return DataType::UNKNOWN;
    }

    auto maxElement = std::max_element(confidenceScores.begin(), confidenceScores.end(),
        [](const auto& a, const auto& b) {
            return a.second < b.second;
        });

    return maxElement->first;
}

std::string IntelligentRecognizer::classifyVariable(const std::string& variableName) const {
    return "Not Implemented";
}

std::vector<std::string> IntelligentRecognizer::classifyVariables(const std::vector<std::string>& variableNames) const {
     return {};
}

void IntelligentRecognizer::updateClassificationConfig(const VariableClassificationConfig& config) {
    //
}

std::vector<DataType> IntelligentRecognizer::determineDataTypeFromVariables(const std::vector<oscean::core_services::VariableMeta>& variables) const {
    return {};
}

std::string IntelligentRecognizer::getNormalizedVariableName(const std::string& originalName) const {
    return originalName;
}

bool IntelligentRecognizer::shouldIncludeVariableForDataType(const std::string& varType, DataType dataType) const {
    return false;
}

void IntelligentRecognizer::enrichWithSpatialInfo(core_services::FileMetadata& metadata) const {
    //
}

void IntelligentRecognizer::enrichWithTemporalInfo(core_services::FileMetadata& metadata) const {
    //
}

} // namespace oscean::core_services::metadata::impl 