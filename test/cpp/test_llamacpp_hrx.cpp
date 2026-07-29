#include "lemon/backend_manager.h"
#include "lemon/backends/backend_descriptor_registry.h"
#include "lemon/backends/llamacpp/llamacpp.h"
#include "lemon/backends/llamacpp/llamacpp_server.h"
#include "lemon/recipe_options.h"
#include "lemon/system_info.h"
#include "lemon/utils/archive_platform.h"
#include "lemon/utils/json_utils.h"
#include "lemon/utils/path_utils.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using lemon::BackendManager;
using lemon::BackendSupport;
using lemon::CPUInfo;
using lemon::GPUInfo;
using lemon::NPUInfo;
using lemon::RecipeOptions;
using lemon::SystemInfo;
using lemon::backends::LlamaCppServer;
using nlohmann::json;

namespace {

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (condition) {
        std::cout << "PASS: " << label << std::endl;
    } else {
        std::cerr << "FAIL: " << label << std::endl;
        ++failures;
    }
}

const BackendSupport* find_hrx_support() {
    const auto& support = lemon::backends::llamacpp::descriptor.support;
    auto it = std::find_if(
        support.begin(), support.end(),
        [](const BackendSupport& row) { return row.backend == "hrx"; });
    return it == support.end() ? nullptr : &*it;
}

class TestSystemInfo : public SystemInfo {
public:
    CPUInfo get_cpu_device() override { return {}; }
    GPUInfo get_amd_igpu_device() override { return {}; }
    std::vector<GPUInfo> get_amd_dgpu_devices() override { return {}; }
    std::vector<GPUInfo> get_nvidia_gpu_devices() override { return {}; }
    NPUInfo get_npu_device() override { return {}; }
};

void test_support_gating() {
    const BackendSupport* hrx = find_hrx_support();

#if defined(__linux__) && defined(__x86_64__)
    expect(hrx != nullptr, "HRX support row exists on Linux x86-64");
    if (hrx) {
        expect(hrx->supported_os == std::set<std::string>{"linux"},
               "HRX support row is Linux-only");
        auto amd_gpu = hrx->devices.find("amd_gpu");
        expect(amd_gpu != hrx->devices.end()
                   && amd_gpu->second
                       == std::set<std::string>{"gfx1100", "gfx1151", "gfx1201"},
               "HRX support row contains the packaged GPU targets");
    }
    expect(SystemInfo::backend_supports_arch("llamacpp", "hrx", "gfx1100"),
           "HRX accepts gfx1100");
    expect(SystemInfo::backend_supports_arch("llamacpp", "hrx", "gfx1151"),
           "HRX accepts gfx1151");
    expect(SystemInfo::backend_supports_arch("llamacpp", "hrx", "gfx1201"),
           "HRX accepts gfx1201");
    expect(!SystemInfo::backend_supports_arch("llamacpp", "hrx", "gfx1101"),
           "HRX rejects gfx1101");
    expect(!SystemInfo::backend_supports_arch("llamacpp", "hrx", "gfx1150"),
           "HRX rejects gfx1150");
    expect(!SystemInfo::backend_supports_arch("llamacpp", "hrx", "gfx1152"),
           "HRX rejects gfx1152");
    expect(!SystemInfo::backend_supports_arch("llamacpp", "hrx", "gfx1200"),
           "HRX rejects gfx1200");
#else
    expect(hrx == nullptr, "HRX support row is absent outside Linux x86-64");
    expect(!SystemInfo::backend_supports_arch("llamacpp", "hrx", "gfx1100"),
           "HRX rejects gfx1100 outside Linux x86-64");
    expect(!SystemInfo::backend_supports_arch("llamacpp", "hrx", "gfx1151"),
           "HRX rejects gfx1151 outside Linux x86-64");
    expect(!SystemInfo::backend_supports_arch("llamacpp", "hrx", "gfx1201"),
           "HRX rejects gfx1201 outside Linux x86-64");
#endif
}

void test_install_contract() {
    constexpr const char* kRepo = "ROCm/ggml-staging-release";
    constexpr const char* kTag = "hrx-b2";
    constexpr const char* kFilename =
        "llama-hrx-b2-bin-manylinux-hrx-x64.tar.gz";
    constexpr const char* kChecksum =
        "sha256:bcc514835bea3e342e390df0b233426070802df8ce13eb6be4a06beeca619f63";

#if defined(__linux__) && defined(__x86_64__)
    const auto params = LlamaCppServer::get_install_params("hrx", kTag);
    expect(params.repo == kRepo, "HRX uses the exact release repository");
    expect(params.filename == kFilename, "HRX uses the exact release filename");

    BackendManager manager;
    const auto resolved = manager.get_install_params("llamacpp", "hrx");
    expect(resolved.repo == kRepo, "HRX dry-run repository is pinned");
    expect(resolved.version == kTag, "HRX dry-run tag is pinned");
    expect(resolved.filename == kFilename, "HRX dry-run filename is pinned");
#else
    bool rejected = false;
    try {
        (void)LlamaCppServer::get_install_params("hrx", kTag);
    } catch (const std::exception&) {
        rejected = true;
    }
    expect(rejected, "HRX install mapping rejects non-Linux-x86-64 hosts");
#endif

    const json versions = lemon::utils::JsonUtils::load_from_file(
        lemon::utils::get_resource_path("resources/backend_versions.json"));
    expect(versions.at("llamacpp").at("hrx") == kTag,
           "HRX built-in pin is exact");
    expect(versions.at("checksums")
                   .at("github")
                   .at(kRepo)
                   .at(kTag)
                   .at(kFilename) == kChecksum,
           "HRX checksum is exact");

    const std::string entries =
        "llama-hrx-b2/\n"
        "llama-hrx-b2/bin/llama-server\n"
        "llama-hrx-b2/lib/libhsa-runtime64.so\n";
    expect(lemon::utils::compute_tarball_strip_components(entries) == 1,
           "normal extraction flattens HRX to its bin and lib directories");
}

void test_recipe_options() {
    RecipeOptions explicit_hrx(
        "llamacpp", {{"llamacpp_backend", "hrx"}});
    expect(explicit_hrx.get_option("llamacpp_backend") == "hrx",
           "explicit RecipeOptions preserve HRX");

    const json defaults =
        lemon::backends::llamacpp::descriptor.config_defaults();
    expect(defaults.contains("hrx_args"),
           "HRX has a standard argument override");
    expect(defaults.value("hrx_bin", "") == "builtin",
           "HRX has a standard binary override");
}

void test_automatic_default() {
#if defined(__linux__) && defined(__x86_64__)
    TestSystemInfo system_info;
    const json devices = {
        {"cpu", {
            {"name", "Test CPU"},
            {"family", "x86_64"},
            {"available", true},
        }},
        {"amd_gpu", json::array({
            {
                {"name", "AMD Radeon 8060S"},
                {"family", "gfx1151"},
                {"available", true},
                {"integrated", true},
            },
        })},
    };

    const json recipes = system_info.build_recipes_info(devices);
    const json& llamacpp = recipes.at("llamacpp");
    expect(llamacpp.at("backends").contains("hrx"),
           "system info exposes HRX");
    expect(llamacpp.value("default_backend", "") != "hrx",
           "automatic default assignment excludes HRX");
    expect(!llamacpp.at("backends").at("hrx").contains("auto_select"),
           "HRX uses the existing backend status shape");
#endif
}

}  // namespace

int main() {
    const auto nonce =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path cache_dir =
        fs::temp_directory_path()
        / ("lemonade-hrx-test-" + std::to_string(nonce));
    fs::create_directories(cache_dir);
    lemon::utils::set_cache_dir(cache_dir.string());

    try {
        test_support_gating();
        test_install_contract();
        test_recipe_options();
        test_automatic_default();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: unexpected exception: " << error.what() << std::endl;
        ++failures;
    }

    std::error_code cleanup_error;
    fs::remove_all(cache_dir, cleanup_error);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
