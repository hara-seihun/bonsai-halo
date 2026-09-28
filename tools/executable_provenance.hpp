#pragma once
#include <filesystem>
#include <cstdio>
#include <stdexcept>
#include <string>

inline std::string executable_sha256() {
    const std::string path=std::filesystem::read_symlink("/proc/self/exe").string();
    // Hash the running image through procfs, not argv[0] or the workspace's current build.
    // This still works if the executable was replaced or unlinked after launch.
    const std::string cmd="sha256sum /proc/"+std::filesystem::read_symlink("/proc/self").string()+"/exe";
    FILE *p=popen(cmd.c_str(),"r");
    if(!p)throw std::runtime_error("cannot hash running executable");
    char line[256]{};
    const bool read=fgets(line,sizeof(line),p)!=nullptr;
    const int status=pclose(p);
    std::string digest=line;
    if(!read||status||digest.size()<64||digest.find_first_not_of("0123456789abcdef",0)!=64)
        throw std::runtime_error("cannot hash running executable");
    return digest.substr(0,64)+"  "+path;
}
