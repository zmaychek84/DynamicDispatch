#include <cstdio>
#include <utils/tfuncs.hpp>
#include <utils/tmpfile.hpp>
namespace Utils {

FILE *create_tmpfile() {
#if _WIN32
  FILE *tmp_file = nullptr;
  auto err = tmpfile_s(&tmp_file);
  DD_ASSERT(err == 0, "tmpfile_s error");
#else
  FILE *tmp_file = tmpfile();
  DD_ASSERT(tmp_file, "tmpfile error");
#endif
  return tmp_file;
}

void dump_to_tmpfile(FILE *file, char *data, size_t size) {
  auto write_size = std::fwrite(data, 1, size, file);
  DD_ASSERT(write_size == size, "tmpfile write error");
  fseek64(file, 0, SEEK_SET);
}

void save_tmpfile_on_disk(const std::filesystem::path &path, FILE *file) {
  constexpr size_t buffer_size = 64 * 1024;
  char *buffer =
      new char[buffer_size]; // 64KB, on heap to avoid crash the stack
#ifdef _WIN32
  FILE *dest;
  fopen_s(&dest, path.string().c_str(), "wb");
#else
  FILE *dest = fopen(path.string().c_str(), "wb");
#endif
  DD_ASSERT(dest, "open file failed");
  size_t read_count;
  while ((read_count = fread(buffer, 1, buffer_size, file)) > 0) {
    fwrite(buffer, 1, read_count, dest);
  }
  fseek64(file, 0, SEEK_SET);
  fclose(dest);
  delete[] buffer;
}
}; // namespace Utils
