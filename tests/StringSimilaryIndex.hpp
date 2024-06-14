#ifndef STRING_SIMILARY_INDEX
#define STRING_SIMILARY_INDEX

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

int64_t levenshtein_distance(const std::string& s1, const std::string& s2) {
  size_t len1 = s1.size(), len2 = s2.size();
  std::vector<std::vector<int64_t>> dp(len1 + 1,
                                       std::vector<int64_t>(len2 + 1));

  for (size_t i = 0; i <= len1; ++i) {
    for (size_t j = 0; j <= len2; ++j) {
      if (i == 0)
        dp[i][j] = j;
      else if (j == 0)
        dp[i][j] = i;
      else if (s1[i - 1] == s2[j - 1])
        dp[i][j] = dp[i - 1][j - 1];
      else
        dp[i][j] = 1 + std::min({dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]});
    }
  }
  return dp[len1][len2];
}

double similarity_index(const std::string& s1, const std::string& s2) {
  size_t max_len = std::max(s1.size(), s2.size());
  if (max_len == 0) return 1.0;
  auto dist = levenshtein_distance(s1, s2);
  return 1.0 - static_cast<double>(dist) / max_len;
}

#endif