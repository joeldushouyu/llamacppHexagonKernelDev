#pragma once

#include <math.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

inline float random_float(float min, float max)
{
    return min + static_cast<float>(rand()) / (static_cast<float>((float)RAND_MAX / (max - min)));
}

template <typename T>
void save_data_to_csv_file(const std::string &filename,
                           const std::vector<std::string> &headers,
                           const std::vector<std::vector<T>> &data,
                           int number_of_decimal = 6)
{
    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    const size_t cols = data.size();

    // Prepare headers: if too few, generate "colN" names; if too many, ignore extras.
    std::vector<std::string> hdrs;
    hdrs.reserve(cols);
    for (size_t i = 0; i < cols; ++i)
    {
        if (i < headers.size() && !headers[i].empty())
            hdrs.push_back(headers[i]);
        else
            hdrs.push_back("col" + std::to_string(i));
    }

    // Determine number of rows = maximum column length
    size_t rows = 0;
    for (const auto &col : data)
        rows = std::max(rows, col.size());

    // Helper to convert a cell to a properly escaped string
    auto cell_to_string = [&](const T &val) -> std::string {
        std::ostringstream ss;
        // Use fixed + precision for numeric-like types; it's harmless for strings.
        ss << std::fixed << std::setprecision(number_of_decimal) << val;
        std::string s = ss.str();

        // If the cell contains comma, quote or newline, escape quotes and wrap in quotes
        if (s.find_first_of(",\"\n") != std::string::npos)
        {
            std::string escaped;
            escaped.reserve(s.size() + 2);
            escaped.push_back('"');
            for (char c : s)
            {
                if (c == '"')
                    escaped.append("\"\""); // double quotes inside a CSV field
                else
                    escaped.push_back(c);
            }
            escaped.push_back('"');
            return escaped;
        }
        return s;
    };

    // Write headers
    for (size_t c = 0; c < cols; ++c)
    {
        file << hdrs[c];
        if (c + 1 < cols) file << ",";
    }
    file << "\n";

    // Write rows: for each row index, take element from each column (if present)
    for (size_t r = 0; r < rows; ++r)
    {
        for (size_t c = 0; c < cols; ++c)
        {
            if (r < data[c].size())
            {
                file << cell_to_string(data[c][r]);
            }
            // else write nothing (empty cell)
            if (c + 1 < cols) file << ",";
        }
        file << "\n";
    }

    std::cout << "Data saved to " << filename << " (" << rows << " rows x " << cols << " columns)\n";
}
