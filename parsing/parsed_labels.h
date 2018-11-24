#ifndef PARSED_LABELS_H
#define PARSED_LABELS_H

#include <string>
#include <vector>
#include <unordered_map>

#include <QString>

class parsed_labels_t
{
public:
    parsed_labels_t(QString const &labels_filepath,
                    QString const &descriptions_filepath);

public:
    void read();
    std::vector<std::string> describe(std::vector<int> const &indices);

private:
    void read_labels();
    void read_descriptions();

private:
    QString labels_filepath_;
    QString descriptions_filepath_;
    std::vector<std::string> labels_;
    std::unordered_map<std::string, std::string> labels_to_words_;
};

#endif // PARSED_LABELS_H
