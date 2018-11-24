#include "parsed_labels.h"

#include <cassert>
#include <string>
#include <unordered_set>

#include <QFile>
#include <QTextStream>

parsed_labels_t::parsed_labels_t(QString const &labels_filepath,
                             QString const &descriptions_filepath):
    labels_filepath_(labels_filepath),
    descriptions_filepath_(descriptions_filepath)
{ }

void parsed_labels_t::read() {
    read_labels();
    read_descriptions();
}

std::vector<std::string> parsed_labels_t::describe(const std::vector<int> &indices) {
    std::vector<std::string> descriptions;
    for (auto i: indices) {
        const auto &label = labels_.at(i);
        auto it = labels_to_words_.find(label);
        descriptions.push_back(it->second);
    }
    return descriptions;
}

void parsed_labels_t::read_labels() {
    QFile output_labes_file(labels_filepath_);
    if (output_labes_file.open(QIODevice::ReadOnly)) {
        QTextStream in(&output_labes_file);
        while (!in.atEnd()) {
            auto line = in.readLine();
            labels_.emplace_back(line.trimmed().toStdString());
        }
    }
    assert(labels_.size() == 200);
}

void parsed_labels_t::read_descriptions() {
    std::unordered_set<std::string> existing_labels(labels_.begin(), labels_.end());
    auto it_end = existing_labels.end();

    QFile labels_text_file(descriptions_filepath_);
    if (labels_text_file.open(QIODevice::ReadOnly)) {
        QTextStream in(&labels_text_file);

        while (!in.atEnd()) {
            auto line = in.readLine();
            auto name_labes = line.split('\t', QString::SkipEmptyParts);
            for (auto &l: name_labes) { l = l.trimmed(); }
            assert(name_labes.size() == 2);
            auto label = name_labes[0].toStdString();
            auto description = name_labes[1].toStdString();

            auto it = existing_labels.find(label);
            if (it != it_end) {
                assert(labels_to_words_.find(label) == labels_to_words_.end());
                labels_to_words_.emplace(label, description);
            }
        }
    }
}
