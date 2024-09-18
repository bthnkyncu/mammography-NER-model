import json

class DataPreparation:
    @staticmethod
    def load_data(file_paths):
        reports = []
        labels = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    reports.append(data['text'])
                    labels.append(data['label'])
        return reports, labels

    @staticmethod
    def tokenize_with_positions(text):
        words = []
        current_pos = 0
        for word in text.split():
            start = text.find(word, current_pos)
            end = start + len(word)
            words.append((word, start, end))
            current_pos = end
        return words

    @staticmethod
    def convert_to_bio(reports, labels):
        bio_data = []
        for report, label in zip(reports, labels):
            words_with_positions = DataPreparation.tokenize_with_positions(report)
            bio_tags = ["O"] * len(words_with_positions)

            for entity in label:
                start, end, entity_label = entity
                for i, (word, word_start, word_end) in enumerate(words_with_positions):
                    if word_start == start:
                        bio_tags[i] = f"B-{entity_label}"
                    elif word_start > start and word_end <= end:
                        bio_tags[i] = f"I-{entity_label}"

            bio_data.append((report, bio_tags))
        return bio_data