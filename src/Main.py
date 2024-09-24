import torch
import torch.optim as optim
from NERModel import NERModel
from NERDataset import NERDataset
from VocabBuilder import VocabBuilder
from CurriculumLearning import CurriculumLearning
from DataPreparation import DataPreparation
from ModelTraining import ModelTraining

class Main:
    @staticmethod
    def main():
        file_paths = [
            r"C:\Users\Batuhan Koyuncu\Datas\1\all.jsonl",
            r"C:\Users\Batuhan Koyuncu\Datas\2\all.jsonl",
            r"C:\Users\Batuhan Koyuncu\Datas\3\all.jsonl",
            r"C:\Users\Batuhan Koyuncu\Datas\4\all.jsonl"
        ]

        reports, labels = DataPreparation.load_data(file_paths)
        bio_data = DataPreparation.convert_to_bio(reports, labels)

        sentences = [report for report, _ in bio_data]
        tags = [tags for _, tags in bio_data]

        word_to_ix = VocabBuilder.build_vocab(sentences)
        tag_to_ix = VocabBuilder.build_tag_vocab(tags)

        word_to_ix["<PAD>"] = len(word_to_ix)
        tag_to_ix["<PAD>"] = len(tag_to_ix)

        embedding_dim = 128
        hidden_dim = 64

        model = NERModel(len(word_to_ix), len(tag_to_ix), embedding_dim, hidden_dim)

        optimizer = optim.Adam([
            {'params': model.embedding.parameters(), 'lr': 1e-3},
            {'params': model.lstm.parameters(), 'lr': 1e-4},
            {'params': model.attention.parameters(), 'lr': 1e-4},
            {'params': model.hidden2tag.parameters(), 'lr': 1e-3}
        ])

        easy_data, medium_data, hard_data = CurriculumLearning.split_data_by_difficulty(bio_data)

        CurriculumLearning.train_model_with_curriculum_learning(model, easy_data, medium_data, hard_data, word_to_ix, tag_to_ix, optimizer, epochs=3150)

if __name__ == "__main__":
    Main.main()