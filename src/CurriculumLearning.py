from torch.utils.data import DataLoader

class CurriculumLearning:
    @staticmethod
    def difficulty_level(report):
        return len(report.split())

    @staticmethod
    def split_data_by_difficulty(bio_data):
        bio_data_sorted = sorted(bio_data, key=lambda x: CurriculumLearning.difficulty_level(x[0]))
        split_1 = int(len(bio_data_sorted) * 0.33)
        split_2 = int(len(bio_data_sorted) * 0.66)

        easy_data = bio_data_sorted[:split_1]
        medium_data = bio_data_sorted[split_1:split_2]
        hard_data = bio_data_sorted[split_2:]

        return easy_data, medium_data, hard_data

    @staticmethod
    def train_model_with_curriculum_learning(model, easy_data, medium_data, hard_data, word_to_ix, tag_to_ix, optimizer, epochs=3150):
        for stage, data in enumerate([easy_data, medium_data, hard_data]):
            print(f"Curriculum Learning Stage {stage + 1}")
            dataset = NERDataset(data, word_to_ix, tag_to_ix)
            data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=NERDataset.collate_fn)
            ModelTraining.train_and_evaluate_model_with_mixed_precision(model, data_loader, optimizer, tag_to_ix, epochs=int(epochs / 3))
