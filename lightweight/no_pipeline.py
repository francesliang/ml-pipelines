import sys


class Pipeline:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.outputs = {}

    def log(self, input_item):
        """
        Log the start of the pipeline processing.
        """
        print("Start pipeline.")
        return input_item

    def import_data(self, input_item):
        """
        Import data to pipeline.
        """
        # Do import data
        imported_data = None
        return imported_data

    def process_data(self, imported_data):
        """
        Process data in pipeline.
        """
        # Do process data
        processed_data = None
        return processed_data

    def extract_feature(self, feature_name, processed_data):
        """
        Extract features from data.
        """
        # Do extract feature
        extracted_feature = None
        return extracted_feature

    def output_features(self, extracted_features):
        """
        Output data features
        """
        # Do output features
        features = []
        return features

    def run(self, input_item):
        """
        Run pipeline.
        """
        input_item = self.log(input_item)
        imported_data = self.import_data(input_item)
        processed_data = self.process_data(imported_data)
        features = []
        for feature_name in ["feature-1", "feature-2"]:
            feature = self.extract_feature(feature_name, processed_data)
            features.append(feature)
        features = self.output_features(features)


if __name__ == "__main__":
    input_item = None
    data_dir = ""
    pipe = Pipeline(data_dir)
    pipe.run(input_item)

