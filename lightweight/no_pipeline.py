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
        imported_data = None
        # Do import data
        return imported_data

    def process_data(self, imported_data):
        """
        Process data in pipeline.
        """
        processed_data = None
        # Do process data
        return processed_data

    def extract_feature(self, feature_name, processed_data):
        """
        Extract features from data.
        """
        extracted_feature = self._extract(feature_name, processed_data)
        return extracted_feature

    def output_features(self, extracted_features):
        """
        Output data features
        """
        features = []
        # Do output features
        return features

    def _extract(self, feature_name, processed_data):
        feature = None
        # Do extract feature
        return feature

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

