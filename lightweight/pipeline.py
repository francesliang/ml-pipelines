

from consecution import Node


class LogNode(Node):
     """
     Pipeline node to log the start of the pipeline processing.
     """

     def process(self, item):
         print("Start pipeline.")
         self.push(item)


class ImportDataNode(Node):
     """
     Import data to pipeline.
     """

     def process(self, item):
         imported_data = None
         # Do import data
         self.push(imported_data)


class ProcessDataNode(Node):
    """
    Process data in pipeline.
    """

    def process(self, imported_data):
        processed_data = None
        # Do process data
        self.push(processed_data)


class ExtractFeatureNode(Node):
    """
    Extract features from data.
    """

    def _extract(self, processed_data):
        feature = None
        # Do extract feature
        print("Extract feature {}".format(self.name))
        return feature

    def process(self, processed_data):
        feature = self._extract(processed_data)
        self.push(feature)


class OutputFeaturesNode(Node):
    """
    Output data features.
    """

    def process(self, feature):
        features = []
        # Do output features
        self.push(features)


