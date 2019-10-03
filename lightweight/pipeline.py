

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
         self.push(item)


class ProcessDataNode(Node):
    """
    Process data in pipeline.
    """

    def process(self, item):
        self.push(item)


class ExtractFeatureNode(Node):
    """
    Extract features from data.
    """

    def process(self, item):
        print("Extract feature {}".format(self.name))
        self.push(item)


class OutputFeaturesNode(Node):
    """
    Output data features.
    """

    def process(self, item):
        self.push(item)


