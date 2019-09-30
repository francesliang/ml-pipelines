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
     """

     def process(self, item):
         self.push(item)


class ProcessDataNode(Node):
    """
    """

    def process(self, item):
        self.push(item)


class ExtractFeatureNode(Node):
    """
    """

    def process(self, item):
        print("Extract feature {}".format(self.name))
        self.push(item)


class OutputFeaturesNode(Node):
    """
    """

    def process(self, item):
        self.push(item)


