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


