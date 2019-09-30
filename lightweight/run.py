
from consecution import GlobalState, Node, Pipeline

from lightweight.pipeline import (
    LogNode,
    ImportDataNode,
    ProcessDataNode,
    ExtractFeatureNode,
    OutputFeaturesNode
)


def create_pipeline() -> object:
     """
     Create a pipeline
     """
     global_state = GlobalState(
         data_dir="",
         outputs={},
     )

     pipe = Pipeline(
         LogNode("log")
         | ImportDataNode("import")
         | ProcessDataNode("process")
         | [ExtractFeatureNode('feature-1'), ExtractFeatureNode('feature-2')]
         | OutputFeaturesNode("output"),
         global_state=global_state,
     )

     pipe.plot()

     return pipe


def run_pipeline() -> List:
     """
     Run pipeline
     """
     pipe = create_pipeline()
     pipe.consume([])

     outputs = pipe.global_state["outputs"]
     return outputs


if __name__ == "__main__":
    run_pipeline()
