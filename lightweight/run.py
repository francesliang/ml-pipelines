
from consecution import GlobalState, Node, Pipeline

from lightweight.pipeline import (
    LogNode,
    ImportDataNode,
    ProcessDataNode,
    ExtractFeatureNode,
    OutputFeaturesNode
)


feature_list = ['feature_1', 'feature_2', 'feature_3']


def create_pipeline() -> object:
     """
     Create a pipeline
     """
     global_state = GlobalState(
         data_dir="",
         outputs={},
     )

    feature_nodes = []
    for feature in feature_list:
        feature_nodes.append(ExtractFeatureNode(feature))

     pipe = Pipeline(
         LogNode("Log-pipeline")
         | ImportDataNode("Import-data")
         | ProcessDataNode("Process-Data")
         | feature_nodes
         | OutputFeaturesNode("Ouput-features"),
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
