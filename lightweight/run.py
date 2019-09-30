
from consecution import GlobalState, Node, Pipeline

from lightweight.pipeline import LogNode, ImportDataNode


def create_pipeline() -> object:
     """
     Create a pipeline
     """
     global_state = GlobalState(
         data_file="",
         outputs={},
     )

     pipe = Pipeline(
         LogNode("Log processing pipeline")
         | ImportDataNode("Import data")
         | [],
         global_state=global_state,
     )

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
