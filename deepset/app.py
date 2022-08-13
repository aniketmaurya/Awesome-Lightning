from email.policy import default
import logging
from functools import partial

import gradio as gr
import lightning as L
from lightning.app.components.serve import ServeGradio

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING
)
logging.getLogger("haystack").setLevel(logging.INFO)


from haystack.nodes import FARMReader, TransformersReader
from haystack.utils import (
    clean_wiki_text,
    convert_files_to_docs,
    fetch_archive_from_http,
    print_answers,
)


def build_pipeline():
    from haystack.utils import launch_es

    launch_es()

    # Connect to Elasticsearch
    from haystack.document_stores import ElasticsearchDocumentStore

    document_store = ElasticsearchDocumentStore(
        host="localhost", username="", password="", index="document"
    )

    # Let's first fetch some documents that we want to query
    # Here: 517 Wikipedia articles for Game of Thrones
    doc_dir = "data/tutorial1"
    s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt1.zip"
    fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

    # Convert files to dicts
    # You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
    # It must take a str as input, and return a str.
    docs = convert_files_to_docs(
        dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True
    )

    # We now have a list of dictionaries that we can write to our document store.
    # If your texts come from a different source (e.g. a DB), you can of course skip convert_files_to_dicts() and create the dictionaries yourself.
    # The default format here is:
    # {
    #    'content': "<DOCUMENT_TEXT_HERE>",
    #    'meta': {'name': "<DOCUMENT_NAME_HERE>", ...}
    # }
    # (Optionally: you can also add more key-value-pairs here, that will be indexed as fields in Elasticsearch and
    # can be accessed later for filtering or shown in the responses of the Pipeline)

    # Let's have a look at the first 3 entries:
    # print(docs[:3])

    # Now, let's write the dicts containing documents to our DB.
    document_store.write_documents(docs)

    from haystack.nodes import BM25Retriever

    retriever = BM25Retriever(document_store=document_store)

    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

    from haystack.pipelines import ExtractiveQAPipeline

    pipe = ExtractiveQAPipeline(reader, retriever)
    return pipe


class ModelBuildConfig(L.BuildConfig):
    def build_commands(self):
        return ["apt-get install libsndfile1"]


class HaystackDemo(ServeGradio):
    inputs = "text"
    outputs = "text"
    examples = [["Who is the father of Arya Stark?"]]
    default = "Who is the father of Arya Stark?"

    def __init__(self, *args, **kwargs):
        cloud_build_config = ModelBuildConfig(image="elasticsearch")
        local_build_config = L.BuildConfig(image="elasticsearch")
        super().__init__(
            *args,
            **kwargs,
            local_build_config=local_build_config,
            cloud_build_config=cloud_build_config
        )

    def build_model(self):
        return build_pipeline()

    def predict(self, question: str):
        prediction = self.model.run(
            query=question, params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 3}}
        )
        return str(prediction)

    def run(self, *args, **kwargs):
        if self._model is None:
            self._model = self.build_model()
        fn = partial(self.predict, *args, **kwargs)
        fn.__name__ = self.predict.__name__
        gr.Interface(
            fn=fn,
            inputs=self.inputs,
            outputs=self.outputs,
            examples=self.examples,
            title="Game of Thrones QnA System",
        ).launch(
            server_name=self.host,
            server_port=self.port,
            enable_queue=self.enable_queue,
        )


class RootFlow(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.demo = HaystackDemo()

    def run(self):
        self.demo.run()

    def configure_layout(self):
        return {"name": "Game of Thrones QnA System", "content": self.demo.url}


if __name__ == "__main__":
    app = L.LightningApp(RootFlow())
