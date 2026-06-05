from pydoctor.templatewriter import TemplateWriter
from typing import Iterable
from pydoctor import model
import os


class CustomTemplateWriter(TemplateWriter):
    def writeIndividualFiles(self, obs: Iterable[model.Documentable]) -> None:
        TemplateWriter.writeIndividualFiles(self, obs)

        bad_tf_url = (
            "https://github.com/GPflow/tensorflow-intersphinx/raw/master/"
        )
        good_tf_url = "https://www.tensorflow.org/api_docs/python/"

        # post-process html files to replace bad tensorflow intersphinx url
        for root, _, files in os.walk(self.build_directory):
            for file in files:
                if not file.endswith(".html"):
                    continue
                with open(os.path.join(root, file), "r") as f:
                    # Do something with the file
                    content = f.read()
                content = content.replace(bad_tf_url, good_tf_url)

                with open(os.path.join(root, file), "w") as f:
                    f.write(content)
