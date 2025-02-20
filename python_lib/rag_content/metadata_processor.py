# Copyright 2025 Red Hat, Inc.
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

import abc
import logging
from typing import Dict

import requests

LOG = logging.getLogger(__name__)


class MetadataProcessor(object):
    """Metadata processing callback with memory of unreachable URLS.
    FileMetadataProcessor keeps a list of processed files,
    their titles, URLs and if their URLs were reachable.

    Projects should make their own metadata processors.
    Specifically, the `url_function` which is meant to derive URL
    from name of a document, is not implemented.
    """

    def __init__(self):
        # Total number of unreachable documents
        self.total_unreachable_urls = 0

    def get_file_title(sel, file_path: str) -> str:
        """Extract title from the plaintext doc file."""
        title = ""
        try:
            with open(file_path, "r") as file:
                title = file.readline().rstrip("\n").lstrip("# ")
        except Exception:  # noqa: S110
            pass
        return title

    def ping_url(self, url: str) -> bool:
        """Check if the URL parameter is live."""
        try:
            response = requests.get(url, timeout=30)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def populate(self, file_path: str) -> Dict:
        """Populate title and metadata with docs URL.

        Populate the docs_url and title metadata elements with docs URL
        and the page's title.

        Args:
            file_path: str: file path in str
        """
        docs_url = self.url_function(file_path)
        title = self.get_file_title(file_path)

        document = {
            "file_path": file_path,
            "title": title,
            "docs_url": docs_url,
            "unreachable": False,
        }

        if not self.ping_url(docs_url):
            self.total_unreachable_urls += 1
            document["unreachable"] = True

        LOG.debug("Metadata populated for: %s", document)

        return {"docs_url": docs_url, "title": title}

    @abc.abstractmethod
    def url_function(self, file_path: str) -> str:
        """This function must be implemeted in the derived class"""
        raise NotImplementedError
