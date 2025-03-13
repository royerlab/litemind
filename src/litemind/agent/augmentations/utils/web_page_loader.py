import requests
from arbol import asection, aprint
from bs4 import BeautifulSoup

from litemind.agent.augmentations.augmentation_base import Document


class WebPageLoader:
    """
    A utility for loading web pages and converting them to documents.
    """

    @staticmethod
    def load_url(url: str, include_headers: bool = True) -> Document:
        """
        Load a web page and convert it to a document.

        Parameters
        ----------
        url: str
            The URL to load.
        include_headers: bool
            Whether to include headers in the document content.

        Returns
        -------
        Document
            A document representing the web page.
        """
        with asection(f"Loading web page: {url}"):


            # Make the request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"
            }
            response = requests.get(url, headers=headers)

            # Check if the request was successful
            if response.status_code != 200:
                aprint(f"Error fetching page: {response.status_code}")
                return Document(
                    content=f"Error fetching page: {response.status_code}",
                    metadata={"url": url, "error": response.status_code},
                )

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Try to find the page title
            title = "Unknown Title"
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text()

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()

            # Get all the text
            text = soup.get_text()

            # Clean up the text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            # Format the document
            if include_headers:
                content = f"Title: {title}\n"
                content += f"URL: {url}\n\n"
                content += text
            else:
                content = text

            # Create and return the document
            return Document(
                content=content,
                metadata={"title": title, "url": url, "source": "web_page"},
            )

