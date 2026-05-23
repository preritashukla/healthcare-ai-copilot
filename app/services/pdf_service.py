import io
import PyPDF2

class PDFService:
    @staticmethod
    def extract_text_from_path(path: str) -> str:
        """Extracts text from a local file path."""
        text = ""
        with open(path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text

    @staticmethod
    def extract_text_from_bytes(content_bytes: bytes) -> str:
        """Extracts text from in-memory bytes (e.g. uploaded files)."""
        text = ""
        pdf_file = io.BytesIO(content_bytes)
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text

pdf_service = PDFService()
