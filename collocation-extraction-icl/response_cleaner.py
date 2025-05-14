import re

class ResponseCleaner:
    @staticmethod
    def clean(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'Коллокация:\s*', '', text)
        text = re.sub(r'\(коллокация\)', '', text)
        text = re.sub(r'\s*—\s*коллокация со словом.*', '', text)
        text = re.sub(r'[«»]', '', text)
        text = text.lower()
        text = re.sub(r'\.$', '', text)
        return text.strip()
