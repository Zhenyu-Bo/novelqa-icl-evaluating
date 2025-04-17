import os

class NovelQAPathBuilder:
    """
    构建NovelQA的路径
    """
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def __check_path_exists(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件路径 '{file_path}' 不存在")
        return file_path

    def get_book_path(self, book_id: str) -> str:
        if book_id == 'B30':
            return self.__check_path_exists(f"{self.base_dir}/Demonstration/{book_id}.txt")
        return self.__check_path_exists(f"{self.base_dir}/Books/PublicDomain/{book_id}.txt")
    
    def get_question_path(self, book_id: str) -> str:
        if book_id == 'B30':
            return self.__check_path_exists(f"{self.base_dir}/Demonstration/{book_id}.json")
        return self.__check_path_exists(f"{self.base_dir}/Data/PublicDomain/{book_id}.json")
    
    def get_meta_data_path(self) -> str:
        return self.__check_path_exists(f"{self.base_dir}/bookmeta.json")
