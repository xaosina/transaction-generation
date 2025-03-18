from abc import ABC, abstractmethod


class BaseDecoder(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def decode(self, data):
        """
        Абстрактный метод для декодирования данных.
        
        :param data: Входные данные для декодирования.
        :return: Декодированный результат.
        """
        pass

