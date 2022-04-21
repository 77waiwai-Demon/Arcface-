import json

class Config:

    @classmethod
    def from_json_file(cls, file):
        config = Config()
        with open(file, "r", encoding="utf-8") as f:
            config.__dict__ = json.load(f)

        return config

    # def __setitem__(self, key, value):
    #     self.__dict__[key] = value

    def __str__(self):
        return str(self.__dict__)