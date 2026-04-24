from enum import StrEnum

# TODO: move hard coded defaults to here

BM25_K1 = 1.5
BM25_B = 0.75


class MyStrEnum(StrEnum):
    @classmethod
    def values(cls) -> list[str]:
        return [e.value for e in cls]


class EnhanceOptions(MyStrEnum):
    SPELL = "spell"
    REWRITE = "rewrite"
    EXPAND = "expand"


class RerankMethods(MyStrEnum):
    INDIVIDUAL = "individual"
