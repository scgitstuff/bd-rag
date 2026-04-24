from dataclasses import dataclass

# TODO: move hard coded defaults to here

BM25_K1 = 1.5
BM25_B = 0.75


# fuck Enums
# I just want a simple namespaced set of constants
# not all the other shit that gets in the way
@dataclass(kw_only=True, frozen=True)
class EnhanceOptions:
    # these are all class variables; works without ClassVar typing
    SPELL = "spell"
    REWRITE = "rewrite"
    EXPAND = "expand"

    # TODO: there should be a less stupid way to do this
    # I tried what was supposed to work, fields() function
    # but it did not work, this is good enough
    @classmethod
    def values(cls) -> list[str]:
        return [cls.SPELL, cls.REWRITE, cls.EXPAND]


@dataclass(kw_only=True, frozen=True)
class RerankMethods:
    INDIVIDUAL = "individual"

    @classmethod
    def values(cls) -> list[str]:
        return [cls.INDIVIDUAL]
